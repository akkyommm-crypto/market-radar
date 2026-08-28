# -*- coding: utf-8 -*-
"""AI市場レーダー — GitHub Actions ヘッドレス実行版
ノートブック ai_market_radar_v2.ipynb と同一ロジック。
表示系を除き、履歴(score_history.csv / etf_history.csv)はリポジトリ直下に保存。
"""
import os, json, math, time, pickle, warnings, datetime as dt
import csv as _csv
from io import BytesIO, StringIO
import requests
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Actionsでは毎回まっさらな環境なのでキャッシュは使わない
BASE_DIR = "."
CACHE_DIR = "./cache"            # 実行中の一時利用のみ（コミットしない）
OUTPUT_DIR = "."                 # results.json はリポジトリ直下へ
HISTORY_PATH = "./score_history.csv"
ETF_HISTORY_PATH = "./etf_history.csv"
os.makedirs(CACHE_DIR, exist_ok=True)

class _NoDisplay:
    def __call__(self, *a, **k): pass
display = _NoDisplay()
def HTML(x): return x



# ============================================================
# ② 設定（★ここを編集してカスタマイズ★）
# ============================================================


CONFIG = {
    # --- 保存先: TrueならGoogle Driveに履歴・キャッシュを永続化（推奨）---
    "use_drive": True,
    # --- データ取得 ---
    "period": "5y",   # 連動係数の統計を安定させるため5年に拡張
    "batch_size": 300,
    "min_days": 250,  # 5y運用のため最低データ日数も引き上げ
    # --- 流動性フィルタ ---
    "min_turnover_m": 100,     # 20日平均売買代金の下限（百万円）
    "min_price": 100,
    # --- 海外連動 ---
    "impulse_days": 3,         # 米国インパルスを測る営業日数
    "beta_min_corr": 0.25,     # 素の相関で判定する場合の閾値（残差方式OFF時のみ使用）
    # --- 関連銘柄の発見方式 ---
    # 素のリターン同士で相関を取ると、拾えるのは大半が「市場全体の連動」。
    # 米国側からS&P500、日本側からTOPIX相当(全銘柄平均)を引いた残差同士で
    # 測ることで、「たまたま両方とも地合いで動いた」分を除去する。
    "use_residual_beta": True,
    "us_market_factor": "^GSPC",
    "jp_market_factor": "^N225",   # 無ければ全銘柄平均で代用
    # 60ドライバー×4000銘柄=24万通りを総当たりするため、通常の有意水準では
    # 偶然の一致が数千件通ってしまう。検定回数で補正した閾値を自動計算する。
    "resid_alpha": 0.05,
    "beta_min_corr_floor": 0.12,   # 自動計算した閾値の下限
    "radar_top_n": 8,              # レーダータブでドライバーごとに出す銘柄数
    # --- 材料（TDnet）---
    "tdnet_days": 14,
    "tdnet_limit": 5000,
    # --- 業績（第2段階で上位N銘柄だけ取得。多いほど遅い）---
    "fundamental_top_n": 80,
    # --- シグナル判定（🔥新規買い候補の条件。全部明文化＝検証・調整可能）---
    "signal": {
        # 旧設定(min_score=65)は31日間で1回しか成立せず、🔥がほぼ出ない
        # 状態だった。実データの分布（1位の平均45.5点）に合わせて再調整。
        # 絶対点数はあくまで足切り。実質の選別は順位・点火・出来高で行う。
        # 45点では1銘柄も通らなかったため引き下げ済み。
        "min_score": 36,       # 平滑化後スコアの下限
        "max_rank": 15,        # 上位何位までを候補とみなすか
        "min_ignition": 7.0,   # モメンタム点火18点中の最低点
        "min_v_ratio": 1.3,    # 出来高倍率下限
        "max_dev25": 15,       # 25MA乖離の上限(%) 超えたら「押し目待ち」
        "watch_score": 30,     # 「監視」の下限
        "min_days_in_top": 2,  # 上位100に残っている日数（一発屋を除外）
    },
    # --- 出力 ---
    "top_n": 30,
    # --- 重み（100点満点への倍率。合計が変わったら自動で正規化されます）---
    # --- ETFフロー ---
    "etf_flow_scale": 1.0,     # 銘柄別流入$1Mあたりの加点
    # --- モメンタム点火（新設。立ち上がりを捉える中核）---
    "ignition": {
        "fresh_days": 10,   # 転換を「新しい」とみなす営業日数
        "slope_days": 5,    # 傾きを測る営業日数
        "near_high": 0.85,  # 52週高値のこの比率以上なら高値圏
        "max_dev25": 15,    # これ以上伸びていたら点火帯を外れたと判定
    },
    # --- スコア平滑化: 直近N営業日の平均を表示スコアにする ---
    #     生スコアは前日比で平均5.7点動く一方、1位と30位の差は13.8点しか
    #     なかった。平滑化しないとランキングがノイズで毎日入れ替わる。
    #     実履歴で検証: 3日平均で重複15.7→18.6、5日平均で20.8まで改善。
    "smooth_days": 5,
    # 配点を変えたら必ずこの版数も変える。版数の違う履歴は平滑化に混ぜない
    # （旧スコアと新スコアを平均すると意味の無い数字になるため）
    "scoring_version": "v3-ignition",
    "weights": {"linkage": 12, "ignition": 18, "theme": 20, "supply": 10,
                "etf": 5, "tech": 15, "fund": 10, "material": 10},
}

# --- 監視する米国ドライバー（自動発見の探索母集団。広く取るほど発見の網が広がる）---
# 知識ベースで絞り込むのをやめ、AI/半導体/クラウド/防衛/宇宙/EV等の関連銘柄を広く監視し、
# どの銘柄が日本のどの株と実際に連動するかは①βと相関からAIが自動発見する。
US_DRIVERS = [
    # 半導体・装置
    "NVDA","AMD","INTC","MU","TXN","QCOM","AVGO","TSM","ASML","AMAT",
    "LRCX","KLAC","ADI","MRVL","ON","MCHP","SWKS","QRVO","000660.KS",
    # AI・クラウド・データセンター
    "MSFT","GOOGL","META","AMZN","ORCL","SMCI","DELL","VRT","ANET","CSCO",
    "PSTG","NTAP","HPE","IBM",
    # 通信・光学
    "CIEN","COHR","LITE","AAOI",
    # EV・電池・自動車
    "TSLA","GM","F","ALB","LTHM","ON",
    # 防衛・宇宙
    "LMT","RTX","NOC","GD","BA","LHX",
    # 産業・電力・ロボット
    "ETN","HON","ROK","EMR","PWR","GEV",
    # 素材・化学
    "LIN","APD","DD",
    # 指数
    "^SOX","^GSPC","^IXIC","^N225",
]
US_NAMES = {
    "NVDA":"NVIDIA","AMD":"AMD","INTC":"Intel","MU":"Micron","TXN":"Texas Instruments",
    "QCOM":"Qualcomm","AVGO":"Broadcom","TSM":"TSMC","ASML":"ASML","AMAT":"Applied Materials",
    "LRCX":"Lam Research","KLAC":"KLA","ADI":"Analog Devices","MRVL":"Marvell","ON":"onsemi",
    "MCHP":"Microchip","SWKS":"Skyworks","QRVO":"Qorvo","000660.KS":"SK hynix",
    "MSFT":"Microsoft","GOOGL":"Google","META":"Meta","AMZN":"Amazon","ORCL":"Oracle",
    "SMCI":"Supermicro","DELL":"Dell","VRT":"Vertiv","ANET":"Arista","CSCO":"Cisco",
    "PSTG":"Pure Storage","NTAP":"NetApp","HPE":"HPE","IBM":"IBM",
    "CIEN":"Ciena","COHR":"Coherent","LITE":"Lumentum","AAOI":"Applied Optoelectronics",
    "TSLA":"Tesla","GM":"GM","F":"Ford","ALB":"Albemarle","LTHM":"Livent",
    "LMT":"Lockheed Martin","RTX":"RTX","NOC":"Northrop Grumman","GD":"General Dynamics",
    "BA":"Boeing","LHX":"L3Harris","ETN":"Eaton","HON":"Honeywell","ROK":"Rockwell",
    "EMR":"Emerson","PWR":"Quanta Services","GEV":"GE Vernova","LIN":"Linde",
    "APD":"Air Products","DD":"DuPont","^SOX":"SOX指数","^GSPC":"S&P500","^IXIC":"NASDAQ",
    "^N225":"日経平均",
}
# 韓国市場は日本と同時間帯 → 翌日ではなく同日の反応を見る
SAME_DAY_DRIVERS = {"000660.KS"}

# --- テーマ辞書（★自分の注目銘柄を追加★）---
THEMES = {
    "HBM・後工程":        ["6146", "6857", "6315", "7729", "4062"],
    "半導体製造装置":      ["8035", "6920", "7735", "6323", "6871"],
    "光通信・電線":        ["5803", "5801", "5802", "5805", "4980"],
    "AIデータセンター・電力": ["6501", "6503", "6508", "6641", "6504", "1969", "1942", "1944"],
    "防衛":               ["7011", "7012", "7013"],
    "ロボット":            ["6954", "6506", "6324"],
    "生成AI・ソフト":      ["3993", "3655", "4488"],
    "宇宙":               ["9348", "7011"],
    "核融合":             ["5310", "7711", "7011"],
    "量子":               ["6702", "6501"],
}

# --- TDnet 材料キーワードと配点 ---
TDNET_KEYWORDS = {
    "上方修正": 8, "業績予想の修正": 5, "増配": 6,
    "自己株式の取得": 6, "自己株式取得": 6,
    "資本業務提携": 6, "業務提携": 5, "資本提携": 5,
    "大型受注": 6, "受注": 4, "大型契約": 5,
    "新工場": 5, "設備投資": 4, "量産": 4, "増産": 4,
    "買収": 5, "子会社化": 4, "株式分割": 4, "新製品": 3,
    "データセンター": 4, "半導体": 3, "AI": 3,
}

# --- US上場・日本株ETF（フロー監視対象）---
ETF_LIST = ["EWJ", "BBJP", "FLJP", "JPXN", "EWJV",
            "DXJ", "HEWJ", "DBJP", "DFJ", "SCJ", "OPPJ", "FJP"]
ETF_HEDGED = {"DXJ", "HEWJ", "DBJP"}   # 為替ヘッジ型
# iSharesの商品ID。保有銘柄CSVは先頭に発行口数・純資産・基準日が
# 入っているので、口数の取得元としてもこのCSVを使う。
# yfinanceのsharesOutstandingはETFでは更新されない（毎日同じ値を返す）ため、
# 差分が常にゼロになりフローが出なかった。
# ※IDが変わったら商品ページの「Download Holdings」のリンクから取り直す。
#   確証が無いIDは候補を並べておけば、順に試して通ったものを使う。
ISHARES_PRODUCTS = {
    "EWJ":  ["239665"],
    "SCJ":  ["239666", "239627"],
    "JPXN": ["239831"],
    "EWJV": ["307263"],
    "HEWJ": ["259624"],
}

def ishares_url(product_id, ticker):
    return ("https://www.ishares.com/us/products/" + product_id +
            "/x/1467271812596.ajax?fileType=csv"
            "&fileName=" + ticker + "_holdings&dataType=fund")

# 保有銘柄の配分に使うETF（上のIDから自動生成）
HOLDINGS_SOURCES = {t: ishares_url(p[0], t) for t, p in ISHARES_PRODUCTS.items()}

# GitHub ActionsはUTCで動くため、そのままだと日本時間の朝6時実行が
# 前日の日付で記録される（=履歴に日曜が並ぶ）。JSTで固定する。
JST = dt.timezone(dt.timedelta(hours=9))
TODAY = dt.datetime.now(JST).date()
UA = {"User-Agent": "Mozilla/5.0 (ai-market-radar; personal use)"}


# ============================================================
# ④ 東証全銘柄リストの取得（JPX公式）
# ============================================================
def get_ticker_list():
    cache_file = os.path.join(CACHE_DIR, f"tickers_{TODAY:%Y%m%d}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        df = pd.read_excel(BytesIO(r.content), dtype=str)
    except Exception as e:
        raise RuntimeError(
            "JPX銘柄一覧の取得に失敗。URL変更の可能性があります。\n"
            "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html で"
            "data_j.xls の最新URLを確認してください。\n" + f"エラー: {e}"
        )

    targets = ["プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）"]
    df = df[df["市場・商品区分"].isin(targets)].copy()
    df["code"] = df["コード"].astype(str).str.strip()
    out = df[["code", "銘柄名", "市場・商品区分", "33業種区分"]].rename(
        columns={"銘柄名": "name", "市場・商品区分": "market", "33業種区分": "sector"}
    ).reset_index(drop=True)
    with open(cache_file, "wb") as f:
        pickle.dump(out, f)
    return out

tickers_df = get_ticker_list()
NAME_MAP = dict(zip(tickers_df["code"], tickers_df["name"]))
SECTOR_MAP = dict(zip(tickers_df["code"], tickers_df["sector"]))
print(f"対象銘柄数: {len(tickers_df)}")


# ============================================================
# ⑤ 全銘柄の株価取得（初回10〜20分 / 同日2回目以降はキャッシュ）
# ============================================================
def download_prices(codes):
    cache_file = os.path.join(CACHE_DIR, f"prices_{TODAY:%Y%m%d}.pkl")
    if os.path.exists(cache_file):
        print("キャッシュから読み込み中...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    prices = {}
    n_chunks = (len(codes) - 1) // CONFIG["batch_size"] + 1
    for i in range(0, len(codes), CONFIG["batch_size"]):
        chunk = codes[i : i + CONFIG["batch_size"]]
        symbols = [c + ".T" for c in chunk]
        print(f"  取得中 {i // CONFIG['batch_size'] + 1}/{n_chunks} "
              f"({len(prices)}銘柄完了)", end="\r")
        try:
            raw = yf.download(symbols, period=CONFIG["period"], group_by="ticker",
                              threads=True, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"\n  バッチ取得失敗（スキップして続行）: {e}")
            time.sleep(3)
            continue
        for c in chunk:
            sym = c + ".T"
            try:
                d = raw if len(symbols) == 1 else raw[sym]
                d = d.dropna(subset=["Close"])
                if len(d) >= CONFIG["min_days"]:
                    prices[c] = d[["Open", "High", "Low", "Close", "Volume"]].copy()
            except Exception:
                pass
        time.sleep(1)

    print(f"\n取得完了: {len(prices)}銘柄")
    with open(cache_file, "wb") as f:
        pickle.dump(prices, f)
    return prices

prices = download_prices(list(tickers_df["code"]))

# --- 全銘柄の終値行列（β計算・テーマ強度で使う）---
close_df = pd.DataFrame({c: df["Close"] for c, df in prices.items()})
close_df.index = pd.to_datetime(close_df.index).tz_localize(None).normalize()
jp_ret = close_df.pct_change()
print(f"終値行列: {close_df.shape[0]}日 × {close_df.shape[1]}銘柄")


# ============================================================
# ⑥ 米国ドライバーの取得とインパルス（直近の動き）計算
# ============================================================
def fetch_us_drivers():
    cache_file = os.path.join(CACHE_DIR, f"us_{TODAY:%Y%m%d}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    us_close = {}
    try:
        raw = yf.download(US_DRIVERS, period=CONFIG["period"], group_by="ticker",
                          threads=True, progress=False, auto_adjust=True)
        for sym in US_DRIVERS:
            try:
                c = (raw[sym] if len(US_DRIVERS) > 1 else raw)["Close"].dropna()
                c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
                if len(c) >= 60:
                    us_close[sym] = c
            except Exception:
                pass
    except Exception as e:
        print(f"⚠ 米国株取得失敗（海外連動スコアは0で続行）: {e}")
    if us_close:   # 取得失敗（空）はキャッシュしない
        with open(cache_file, "wb") as f:
            pickle.dump(us_close, f)
    return us_close

us_close = fetch_us_drivers()
N_IMP = CONFIG["impulse_days"]
us_impulse = {}   # 直近N営業日のリターン
for sym, c in us_close.items():
    if len(c) > N_IMP:
        us_impulse[sym] = float(c.iloc[-1] / c.iloc[-1 - N_IMP] - 1)

print(f"米国ドライバー（直近{N_IMP}営業日）:")
for sym, imp in sorted(us_impulse.items(), key=lambda x: -x[1]):
    mark = "🔥" if imp >= 0.03 else ("↑" if imp > 0 else "↓")
    print(f"  {mark} {US_NAMES.get(sym, sym):<11} {imp*100:+5.1f}%")


# ============================================================
# ⑦ 連動係数βの自動発見（過去5年の実データ・知識ベース不使用）
#    米国ドライバーの日次リターン → 日本株の翌営業日リターン の回帰係数
#    ※事前にリストアップした「これは連動するはず」という思い込みは使わない。
#      約60銘柄の米国ドライバー × 全日本株の組み合わせを機械的に総当たりし、
#      相関が閾値を超えたペアだけを「発見された連動」として採用する。
# ============================================================
def _residualize(Y, f):
    """YからファクターfのOLS成分を除いた残差を返す（YはSeriesでもDataFrameでも可）"""
    common = Y.index.intersection(f.index)
    Y = Y.loc[common]
    fc = f.loc[common].astype(float)
    fc = fc - fc.mean()
    d = float((fc ** 2).sum())
    Yc = Y.sub(Y.mean())
    if d <= 0:
        return Yc
    if isinstance(Y, pd.Series):
        return Yc - fc * (float((Yc * fc).sum()) / d)
    b = Yc.mul(fc, axis=0).sum() / d
    return Yc - pd.DataFrame(np.outer(fc.values, b.values),
                             index=Yc.index, columns=Yc.columns)


def _inv_norm(p):
    """標準正規分布の逆累積分布（Acklamの近似）。scipyを使わずに済ませる"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p <= 0 or p >= 1:
        return 0.0
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def corr_threshold(n_obs, n_tests):
    """多重検定を補正した相関の採用下限。

    24万ペアを総当たりするので、素朴に「相関0.25以上」とすると
    偶然そう見えるだけのペアが大量に混ざる。検定回数で割った
    有意水準から必要な相関を逆算する。
    """
    if n_obs < 30:
        return 1.0
    alpha = CONFIG.get("resid_alpha", 0.05) / max(int(n_tests), 1)
    z = abs(_inv_norm(alpha / 2))
    thr = z / math.sqrt(max(n_obs - 3, 1))
    return float(min(0.95, max(CONFIG.get("beta_min_corr_floor", 0.12), thr)))


def compute_betas():
    betas, corrs = {}, {}
    jp_next = jp_ret.shift(-1)   # 日本株の「翌営業日リターン」
    use_res = bool(CONFIG.get("use_residual_beta", True))
    mkt_sym = CONFIG.get("us_market_factor", "^GSPC")
    jp_sym = CONFIG.get("jp_market_factor", "^N225")
    us_mkt = us_close.get(mkt_sym)
    us_mkt_ret = us_mkt.pct_change().dropna() if us_mkt is not None else None
    # 日本側の市場要因。日経平均が取れていればそれを使い、
    # 取れなければ全銘柄の等ウェイト平均で代用する。
    jp_mkt = us_close.get(jp_sym)
    if jp_mkt is not None and len(jp_mkt) > 250:
        jp_mkt_ret = jp_mkt.pct_change().dropna()
        jp_src = jp_sym
    else:
        jp_mkt_ret = jp_ret.mean(axis=1)
        jp_src = "全銘柄平均"
    n_tests = max(1, len(us_close) * max(1, jp_ret.shape[1]))
    thr_used = []

    for sym, c in us_close.items():
        if use_res and sym in (mkt_sym, jp_sym):
            continue   # 市場要因そのものはドライバーにしない（残差がゼロになる）
        r = c.pct_change().dropna()
        same_day = sym in SAME_DAY_DRIVERS
        target = jp_ret if same_day else jp_next
        common = r.index.intersection(target.index)
        if len(common) < 250:   # 5年運用なので十分なサンプル数を要求
            continue
        x = r.loc[common]
        Y = target.loc[common]
        if use_res:
            if us_mkt_ret is not None:
                x = _residualize(x, us_mkt_ret.reindex(common).fillna(0.0))
            # 翌日リターンを見ている場合、除去すべき市場要因も翌日のもの
            jm = jp_mkt_ret if same_day else jp_mkt_ret.shift(-1)
            Y = _residualize(Y, jm.reindex(common).fillna(0.0))
            thr = corr_threshold(len(common), n_tests)
        else:
            thr = CONFIG["beta_min_corr"]
        thr_used.append(thr)
        xc = x - x.mean()
        denom = float((xc ** 2).sum())
        if denom <= 0:
            continue
        beta = Y.sub(Y.mean()).mul(xc, axis=0).sum() / denom
        corr = Y.corrwith(x)
        # 閾値未満のペアはノイズとして0扱い
        beta = beta.where(corr.abs() >= thr, 0.0)
        betas[sym] = beta.clip(-0.3, 2.0).fillna(0.0)
        corrs[sym] = corr.fillna(0.0)

    if thr_used:
        mode = "残差(市場要因を除去)" if use_res else "素のリターン"
        n_pair = int(sum((b != 0).sum() for b in betas.values()))
        print(f"  連動判定: {mode} / 相関閾値 {min(thr_used):.3f} "
              f"/ 採用ペア {n_pair:,}件 (総当たり {n_tests:,}通り)")
        if use_res:
            print(f"  除去した市場要因: 米国={mkt_sym} 日本={jp_src}")
    return betas, corrs

betas, corrs = compute_betas()
print(f"β自動発見 完了: {len(betas)}本の米国ドライバーで計算 (サンプル{CONFIG['period']}分)")

# --- 発見結果: ドライバーごとに最も連動が強い日本株TOP5を一覧表示 ---
print("\n[自動発見] 各ドライバーの連動が強い日本株 TOP5（相関降順）:")
discovered = []
for sym, beta in betas.items():
    c = corrs.get(sym)
    if c is None:
        continue
    top = c.abs().sort_values(ascending=False).head(5)
    for code, ac in top.items():
        if ac < CONFIG["beta_min_corr"]:
            continue
        discovered.append({"driver": sym, "code": code,
                           "beta": float(beta.get(code, 0)), "corr": float(c.get(code, 0))})
disc_df = pd.DataFrame(discovered)
if len(disc_df):
    for sym, g in disc_df.groupby("driver"):
        g = g.sort_values("corr", ascending=False, key=abs).head(5)
        line = " / ".join(f"{r.code}{NAME_MAP.get(r.code,'')}(β{r.beta:.2f} r{r.corr:.2f})"
                          for r in g.itertuples())
        print(f"  {US_NAMES.get(sym,sym):<16} {line}")
else:
    print("  （5年データが揃っていないためこの実行では未発見。翌回以降に安定します）")


# ============================================================
# ⑧ 市場環境ダッシュボード（スイングで最初に見るべき「地合い」）
#    順張りモメンタムは地合いが悪い時にやらないのが最大の防御
# ============================================================
def fetch_series(sym, period="6mo"):
    try:
        raw = yf.download(sym, period=period, progress=False, auto_adjust=True)
        c = raw["Close"].squeeze().dropna()
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        return c if len(c) > 30 else None
    except Exception:
        return None

n225   = fetch_series("^N225")
topixp = fetch_series("1305.T")     # TOPIX代用ETF（1306.Tは分割で履歴破損のため使わない）
usdjpy = fetch_series("USDJPY=X")
vix    = fetch_series("^VIX")

# --- 市場全体の20日リターン（個別銘柄のRS計算に使う）---
base = n225 if n225 is not None else topixp
MKT_R20 = float(base.iloc[-1] / base.iloc[-21] - 1) if base is not None and len(base) > 21 else 0.0
# --- 市場の20日リターン「系列」: 個別銘柄のRS20の推移を出すのに使う ---
MKT_R20_SER = base.pct_change(20) if base is not None else None

# --- 市場の幅（ブレッドス）: 全銘柄データから直接計算 ---
ma25_all = close_df.rolling(25).mean().iloc[-1]
breadth = float((close_df.iloc[-1] > ma25_all).mean() * 100)
new_high = int((close_df.iloc[-1] >= close_df.iloc[:-1].max()).sum())
last_ret = jp_ret.iloc[-1]
adv, dec = int((last_ret > 0).sum()), int((last_ret < 0).sum())

# --- 信号判定 ---
lights = []   # (項目, 表示値, 色, コメント)
def light(name, val, color, note):
    lights.append((name, val, color, note))

if base is not None:
    ma25b = base.rolling(25).mean()
    up = float(base.iloc[-1]) > float(ma25b.iloc[-1])
    rising = float(ma25b.iloc[-1]) > float(ma25b.iloc[-6])
    col = "green" if (up and rising) else ("yellow" if up else "red")
    light("地合い(日経vs25MA)", f"{float(base.iloc[-1]):,.0f}",
          col, "上昇トレンド" if col == "green" else ("持ち合い" if col == "yellow" else "調整中"))

col = "green" if breadth >= 55 else ("yellow" if breadth >= 40 else "red")
light("市場の幅(25MA上%)", f"{breadth:.0f}%", col,
      f"新高値{new_high}銘柄 / 騰落 {adv}:{dec}")

if vix is not None:
    v = float(vix.iloc[-1])
    col = "green" if v < 18 else ("yellow" if v < 25 else "red")
    light("VIX(米リスク)", f"{v:.1f}", col,
          "リスクオン" if col == "green" else ("警戒" if col == "yellow" else "リスクオフ⚠"))

if usdjpy is not None and len(usdjpy) > 6:
    chg5 = float(usdjpy.iloc[-1] / usdjpy.iloc[-6] - 1) * 100
    col = "green" if chg5 > -1.5 else ("yellow" if chg5 > -3 else "red")
    light("ドル円(5日)", f"{float(usdjpy.iloc[-1]):.1f} ({chg5:+.1f}%)", col,
          "急激な円高⚠" if col != "green" else "安定")

sox = us_close.get("^SOX")
if sox is not None and len(sox) > 6:
    s5 = float(sox.iloc[-1] / sox.iloc[-6] - 1) * 100
    col = "green" if s5 > 0 else ("yellow" if s5 > -3 else "red")
    light("SOX指数(5日)", f"{s5:+.1f}%", col, "半導体テーマの追い風" if col == "green" else "半導体テーマ逆風")

greens = sum(1 for l in lights if l[2] == "green")
reds = sum(1 for l in lights if l[2] == "red")
if reds >= 2:
    REGIME, REGIME_LABEL = "red", "🔴 新規エントリー見送り推奨（守りの局面）"
elif greens >= max(3, len(lights) - 1):
    REGIME, REGIME_LABEL = "green", "🟢 押し目買い・ブレイク買いに適した地合い"
else:
    REGIME, REGIME_LABEL = "yellow", "🟡 銘柄を選別。ロットを落として"

cmap = {"green": "#7fdc7f", "yellow": "#ffd700", "red": "#ff6b6b"}
rows = "".join(
    f'<tr><td style="color:#aaa;padding:4px 10px;">{n}</td>'
    f'<td style="color:{cmap[c]};font-weight:bold;padding:4px 10px;">●</td>'
    f'<td style="color:#fff;padding:4px 10px;">{v}</td>'
    f'<td style="color:#888;padding:4px 10px;">{note}</td></tr>'
    for n, v, c, note in lights)
display(HTML(
    f'<div style="background:#0e0e16;padding:16px;border-radius:12px;font-family:sans-serif;">'
    f'<h2 style="color:#7fdcff;margin:0 0 4px;">🌡 市場環境</h2>'
    f'<p style="color:#ddd;font-size:1.05em;margin:4px 0 10px;">{REGIME_LABEL}</p>'
    f'<table style="border-collapse:collapse;">{rows}</table></div>'))


# ============================================================
# ⑨ 🇺🇸 海外イベント・未反応度スコア（配点30の中核）
#    期待反応 = β × 米国インパルス
#    未反応ギャップ = 期待反応 − 実際の日本株リターン
#    「連動するはずなのにまだ動いていない」ほど高得点
# ============================================================
def build_linkage_scores():
    """code → dict(pts, driver, expected, actual, gap, reason)"""
    W = CONFIG["weights"]["linkage"]
    out = {}
    if not betas:
        return out

    # 日本株の直近リターン（米国インパルスと同じ日数+1日で反応を観測）
    n = CONFIG["impulse_days"] + 1
    valid = close_df.iloc[-1 - n:]
    jp_recent = (valid.iloc[-1] / valid.iloc[0] - 1).fillna(0.0)

    for sym, imp in us_impulse.items():
        if imp < 0.01:        # 米国側が1%以上動いたイベントのみ対象
            continue
        beta = betas.get(sym)
        if beta is None:
            continue
        expected = beta * imp                     # 期待反応（Series）
        cand = expected[expected >= 0.01].index    # 期待1%以上の銘柄のみ
        for code in cand:
            exp_r = float(expected[code])
            act_r = float(jp_recent.get(code, 0.0))
            gap = exp_r - act_r
            # 連動の強さ = 期待反応の大きさ。W(12点)を上限に配点。
            # ※旧版の「未反応ボーナス」は廃止。まだ動いていない銘柄ほど
            #   高得点になる設計はモメンタム狙いと逆を向くため。
            #   連動はエントリー理由の裏付けとして使い、点火はignitionで測る。
            pts = min(exp_r * 400, float(W))
            cur = out.get(code)
            if cur is None or pts > cur["pts"]:
                out[code] = {
                    "pts": pts, "driver": US_NAMES.get(sym, sym),
                    "expected": exp_r, "actual": act_r, "gap": gap,
                    "reason": (f"{US_NAMES.get(sym,sym)}{imp*100:+.1f}%連動"
                               f"(β{float(beta[code]):.2f})"
                               + (f" 実反応{act_r*100:+.1f}%" if abs(act_r) > 0.005 else "")),
                }
    return out

linkage_scores = build_linkage_scores()
print(f"海外連動の対象銘柄: {len(linkage_scores)}")
unre = sorted(linkage_scores.items(), key=lambda x: -x[1]["gap"])[:10]
print("\n[未反応ギャップ TOP10（期待反応−実反応）]")
for code, d in unre:
    print(f"  {code} {NAME_MAP.get(code,''):<14} {d['driver']:<10} "
          f"期待{d['expected']*100:+.1f}% 実際{d['actual']*100:+.1f}% → ギャップ{d['gap']*100:+.1f}%")


# ============================================================
# ⑩ 材料スコア: TDnet適時開示（10点）
#    ※やのしん氏の非公式API。負荷をかけない常識的な利用を。
# ============================================================
def fetch_tdnet():
    cache_file = os.path.join(CACHE_DIR, f"tdnet_{TODAY:%Y%m%d}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    start = TODAY - dt.timedelta(days=CONFIG["tdnet_days"])
    url = (f"https://webapi.yanoshin.jp/webapi/tdnet/list/"
           f"{start:%Y%m%d}-{TODAY:%Y%m%d}.json?limit={CONFIG['tdnet_limit']}")
    disclosures = {}
    try:
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        for it in r.json().get("items", []):
            td = it.get("Tdnet", {})
            code = str(td.get("company_code", "")).strip()
            if len(code) == 5 and code.endswith("0"):
                code = code[:4]
            title = td.get("title", "")
            pub = str(td.get("pubdate", ""))[:10]
            if code and title:
                disclosures.setdefault(code, []).append((pub, title))
        print(f"適時開示 取得: 対象企業 {len(disclosures)}社")
    except Exception as e:
        print(f"⚠ TDnet取得失敗（材料スコア0で続行）: {e}")
    if disclosures:   # 取得失敗（空）はキャッシュしない
        with open(cache_file, "wb") as f:
            pickle.dump(disclosures, f)
    return disclosures

def score_material(code):
    W = CONFIG["weights"]["material"]
    pts, reasons = 0.0, []
    for pub, title in disclosures.get(code, []):
        try:
            days_ago = (TODAY - dt.date.fromisoformat(pub)).days
        except Exception:
            days_ago = CONFIG["tdnet_days"]
        decay = max(0.0, 1 - days_ago / CONFIG["tdnet_days"])
        best_p, best_kw = 0, None
        for kw, p in TDNET_KEYWORDS.items():
            if kw in title and p > best_p:
                best_p, best_kw = p, kw
        if best_kw:
            pts += best_p * decay * (W / 20)   # v1は20点満点設計→重みに合わせ縮尺
            reasons.append(f"開示[{best_kw}]({days_ago}日前)")
    return min(pts, W), reasons[:2]

disclosures = fetch_tdnet()


# ============================================================
# ⑪ テーマスコア（20点）: テーマ辞書 × テーマの今の強さ
# ============================================================
def build_theme_scores():
    W = CONFIG["weights"]["theme"]
    theme_strength = {}
    for theme, codes in THEMES.items():
        rets = []
        for c in codes:
            if c in close_df.columns and close_df[c].dropna().shape[0] > 6:
                s = close_df[c].dropna()
                rets.append(float(s.iloc[-1] / s.iloc[-6] - 1))
        if rets:
            theme_strength[theme] = float(np.mean(rets))

    scores = {}
    for theme, codes in THEMES.items():
        s = theme_strength.get(theme, 0.0)
        pts = min(W * 0.3 + max(0.0, s) * 200 * (W / 20), W)
        for c in codes:
            old = scores.get(c, (0.0, []))
            if pts > old[0]:
                scores[c] = (pts, [f"{theme}({s*100:+.1f}%)"])
    return scores, theme_strength

theme_scores, theme_strength = build_theme_scores()
print("テーマ強度（メンバー5日平均リターン）:")
for t, s in sorted(theme_strength.items(), key=lambda x: -x[1]):
    print(f"  {t:<18} {s*100:+5.1f}% " + "█" * max(0, int(s * 200)))


# ============================================================
# ⑫ テクニカル（15点）と 需給=出来高分析（15点）
# ============================================================
def calc_rsi(close, n=14):
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(n).mean()
    loss = (-diff.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def score_technical(df):
    """テクニカル15点: MA順列5 / 52週高値5 / RSI 2.5 / MACD 2.5"""
    W = CONFIG["weights"]["tech"]
    u = W / 15.0   # 重み変更に追従する縮尺
    pts, reasons = 0.0, []
    c = df["Close"]
    close = float(c.iloc[-1])
    ma25 = c.rolling(25).mean().iloc[-1]
    ma75 = c.rolling(75).mean().iloc[-1]
    ma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan

    if close > ma25 > ma75:
        pts += 3 * u; reasons.append("MA上昇配列")
        if not np.isnan(ma200) and ma75 > ma200:
            pts += 2 * u; reasons.append("パーフェクトオーダー")
    elif close > ma25:
        pts += 1 * u

    hi52 = float(df["High"].iloc[:-1].max()) if len(df) > 1 else close
    rhi = close / hi52 if hi52 > 0 else 0
    if rhi >= 1.0:   pts += 5 * u; reasons.append("52週高値ブレイク!")
    elif rhi >= 0.97: pts += 3.5 * u; reasons.append("52週高値目前")
    elif rhi >= 0.90: pts += 2 * u
    elif rhi >= 0.80: pts += 1 * u

    rsi = float(calc_rsi(c).iloc[-1])
    if 55 <= rsi <= 75: pts += 2.5 * u
    elif 45 <= rsi < 55 or 75 < rsi <= 85: pts += 1.2 * u
    if rsi > 85: reasons.append(f"RSI過熱({rsi:.0f})⚠")

    ema12 = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26; sig = macd.ewm(span=9).mean()
    if macd.iloc[-1] > sig.iloc[-1]:
        pts += (2.5 if macd.iloc[-1] > 0 else 1.5) * u

    return min(pts, W), reasons, {"close": close, "rsi": rsi, "rhi": rhi}

def score_supply(df):
    """需給(出来高)15点: 出来高急増7 / 買い出来高比率5 / 売買代金トレンド3"""
    W = CONFIG["weights"]["supply"]
    u = W / 15.0
    pts, reasons = 0.0, []
    v = df["Volume"]

    v20 = v.rolling(20).mean().iloc[-2] if len(v) > 21 else np.nan
    v_ratio = float(v.iloc[-1] / v20) if v20 and v20 > 0 else 0
    if v_ratio >= 3:   pts += 7 * u; reasons.append(f"出来高急増{v_ratio:.1f}倍")
    elif v_ratio >= 2: pts += 5 * u; reasons.append(f"出来高増{v_ratio:.1f}倍")
    elif v_ratio >= 1.5: pts += 3 * u
    elif v_ratio >= 1.2: pts += 1.5 * u

    tail = df.tail(20)
    up = tail["Close"] > tail["Close"].shift(1)
    tot = float(tail["Volume"].sum())
    up_ratio = float(tail.loc[up, "Volume"].sum() / tot) if tot > 0 else 0
    if up_ratio >= 0.65: pts += 5 * u; reasons.append("買い出来高優勢")
    elif up_ratio >= 0.55: pts += 3 * u
    elif up_ratio >= 0.50: pts += 1.5 * u

    to = (df["Close"] * df["Volume"])
    to20 = float(to.tail(20).mean()); to60 = float(to.tail(60).mean())
    if to60 > 0 and to20 / to60 >= 1.5:
        pts += 3 * u; reasons.append("資金流入増")
    elif to60 > 0 and to20 / to60 >= 1.2:
        pts += 1.5 * u

    return min(pts, W), reasons, {"v_ratio": v_ratio, "turnover_m": to20 / 1e6}

def rs20_series(df):
    """対市場の20日相対リターン(%)の系列"""
    r20 = df["Close"].pct_change(20)
    if MKT_R20_SER is None:
        return r20 * 100
    m = MKT_R20_SER.reindex(r20.index).ffill()
    return (r20 - m) * 100


def _days_since_cross_up(ser, level=0.0):
    """最後にlevel以下だったのが何営業日前か。
    今がlevel以下なら-1（まだ転換していない）。ずっと上ならデータ長。"""
    s = ser.dropna()
    if len(s) == 0 or float(s.iloc[-1]) <= level:
        return -1
    below = np.where(s.values <= level)[0]
    if len(below) == 0:
        return len(s)
    # 転換した当日を0日目とする
    return int(len(s) - 2 - below[-1])


def score_ignition(df, t_info):
    """モメンタム点火18点: RS転換6 / RS傾き4 / 25MA転換4 / 高値圏かつ未過熱4

    「まだ動いていない」ではなく「動き始めた」ものに配点する。
    水準ではなく変化を見るので、日々のブレも小さい。
    """
    W = CONFIG["weights"]["ignition"]
    u = W / 18.0
    ig = CONFIG["ignition"]
    pts, reasons = 0.0, []
    c = df["Close"]

    # (1) RS20がマイナスからプラスに転換した直後か。浅いほど高評価
    rs = rs20_series(df)
    rs_val = rs.dropna()
    rs_now = float(rs_val.iloc[-1]) if len(rs_val) else 0.0
    d_rs = _days_since_cross_up(rs)
    if d_rs >= 0:
        if d_rs <= ig["fresh_days"]:
            pts += 6 * u; reasons.append(f"RS転換{d_rs}日目")
        elif d_rs <= ig["fresh_days"] * 3:
            pts += 3 * u
        else:
            pts += 1.5 * u

    # (2) RSの傾き。強さの水準ではなく、強くなっている最中かを見る
    k = int(ig["slope_days"])
    rs_prev = rs_now
    if len(rs_val) > k:
        rs_prev = float(rs_val.iloc[-1 - k])
    rs_slope = rs_now - rs_prev
    if rs_slope > 3:
        pts += 4 * u; reasons.append(f"RS加速{rs_slope:+.1f}")
    elif rs_slope > 0:
        pts += 2 * u

    # (3) 25MAを上抜けた直後で、かつ25MA自体が上向きに転じているか
    ma25 = c.rolling(25).mean()
    d_ma = _days_since_cross_up(c - ma25)
    ma_val = ma25.dropna()
    ma_up = len(ma_val) > k and float(ma_val.iloc[-1]) > float(ma_val.iloc[-1 - k])
    if d_ma >= 0 and ma_up and d_ma <= ig["fresh_days"]:
        pts += 4 * u; reasons.append(f"25MA上抜け{d_ma}日目")
    elif d_ma >= 0 and ma_up:
        pts += 2 * u

    # (4) 高値圏だが伸びきっていない帯。一番おいしいところ
    rhi = float(t_info.get("rhi", 0.0))
    dev25 = 0.0
    if len(ma_val) and float(ma_val.iloc[-1]) > 0:
        dev25 = (float(c.iloc[-1]) / float(ma_val.iloc[-1]) - 1) * 100
    # ※市場に勝っていること(rs_now>0)を条件に入れないと、ただ高値圏で
    #   動いていないだけの銘柄に点が入ってしまう
    if rhi >= ig["near_high"] and dev25 < ig["max_dev25"] and rs_now > 0:
        pts += 4 * u; reasons.append("高値圏かつ未過熱")
    elif rhi >= ig["near_high"] and rs_now > 0:
        pts += 1 * u

    return min(pts, W), reasons, {"rs_now": rs_now, "rs_slope": rs_slope,
                                  "rs_days": d_rs, "ma_days": d_ma}


print("採点関数 定義OK")


# ============================================================
# ⑬ ETFフロー分析: US上場・日本株ETFの資金流出入（需給5点）
#    フロー = 発行口数の変化 × 価格
#    ※AUM変化と違い株価変動が混ざらない純粋な資金の出入り
#    ※口数を毎回スナップショット保存 → 2回目以降の実行でフローが出ます
# ============================================================
def _num(v):
    if v is None:
        return None
    v = str(v).replace(",", "").replace("$", "").replace('"', "").strip()
    if v in ("", "-", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_ishares_header(text):
    """保有銘柄CSVの先頭メタ行から発行口数・純資産・基準日を取り出す"""
    out = {"shares": None, "tna": None, "as_of": None, "header_line": None}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().lower().startswith("ticker,"):
            out["header_line"] = i
            break
        try:
            row = next(_csv.reader([line]))
        except Exception:
            continue
        if len(row) < 2:
            continue
        key, val = row[0].strip().lower(), row[1].strip()
        if key == "shares outstanding":
            out["shares"] = _num(val)
        elif key == "total net assets":
            out["tna"] = _num(val)
        elif key.startswith("fund holdings as of"):
            out["as_of"] = val
    return out


def ishares_fund_stats(ticker):
    """iShares公式CSVから発行口数を取る。取れなければNone"""
    for pid in ISHARES_PRODUCTS.get(ticker, []):
        url = ishares_url(pid, ticker)
        try:
            r = requests.get(url, headers=UA, timeout=45)
        except Exception as e:
            print(f"  {ticker}: 接続エラー {type(e).__name__}")
            continue
        if r.status_code != 200 or len(r.content) < 1000:
            print(f"  {ticker}: HTTP {r.status_code} / {len(r.content)}バイト "
                  f"(商品ID {pid} が古い可能性)")
            continue
        h = parse_ishares_header(r.content.decode("utf-8-sig", errors="replace"))
        if h["shares"]:
            HOLDINGS_SOURCES[ticker] = url   # 通ったURLを保有銘柄取得にも使う
            return h
        print(f"  {ticker}: CSVは取れたが口数の行が見つからない（書式変更の疑い）")
    return None


def etf_prices():
    """終値だけをまとめて取得（.infoは不安定なので使わない）"""
    try:
        px = yf.download(ETF_LIST, period="5d", progress=False,
                         auto_adjust=False, threads=True)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame()
        return {t: float(px[t].dropna().iloc[-1])
                for t in px.columns if px[t].dropna().size}
    except Exception as e:
        print(f"  ⚠ ETF価格の取得に失敗: {e}")
        return {}


def snapshot_etfs():
    print("ETF口数の取得:")
    px = etf_prices()
    rows = []
    for t in ETF_LIST:
        price = px.get(t)
        sh = aum = None
        as_of = str(TODAY)
        src = ""
        st = ishares_fund_stats(t) if t in ISHARES_PRODUCTS else None
        if st:
            sh, aum, src = st["shares"], st["tna"], "ishares"
            if st["as_of"]:
                as_of = st["as_of"]
        else:
            # iShares以外（DXJ/BBJPなど）はyfinanceに頼るしかない。
            # 値が動かないことが多いので参考程度。
            try:
                info = yf.Ticker(t).info or {}
                sh = info.get("sharesOutstanding")
                aum = info.get("totalAssets")
                price = price or info.get("previousClose")
                src = "yfinance"
                time.sleep(0.1)
            except Exception:
                pass
        if sh and price:
            rows.append({"date": str(TODAY), "as_of": as_of, "ticker": t,
                         "shares": float(sh), "price": float(price),
                         "aum": float(aum or sh * price), "src": src})
    if not rows:
        print("⚠ ETFデータ取得失敗（ETFフローは0点で続行）")
        return None
    n_is = sum(1 for r in rows if r["src"] == "ishares")
    print(f"  取得できたのは {len(rows)}/{len(ETF_LIST)}本 "
          f"(iShares公式 {n_is}本 / yfinance {len(rows)-n_is}本)")
    new = pd.DataFrame(rows)
    if os.path.exists(ETF_HISTORY_PATH):
        h = pd.read_csv(ETF_HISTORY_PATH)
        if "as_of" not in h.columns:
            h["as_of"] = h["date"]
        if "src" not in h.columns:
            h["src"] = "yfinance"
        # 同じ基準日のデータは上書き。基準日が変わって初めて新しい行になるので、
        # 「同じ数字が何日も並んで差分ゼロ」という状態を防げる。
        key = set(zip(new["ticker"], new["as_of"]))
        h = h[[(t, a) not in key for t, a in zip(h["ticker"], h["as_of"])]]
        h = pd.concat([h, new], ignore_index=True)
    else:
        h = new
    h.to_csv(ETF_HISTORY_PATH, index=False)
    print(f"ETF口数スナップショット保存: 累計{len(h)}行 → {ETF_HISTORY_PATH}")
    return h

def calc_etf_flows(h):
    """直近2スナップショット間の推定フロー（百万ドル）"""
    flows = {}
    if h is None:
        return flows
    col = "as_of" if "as_of" in h.columns else "date"
    stale = []
    for t, g in h.groupby("ticker"):
        g = g.drop_duplicates(subset=[col], keep="last").copy()
        g["_d"] = pd.to_datetime(g[col], errors="coerce")
        g = g.sort_values("_d")
        if len(g) < 2:
            continue
        prev, now = g.iloc[-2], g.iloc[-1]
        if not prev["shares"]:
            continue
        d_sh = now["shares"] - prev["shares"]
        if d_sh == 0:
            stale.append(t)
        days = (now["_d"] - prev["_d"]).days if pd.notna(now["_d"]) and pd.notna(prev["_d"]) else 1
        flows[t] = {
            "flow_musd": d_sh * now["price"] / 1e6,
            "d_sh_pct": d_sh / prev["shares"] * 100,
            "days": max(int(days), 1),
            "aum_musd": now["aum"] / 1e6,
        }
    if stale:
        print(f"  ※口数が前回から変化していないETF: {', '.join(stale)}")
    return flows

etf_hist = snapshot_etfs()
etf_flows = calc_etf_flows(etf_hist)

if not etf_flows:
    print("ETFフロー: スナップショット蓄積中（次回の実行から表示・加点されます）")
else:
    print(f"\n{'ETF':<6}{'AUM($M)':>9}{'口数変化':>9}{'フロー($M)':>11}  種別（{list(etf_flows.values())[0]['days']}日間）")
    for t, f in sorted(etf_flows.items(), key=lambda x: -x[1]["flow_musd"]):
        kind = "ヘッジ" if t in ETF_HEDGED else "無ヘッジ"
        print(f"{t:<6}{f['aum_musd']:>9,.0f}{f['d_sh_pct']:>8.2f}%{f['flow_musd']:>11,.1f}  {kind}")
    hedged = sum(f["flow_musd"] for t, f in etf_flows.items() if t in ETF_HEDGED)
    unhedged = sum(f["flow_musd"] for t, f in etf_flows.items() if t not in ETF_HEDGED)
    print(f"\nヘッジ型合計 {hedged:+,.0f}M$ / 無ヘッジ型合計 {unhedged:+,.0f}M$")
    if hedged > 0 and hedged > max(unhedged, 0) * 2:
        print("→ 円リスクを避けて日本株を買う海外勢のサイン🔥")

# --- 保有銘柄CSV → 銘柄別にフローを配分 ---
def fetch_holdings(ticker, url):
    """iShares形式の保有銘柄CSVを取得 → {証券コード: 組入比率%}"""
    try:
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        lines = r.text.splitlines()
        head = next(i for i, l in enumerate(lines)
                    if "Ticker" in l and "Weight" in l)
        from io import StringIO
        df = pd.read_csv(StringIO("\n".join(lines[head:])))
        w_col = next(c for c in df.columns if "Weight" in c)
        out = {}
        for _, row in df.iterrows():
            tk = str(row.get("Ticker", "")).strip().split()[0] if pd.notna(row.get("Ticker")) else ""
            if len(tk) == 4 and tk[0].isdigit() and tk in NAME_MAP:
                try:
                    out[tk] = out.get(tk, 0.0) + float(str(row[w_col]).replace(",", ""))
                except Exception:
                    pass
        print(f"  {ticker}: 保有{len(out)}銘柄をマッピング")
        return out
    except Exception as e:
        print(f"  ⚠ {ticker} 保有銘柄CSV取得失敗（HOLDINGS_SOURCESのURLを確認）: {e}")
        return {}

stock_flow_musd = {}
if etf_flows:
    for t, url in HOLDINGS_SOURCES.items():
        f = etf_flows.get(t)
        if not f or abs(f["flow_musd"]) < 0.5:   # 微小フローは無視
            continue
        for code, w in fetch_holdings(t, url).items():
            stock_flow_musd[code] = stock_flow_musd.get(code, 0.0) + f["flow_musd"] * w / 100

def build_etf_scores():
    W = CONFIG["weights"]["etf"]
    scores = {}
    for code, fl in stock_flow_musd.items():
        if fl <= 0:
            continue
        pts = min(W, fl * CONFIG["etf_flow_scale"])
        if pts >= 0.3:
            scores[code] = (pts, [f"ETF流入+${fl:.1f}M"])
    return scores

etf_scores = build_etf_scores()
if etf_scores:
    print(f"\n銘柄別ETFフロー加点: {len(etf_scores)}銘柄")
    for code, (p, rs) in sorted(etf_scores.items(), key=lambda x: -x[1][0])[:5]:
        print(f"  {code} {NAME_MAP.get(code,''):<14} {rs[0]} → +{p:.1f}点")


# ============================================================
# ⑭ 第1段階: 全銘柄採点 → 第2段階: 上位のみ業績取得（10点）
# ============================================================
def screen_stage1():
    rows = []
    n = len(prices)
    for i, (code, df) in enumerate(prices.items()):
        if i % 500 == 0:
            print(f"  採点中 {i}/{n}", end="\r")
        try:
            close = float(df["Close"].iloc[-1])
            if close < CONFIG["min_price"]:
                continue
            t_pts, t_rs, t_info = score_technical(df)
            s_pts, s_rs, s_info = score_supply(df)
            i_pts, i_rs, i_info = score_ignition(df, t_info)
            c_ser = df["Close"]
            # RS: 市場(日経)に対する20日相対リターン。モメンタムの本丸
            r20 = float(c_ser.iloc[-1] / c_ser.iloc[-21] - 1) if len(c_ser) > 21 else 0.0
            rs20 = (r20 - MKT_R20) * 100
            # ATR(14): ボラティリティ → 損切り目安 = 終値 - 2ATR
            tr = pd.concat([df["High"] - df["Low"],
                            (df["High"] - c_ser.shift()).abs(),
                            (df["Low"] - c_ser.shift()).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            stop = close - 2 * atr
            # 25MA乖離率: 伸びすぎは平均回帰リスク → 過熱ペナルティ
            ma25v = float(c_ser.rolling(25).mean().iloc[-1])
            dev25 = (close / ma25v - 1) if ma25v > 0 else 0.0
            penalty = 0.0
            if dev25 > 0.20:
                penalty = -5.0; t_rs = t_rs + [f"過熱⚠25MA+{dev25*100:.0f}%"]
            elif dev25 > 0.15:
                penalty = -3.0; t_rs = t_rs + [f"伸びすぎ注意(25MA+{dev25*100:.0f}%)"]
            if s_info["turnover_m"] < CONFIG["min_turnover_m"]:
                continue
            mat_pts, mat_rs = score_material(code)
            th_pts, th_rs = theme_scores.get(code, (0.0, []))
            e_pts, e_rs = etf_scores.get(code, (0.0, []))
            lk = linkage_scores.get(code)
            lk_pts = lk["pts"] if lk else 0.0
            lk_rs = [lk["reason"]] if lk else []

            rows.append({
                "code": code, "name": NAME_MAP.get(code, ""),
                "sector": SECTOR_MAP.get(code, ""),
                "linkage": round(lk_pts, 1), "ignition": round(i_pts, 1),
                "rs_days": i_info["rs_days"], "rs_slope": round(i_info["rs_slope"], 1),
                "theme": round(th_pts, 1),
                "supply": round(s_pts, 1), "etf": round(e_pts, 1),
                "tech": round(t_pts, 1), "penalty": round(penalty, 1),
                "rs20": round(rs20, 1), "atr_pct": round(atr / close * 100, 1),
                "dev25": round(dev25 * 100, 1), "stop": round(stop),
                "material": round(mat_pts, 1), "fund": 0.0,
                "driver": lk["driver"] if lk else "",
                "gap_pct": round(lk["gap"] * 100, 1) if lk else 0.0,
                "close": close,
                "chg5d": round(float(df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100, 1)
                         if len(df) > 6 else 0.0,
                "v_ratio": round(s_info["v_ratio"], 1),
                "turnover_m": round(s_info["turnover_m"]),
                "reasons": i_rs + lk_rs + th_rs + e_rs + s_rs + t_rs + mat_rs,
            })
        except Exception:
            continue
    df_out = pd.DataFrame(rows)
    df_out["score"] = (df_out["linkage"] + df_out["ignition"] + df_out["theme"]
                       + df_out["supply"] + df_out["etf"] + df_out["tech"]
                       + df_out["material"] + df_out["penalty"])
    df_out = df_out.sort_values("score", ascending=False).reset_index(drop=True)
    print(f"\n第1段階 完了: {len(df_out)}銘柄")
    return df_out

def add_fundamentals(df_out):
    """上位N銘柄だけyfinanceで業績を取得して加点（成長率/ROE/利益率/PER）"""
    W = CONFIG["weights"]["fund"]
    n = min(CONFIG["fundamental_top_n"], len(df_out))
    cache_file = os.path.join(CACHE_DIR, f"fund_{TODAY:%Y%m%d}.pkl")
    fund_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            fund_cache = pickle.load(f)

    print(f"第2段階: 上位{n}銘柄の業績を取得中（1〜2分）...")
    for i in range(n):
        code = df_out.at[i, "code"]
        if code in fund_cache:
            info = fund_cache[code]
        else:
            try:
                info = yf.Ticker(code + ".T").info or {}
                time.sleep(0.15)
            except Exception:
                info = {}
            fund_cache[code] = info
        pts, rs = 0.0, []
        try:
            rg = info.get("revenueGrowth")
            if rg is not None:
                if rg >= 0.15: pts += 3; rs.append(f"増収+{rg*100:.0f}%")
                elif rg >= 0.05: pts += 1.5
            roe = info.get("returnOnEquity")
            if roe is not None:
                if roe >= 0.15: pts += 2; rs.append(f"ROE{roe*100:.0f}%")
                elif roe >= 0.08: pts += 1
            om = info.get("operatingMargins")
            if om is not None:
                if om >= 0.15: pts += 2
                elif om >= 0.08: pts += 1
            pe = info.get("trailingPE")
            if pe is not None and 0 < pe <= 25:
                pts += 3; rs.append(f"PER{pe:.0f}倍")
            elif pe is not None and pe <= 40:
                pts += 1.5
        except Exception:
            pass
        pts = min(pts * (W / 10), W)
        df_out.at[i, "fund"] = round(pts, 1)
        if rs:
            df_out.at[i, "reasons"] = df_out.at[i, "reasons"] + rs

    with open(cache_file, "wb") as f:
        pickle.dump(fund_cache, f)
    df_out["score"] = (df_out["linkage"] + df_out["ignition"] + df_out["theme"]
                       + df_out["supply"] + df_out["etf"] + df_out["tech"]
                       + df_out["material"] + df_out["fund"] + df_out["penalty"])
    return df_out.sort_values("score", ascending=False).reset_index(drop=True)

def apply_smoothing(df_out):
    """直近N営業日の生スコアの平均を『表示スコア』にする。

    生スコアは同じ銘柄でも前日比で平均5.7点動くため、平滑化しないと
    上位30の半分が毎朝入れ替わる。数日〜2か月保有する使い方と噛み合わない。
    履歴の無い銘柄は当日値をそのまま使う（新顔を不利にしない）。
    """
    n = int(CONFIG.get("smooth_days", 1))
    df_out["score_raw"] = df_out["score"]
    df_out["days_in_top"] = 1
    if n <= 1 or not os.path.exists(HISTORY_PATH):
        df_out["rank"] = range(1, len(df_out) + 1)
        return df_out
    try:
        h = pd.read_csv(HISTORY_PATH, dtype={"code": str})
    except Exception:
        df_out["rank"] = range(1, len(df_out) + 1)
        return df_out
    h = h[h["date"] != str(TODAY)]
    # 配点が違う時期の履歴を平均に混ぜると数字の意味が壊れる
    ver = CONFIG.get("scoring_version", "")
    if "sv" in h.columns:
        h = h[h["sv"].astype(str) == ver]
    else:
        h = h.iloc[0:0]
    if h.empty:
        print("  平滑化: 同じ配点の履歴がまだ無いので当日値をそのまま使用")
        df_out["rank"] = range(1, len(df_out) + 1)
        return df_out
    if "score_raw" in h.columns:
        h["score_raw"] = h["score_raw"].fillna(h["score"])
    else:
        h["score_raw"] = h["score"]
    keep = sorted(h["date"].unique())[-(n - 1):]
    h = h[h["date"].isin(keep)]
    prev = h.groupby("code")["score_raw"].apply(list).to_dict()
    sm, dit = [], []
    for r in df_out.itertuples():
        past = [float(x) for x in prev.get(str(r.code), []) if pd.notna(x)]
        vals = [float(r.score_raw)] + past
        sm.append(round(sum(vals) / len(vals), 1))
        dit.append(1 + len(past))
    df_out["score"] = sm
    df_out["days_in_top"] = dit
    df_out = df_out.sort_values("score", ascending=False).reset_index(drop=True)
    df_out["rank"] = range(1, len(df_out) + 1)
    return df_out


def judge_signal(row):
    """ルールベースの売買候補判定。条件はCONFIG['signal']で全て明文化。
    ※候補の分類であって売買推奨ではない。最終判断は人間が行う。"""
    sg = CONFIG["signal"]
    if REGIME == "red":
        return "見送り(地合い🔴)"
    strong = ((row["score"] >= sg["min_score"]) and (row["rs20"] > 0)
              and (row["rank"] <= sg["max_rank"]))
    hot = row["dev25"] > sg["max_dev25"]
    if strong and hot:
        return "押し目待ち"
    if (strong and row["ignition"] >= sg["min_ignition"]
            and row["v_ratio"] >= sg["min_v_ratio"]
            and row["days_in_top"] >= sg["min_days_in_top"]):
        return "🔥新規買い候補"
    if strong:
        return "監視(強)"
    if row["score"] >= sg["watch_score"]:
        return "監視"
    return ""

def stars(score):
    # 新配点での実測分布（1位40.8点・15位34.5点）に合わせて再調整。
    # ETFフローが0点のままなので、修理後にもう一度見直すこと。
    if score >= 40: return "★★★★★"
    if score >= 36: return "★★★★☆"
    if score >= 32: return "★★★☆☆"
    if score >= 28: return "★★☆☆☆"
    return "★☆☆☆☆"

result = screen_stage1()
result = add_fundamentals(result)
result = apply_smoothing(result)
result["stars"] = result["score"].apply(stars)
result["signal"] = result.apply(judge_signal, axis=1)
result["reasons_str"] = result["reasons"].apply(lambda r: " / ".join(r) if r else "-")
n_fire = int((result["signal"] == "🔥新規買い候補").sum())
n_dip = int((result["signal"] == "押し目待ち").sum())
print(f"採点完了 ／ 🔥新規買い候補: {n_fire}銘柄 ／ 押し目待ち: {n_dip}銘柄")


def signal_funnel(df, sg):
    """シグナルが0件のときに、どの条件で落ちているかを特定するための内訳。
    次回以降の閾値調整はこの出力を見て決める。"""
    print("\n--- シグナル条件の通過状況 ---")
    if df.empty:
        print("  銘柄がありません")
        return
    q = df["score"]
    print(f"  スコア: 1位 {q.iloc[0]:.1f} / 5位 {q.iloc[min(4,len(q)-1)]:.1f} "
          f"/ 15位 {q.iloc[min(14,len(q)-1)]:.1f} / 30位 {q.iloc[min(29,len(q)-1)]:.1f}")
    ign = df["ignition"]
    print(f"  点火点: 最高 {ign.max():.1f} / 中央 {ign.median():.1f} "
          f"(満点 {CONFIG['weights']['ignition']})")
    top = df.head(50)
    conds = [
        (f"スコア>={sg['min_score']}",      top["score"] >= sg["min_score"]),
        (f"順位<={sg['max_rank']}",         top["rank"] <= sg["max_rank"]),
        ("RS20>0",                          top["rs20"] > 0),
        (f"点火>={sg['min_ignition']}",     top["ignition"] >= sg["min_ignition"]),
        (f"出来高>={sg['min_v_ratio']}倍",  top["v_ratio"] >= sg["min_v_ratio"]),
        (f"乖離<={sg['max_dev25']}%",       top["dev25"] <= sg["max_dev25"]),
        (f"継続>={sg['min_days_in_top']}日", top["days_in_top"] >= sg["min_days_in_top"]),
    ]
    for lab, m in conds:
        print(f"    {lab:<18} 上位50中 {int(m.sum()):>2}銘柄が通過")
    both = conds[0][1]
    for _, m in conds[1:]:
        both = both & m
    print(f"    → 全条件クリア    {int(both.sum())}銘柄")
    if int(both.sum()) == 0:
        worst = min(conds, key=lambda cm: int(cm[1].sum()))
        print(f"    ボトルネック: 「{worst[0]}」が最も厳しい")


signal_funnel(result, CONFIG["signal"])


# ============================================================
# ⑰ スコア履歴の蓄積 と 予測力の検証
#    「スコア○点以上はN日後に平均何%動いたか」を実データで確認
# ============================================================
# --- 本日の上位100件を履歴に追記（同日重複は上書き）---
# 平滑化前の生スコアも残す（次回の平滑化と、後からの検証に使う）
hist_new = result.head(100)[["code", "score", "score_raw", "gap_pct",
                             "ignition", "rs20", "rs_days"]].copy()
hist_new["sv"] = CONFIG.get("scoring_version", "")
hist_new.insert(0, "date", str(TODAY))
if os.path.exists(HISTORY_PATH):
    hist = pd.read_csv(HISTORY_PATH, dtype={"code": str})
    hist = hist[hist["date"] != str(TODAY)]
    hist = pd.concat([hist, hist_new], ignore_index=True)
else:
    hist = hist_new
hist.to_csv(HISTORY_PATH, index=False)
print(f"履歴に追記: 累計{len(hist)}行 / {hist['date'].nunique()}日分 → {HISTORY_PATH}")

# --- 検証: 過去のスコアと その後のリターン ---
def validate_history():
    past = hist[hist["date"] < str(TODAY)]
    if past.empty:
        print("\n（履歴がまだ1日分のため検証はスキップ。毎朝実行すると貯まります）")
        return
    recs = []
    for r in past.itertuples():
        s = close_df.get(r.code)
        if s is None:
            continue
        s = s.dropna()
        d = pd.Timestamp(r.date)
        idx = s.index.searchsorted(d)
        if idx >= len(s):
            continue
        base = float(s.iloc[idx])
        row = {"score": r.score}
        for lab, n in [("1日後", 1), ("5日後", 5), ("20日後", 20)]:
            if idx + n < len(s):
                row[lab] = (float(s.iloc[idx + n]) / base - 1) * 100
        recs.append(row)
    if not recs:
        print("\n（検証可能なデータがまだありません）")
        return
    v = pd.DataFrame(recs)
    v["帯"] = pd.cut(v["score"], [0, 30, 38, 45, 101],
                     labels=["〜30", "30-38", "38-45", "45+"])
    print("\n=== スコア帯別の平均リターン（%）===")
    print(v.groupby("帯")[[c for c in ["1日後", "5日後", "20日後"] if c in v]]
           .agg(["mean", "count"]).round(2).to_string())

validate_history()


# ============================================================
# ⑱ 結果の保存（スマホWEBアプリ用JSON + CSV）
# ============================================================
# 関連銘柄は「スコア上位に入ったか」ではなく「過去のチャートで連動が
# 確認できたか」で選ぶ。相関の強い順に並べ、現在のスコアを併記する。
radar_json = []
_by = {c: r for c, r in zip(result["code"], result.to_dict("records"))}
for sym, imp in sorted(us_impulse.items(), key=lambda x: -x[1]):
    if imp < 0.01:
        continue
    dname = US_NAMES.get(sym, sym)
    cr, bt = corrs.get(sym), betas.get(sym)
    if cr is None or bt is None:
        continue
    # β採用済み（＝相関が閾値を超えた）銘柄のうち、スクリーニングを
    # 通過しているものだけを対象にする
    cand = cr[(bt != 0)]
    cand = cand[[c in _by for c in cand.index]]
    if cand.empty:
        continue
    cand = cand.reindex(cand.abs().sort_values(ascending=False).index)
    kids = []
    for code, cv in cand.head(CONFIG["radar_top_n"]).items():
        rec = _by[code]
        kids.append({"code": code, "name": rec.get("name", ""),
                     "score": rec.get("score", 0.0),
                     "stars": rec.get("stars", ""),
                     "signal": rec.get("signal", ""),
                     "ignition": rec.get("ignition", 0.0),
                     "rs20": rec.get("rs20", 0.0),
                     "corr": round(float(cv), 2),
                     "beta": round(float(bt.get(code, 0.0)), 2),
                     "gap_pct": rec.get("gap_pct", 0.0)})
    if not kids:
        continue
    radar_json.append({
        "driver": dname, "impulse_pct": round(imp * 100, 1),
        "n_linked": int((bt != 0).sum()),
        "stocks": kids,
    })

export_cols = ["code", "name", "sector", "score", "score_raw", "days_in_top",
               "rank", "stars", "linkage", "ignition", "rs_days", "rs_slope", "theme",
               "supply", "etf", "tech", "fund", "material", "penalty", "signal", "driver",
               "gap_pct", "rs20", "atr_pct", "dev25", "stop",
               "close", "chg5d", "v_ratio", "turnover_m", "reasons_str"]
out = {
    "version": "v2",
    "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    "weights": CONFIG["weights"],
    "us_drivers": {US_NAMES.get(k, k): round(v * 100, 2) for k, v in us_impulse.items()},
    "theme_strength": {k: round(v * 100, 2) for k, v in theme_strength.items()},
    "etf_flows": {t: round(f["flow_musd"], 1) for t, f in etf_flows.items()},
    "signal_conditions": CONFIG["signal"],
    "market_env": {"regime": REGIME, "label": REGIME_LABEL, "mkt_r20": round(MKT_R20 * 100, 2),
                   "breadth": round(breadth, 1), "new_high": new_high,
                   "lights": [{"name": n, "value": v, "color": c, "note": note}
                              for n, v, c, note in lights]},
    "radar": radar_json,
    "stocks": result.head(100)[export_cols].to_dict(orient="records"),
}

# --- スコア推移データ: 履歴CSVから直近30営業日分を銘柄別に整形 ---
history_series = {}
prev_codes = []
delta_map = {}
try:
    if os.path.exists(HISTORY_PATH):
        _h = pd.read_csv(HISTORY_PATH, dtype={"code": str})
        _dates = sorted(_h["date"].unique())
        _recent_dates = _dates[-30:]
        # 今回出力する上位100銘柄だけ推移を持たせる（JSONサイズ節約）
        _top_codes = set(result.head(100)["code"])
        for code, g in _h[_h["code"].isin(_top_codes)].groupby("code"):
            g = g[g["date"].isin(_recent_dates)].sort_values("date")
            if len(g) >= 2:
                history_series[code] = [
                    {"date": r.date, "score": round(float(r.score), 1),
                     "gap": round(float(getattr(r, "gap_pct", 0) or 0), 1)}
                    for r in g.itertuples()]
        # 前日との比較（新規ランクイン・スコア変化バッジ用）
        if len(_dates) >= 2:
            _prev = _h[_h["date"] == _dates[-2]]
            prev_codes = _prev["code"].tolist()
            _prev_score = dict(zip(_prev["code"], _prev["score"]))
            for code in _top_codes:
                if code in _prev_score:
                    delta_map[code] = round(
                        float(result.loc[result["code"] == code, "score"].iloc[0])
                        - float(_prev_score[code]), 1)
except Exception as _e:
    print(f"（履歴データの整形をスキップ: {_e}）")

out["history"] = history_series
out["prev_codes"] = prev_codes
out["deltas"] = delta_map
print(f"履歴付与: 推移{len(history_series)}銘柄 / 前日比較{len(delta_map)}銘柄")
json_path = os.path.join(OUTPUT_DIR, "results.json")
csv_path = os.path.join(OUTPUT_DIR, f"screening_{TODAY:%Y%m%d}.csv")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
result[export_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
print("保存しました:")
print(" ", json_path, "← WEBアプリ用（GitHub Pagesに置く想定）")
print(" ", csv_path)