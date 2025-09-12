# app.py
# Bybit PERPETUALS (USDT-linear) – Trades, PnL, MA-Stop & MA-(TP/Partial TP, CLOSE-only) + Winrate
# + Konto-saldo & Positionsstorlek (risk i %), jämförelse, exit-markörer + chart, auto-optimering (inkl. Multi-TP),
# projektion, Plotly-fallback, WIN/LOSS-COUNTS

import time, hmac, hashlib
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlencode
from dataclasses import dataclass

import requests, pandas as pd, numpy as np, streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ====== FYLL I DINA NYCKLAR (Bybit v5 – helst Unified Account) ======
API_KEY    = "Klgy68UfhEnZtrYicm"   # låtsasnyckel
API_SECRET = "LPfJxAploIR3MAhFvSsDUfqvudXi6MhmsoYX"  # låtsashemlighet

# ====== KONFIG ======
BYBIT_BASE = "https://api.bybit.com"
RECV_WINDOW = "5000"

# ====== INIT STATE ======
if "trades_df" not in st.session_state: st.session_state["trades_df"] = None
if "opt_open" not in st.session_state: st.session_state["opt_open"] = False
if "opt_results" not in st.session_state: st.session_state["opt_results"] = None
if "opt_stop_periods" not in st.session_state: st.session_state["opt_stop_periods"] = [10, 20, 50]
if "opt_stop_tfs" not in st.session_state: st.session_state["opt_stop_tfs"] = ["5m", "10m", "15m", "60m"]
if "opt_tp_periods" not in st.session_state: st.session_state["opt_tp_periods"] = [10, 20, 50]
if "opt_tp_tfs" not in st.session_state: st.session_state["opt_tp_tfs"] = ["5m", "10m", "15m", "60m"]
if "allow_after_default" not in st.session_state: st.session_state["allow_after_default"] = True
if "acct_balance" not in st.session_state: st.session_state["acct_balance"] = None
if "opt_multi_tp_scenarios" not in st.session_state:
    # default scenarion som testas i auto-optimeringen om Multi-TP-opt är på
    st.session_state["opt_multi_tp_scenarios"] = [
        # ( [pct1,pct2,pct3], [mode1,mode2,mode3], [(per,tf,profit_first) eller None för ORIGINAL] )
        ([100,0,0], ["MA","MA","MA"], [(50,"15m",True), None, None]),
        ([50,50,0], ["MA","ORIGINAL","MA"], [(50,"15m",True), None, None]),
        ([33,33,34], ["MA","MA","ORIGINAL"], [(50,"15m",True),(10,"10m",True), None]),
        ([25,25,50], ["MA","MA","MA"], [(50,"15m",True),(20,"15m",True),(10,"10m",True)]),
        ([70,30,0], ["ORIGINAL","MA","MA"], [None,(50,"15m",True), (10,"10m",True)]),
    ]

# ====== SIGNERING / HTTP ======
def _bybit_headers(api_key: str, api_secret: str, query_str: str = "", body_str: str = "") -> Dict[str, str]:
    ts = str(int(time.time() * 1000))
    payload = ts + api_key + RECV_WINDOW + (body_str if body_str else query_str)
    sign = hmac.new(api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return {"X-BAPI-API-KEY": api_key, "X-BAPI-TIMESTAMP": ts, "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "X-BAPI-SIGN": sign, "Content-Type": "application/json"}

def bybit_get(endpoint: str, params: Dict[str, Any], api_key: Optional[str]=None, api_secret: Optional[str]=None) -> Dict[str, Any]:
    encoded_query = urlencode(sorted(params.items()))
    headers = _bybit_headers(api_key, api_secret, query_str=encoded_query) if api_key and api_secret else None
    url = f"{BYBIT_BASE}{endpoint}?{encoded_query}"
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

# ====== SALDO ======
def fetch_wallet_balance(api_key: str, api_secret: str, account_type: str = "UNIFIED", coin: Optional[str] = "USDT") -> Dict[str, Any]:
    params: Dict[str, Any] = {"accountType": account_type}
    if coin: params["coin"] = coin
    data = bybit_get("/v5/account/wallet-balance", params, api_key, api_secret)
    if str(data.get("retCode")) != "0": raise RuntimeError(f"Bybit API fel: {data.get('retCode')} - {data.get('retMsg')}")
    result = data.get("result") or {}; lst = result.get("list") or []
    if not lst: return {}
    acct = lst[0]; coins = acct.get("coin") or []
    if coin and coins:
        c = coins[0]; avail = c.get("availableToWithdraw") or c.get("availableBalance")
        return {"accountType": acct.get("accountType"), "coin": c.get("coin"),
                "equity": float(c.get("equity", 0)), "walletBalance": float(c.get("walletBalance", 0)),
                "available": float(avail or 0), "unrealisedPnl": float(c.get("unrealisedPnl", 0)),
                "cumRealisedPnl": float(c.get("cumRealisedPnl", 0)), "totalMarginBalance": float(c.get("totalMarginBalance", 0))}
    out = {"accountType": acct.get("accountType"), "coins": []}
    for c in coins:
        avail = c.get("availableToWithdraw") or c.get("availableBalance")
        out["coins"].append({"coin": c.get("coin"), "equity": float(c.get("equity", 0)),
                             "walletBalance": float(c.get("walletBalance", 0)),
                             "available": float(avail or 0),
                             "unrealisedPnl": float(c.get("unrealisedPnl", 0))})
    return out

# ====== POSITION SIZE & R/R ======
def calc_position_size(side: str, entry: float, stop: float, equity: float, risk_pct: float,
                       fee_buffer_pct: float = 0.10, leverage: float = 1.0, available: Optional[float] = None) -> Dict[str, float]:
    side = (side or "").lower().strip()
    if entry <= 0 or stop <= 0 or equity <= 0 or risk_pct <= 0:
        return {"qty": 0.0, "notional": 0.0, "init_margin": 0.0, "risk_amount_usdt": 0.0, "capped": 0.0}
    price_risk = abs(entry - stop)
    if price_risk <= 0:
        return {"qty": 0.0, "notional": 0.0, "init_margin": 0.0, "risk_amount_usdt": 0.0, "capped": 0.0}
    risk_amount = equity * (risk_pct / 100.0)
    effective_risk = risk_amount * (1.0 - (fee_buffer_pct / 100.0)) or risk_amount
    qty = effective_risk / price_risk
    notional = qty * entry
    lev = max(1.0, float(leverage))
    init_margin = notional / lev
    capped = 0.0
    if available and available > 0:
        max_notional = float(available) * lev
        if notional > max_notional:
            qty = (max_notional / entry); capped = 1.0
            notional = qty * entry; init_margin = notional / lev
    return {"qty": float(qty), "notional": float(notional), "init_margin": float(init_margin),
            "risk_amount_usdt": float(risk_amount), "capped": float(capped)}

def calc_rr(side: str, entry: float, stop: float, tp: Optional[float]) -> Dict[str, float]:
    if entry <= 0 or stop <= 0: return {"risk_per_unit": 0.0, "reward_per_unit": 0.0, "rr": 0.0}
    risk_per_unit = abs(entry - stop)
    reward_per_unit = 0.0
    if tp and tp > 0:
        if str(side).lower() == "long": reward_per_unit = max(0.0, tp - entry)
        else: reward_per_unit = max(0.0, entry - tp)
    rr = (reward_per_unit / risk_per_unit) if risk_per_unit > 0 else 0.0
    return {"risk_per_unit": float(risk_per_unit), "reward_per_unit": float(reward_per_unit), "rr": float(rr)}

# ====== HJÄLP ======
def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        else: df[col] = pd.NA
    return df

def timeframe_to_ms(tf: str) -> int:
    return {"5m": 5*60*1000, "10m":10*60*1000, "15m":15*60*1000, "30m":30*60*1000, "60m":60*60*1000}[tf]

# ====== HÄMTA PERPETUAL-TRADES ======
def fetch_all_linear_trades(api_key: str, api_secret: str, start_time_ms: int | None, end_time_ms: int | None) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []; cursor = None; safety_loops = 0
    while True:
        params: Dict[str, Any] = {"category": "linear", "limit": 200}
        if cursor: params["cursor"] = cursor
        if start_time_ms: params["startTime"] = start_time_ms
        if end_time_ms: params["endTime"] = end_time_ms
        data = bybit_get("/v5/execution/list", params, api_key, api_secret)
        if str(data.get("retCode")) != "0": raise RuntimeError(f"Bybit API fel: {data.get('retCode')} - {data.get('retMsg')}")
        result = data.get("result") or {}; rows = result.get("list") or []
        trades.extend(rows); cursor = result.get("nextPageCursor")
        if not cursor: break
        safety_loops += 1
        if safety_loops > 500: break
        time.sleep(0.15)
    return trades

# ====== KLINES (public) ======
@st.cache_data(show_spinner=False, ttl=600)
def fetch_klines_linear(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    if interval not in {"1","3","5","15","30","60","120","240","360","720","D","W","M"}:
        raise ValueError("Ogiltigt intervall.")
    out_rows: List[Dict[str, Any]] = []; cursor = None; safety = 0
    while True:
        params: Dict[str, Any] = {"category":"linear","symbol":symbol,"interval":interval,"limit":1000,"start":start_ms,"end":end_ms}
        if cursor: params["cursor"] = cursor
        data = bybit_get("/v5/market/kline", params)
        if str(data.get("retCode")) != "0": break
        result = data.get("result") or {}
        for r in (result.get("list") or []):
            out_rows.append({"ts": int(r[0]), "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4])})
        cursor = result.get("nextPageCursor")
        if not cursor: break
        safety += 1
        if safety > 100: break
        time.sleep(0.06)
    if not out_rows:
        return pd.DataFrame(columns=["ts","open","high","low","close"]).astype({"ts":"int64","open":"float","high":"float","low":"float","close":"float"})
    df = pd.DataFrame(out_rows).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df

def resample_5m_to_10m(df5: pd.DataFrame) -> pd.DataFrame:
    if df5.empty: return df5.copy()
    df = df5.copy(); dt = pd.to_datetime(df["ts"], unit="ms", utc=True); df = df.set_index(dt)
    o = df["open"].resample("10T").first(); h = df["high"].resample("10T").max()
    l = df["low"].resample("10T").min(); c = df["close"].resample("10T").last()
    out = pd.DataFrame({"open":o,"high":h,"low":l,"close":c}).dropna().reset_index()
    out["ts"] = (out["index"].astype("int64") // 10**6).astype("int64")
    out = out.drop(columns=["index"]); out = out[["ts","open","high","low","close"]]
    return out

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

def build_ma_df(symbol: str, timeframe_sel: str, ma_period: int, t_first_entry: int, t_last_event: int, post_extend_days: int) -> pd.DataFrame:
    tf_map = {"5m":"5", "15m":"15", "30m":"30", "60m":"60"}
    tf_ms = timeframe_to_ms(timeframe_sel)
    warmup = ma_period * tf_ms * 3
    t_start = max(0, int(t_first_entry) - warmup)
    t_end = int(t_last_event) + int(post_extend_days) * 24 * 3600 * 1000
    if timeframe_sel == "10m":
        base = fetch_klines_linear(symbol, "5", t_start, t_end)
        df = resample_5m_to_10m(base)
    else:
        bybit_tf = tf_map[timeframe_sel]
        df = fetch_klines_linear(symbol, bybit_tf, t_start, t_end)
    if df.empty: return df
    df = df.sort_values("ts").reset_index(drop=True)
    df["ma"] = compute_sma(df["close"], ma_period)
    return df

# ====== VILLKOR (CLOSE-only) ======
def first_stop_close_condition(ma_df: pd.DataFrame, t_entry: int, t_deadline: int, pos_side: str, tf: str) -> Tuple[bool, Optional[float], Optional[int]]:
    if ma_df.empty: return False, None, None
    tf_ms = timeframe_to_ms(tf)
    start_ts = t_entry + tf_ms
    sub = ma_df[(ma_df["ts"] >= start_ts) & (ma_df["ts"] <= t_deadline)].dropna(subset=["ma"]).copy()
    if sub.empty: return False, None, None
    if pos_side == "long": hit = sub[sub["close"] < sub["ma"]]
    else: hit = sub[sub["close"] > sub["ma"]]
    if not hit.empty:
        row = hit.iloc[0]
        return True, float(row["close"]), int(row["ts"])
    return False, None, None

# ====== MULTI-TP HJÄLP ======
@dataclass
class TPTier:
    enabled: bool
    pct_of_remaining: float
    mode: str                 # "MA" eller "ORIGINAL"
    ma_period: int = 0
    ma_tf: str = "15m"
    require_profit_first: bool = True

def first_ma_tp_signal(
    ma_df: pd.DataFrame, t_entry: int, t_deadline: int, pos_side: str, tf: str,
    entry_price: float, require_profit_first: bool
) -> Tuple[bool, Optional[float], Optional[int]]:
    if ma_df.empty: return False, None, None
    tf_ms = timeframe_to_ms(tf)
    start_ts = t_entry + tf_ms
    sub = ma_df[(ma_df["ts"] >= start_ts) & (ma_df["ts"] <= t_deadline)].dropna(subset=["ma"]).copy()
    if sub.empty: return False, None, None
    if require_profit_first:
        if str(pos_side).lower() == "long": passed = sub[sub["high"] >= float(entry_price)]
        else: passed = sub[sub["low"] <= float(entry_price)]
        if passed.empty: return False, None, None
        p_ts = int(passed.iloc[0]["ts"]); sub = sub[sub["ts"] >= p_ts]
    if str(pos_side).lower() == "long": tp_hit = sub[sub["close"] <= sub["ma"]]
    else: tp_hit = sub[sub["close"] >= sub["ma"]]
    if not tp_hit.empty:
        r = tp_hit.iloc[0]; return True, float(r["close"]), int(r["ts"])
    return False, None, None

# ====== PnL + EVENTS (ENTRY-notional ROI) – med Multi-TP ======
def compute_linear_fifo_pnl_and_events(
    trades_df: pd.DataFrame,
    stop_enabled: bool = False, stop_ma_period: int = 20, stop_tf: str = "15m",
    tp_enabled: bool = False, tp_ma_period: int = 20, tp_tf: str = "15m",
    allow_after_original: bool = False, post_extend_days: int = 14,
    tp_tiers: Optional[List[TPTier]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades_df.empty:
        summary = pd.DataFrame(columns=["symbol","realized_pnl_usdt","closed_notional_usdt","roi_pct"])
        events = pd.DataFrame(columns=["ts","symbol","side_close","qty_closed","price_close","pnl_usdt","notional_usdt","r",
                                       "stop_hit","tp_hit","after_original","exit_type","original_close_ts","original_close_price"])
        return summary, events

    df = trades_df.copy()
    df = _ensure_numeric(df, ["execQty","execPrice","execValue","execFee","execTime"])
    for c in ["symbol","side"]:
        if c not in df.columns: df[c] = None
    if "execTime" in df.columns: df = df.sort_values(by=["symbol","execTime"]).reset_index(drop=True)
    else: df = df.sort_values(by=["symbol"]).reset_index(drop=True)

    # fallback: en enda TP via MA (gamla beteendet)
    if tp_tiers is None:
        tp_tiers = []
        if tp_enabled:
            tp_tiers.append(TPTier(True, 100.0, "MA", tp_ma_period, tp_tf, True))

    summaries = []; event_rows = []
    ma_cache: Dict[Tuple[str,str,int,int], pd.DataFrame] = {}

    for symbol, g in df.groupby("symbol", sort=False):
        longs: List[Dict[str, float]] = []; shorts: List[Dict[str, float]] = []
        realized_pnl = 0.0; closed_notional = 0.0
        t_min = int(pd.to_numeric(g["execTime"], errors="coerce").min())
        t_max = int(pd.to_numeric(g["execTime"], errors="coerce").max())

        def get_ma(symbol_: str, tf_: str, period_: int) -> pd.DataFrame:
            key = (symbol_, tf_, period_, int(post_extend_days))
            if key not in ma_cache:
                try: ma_cache[key] = build_ma_df(symbol_, tf_, period_, t_min, t_max, post_extend_days)
                except Exception: ma_cache[key] = pd.DataFrame()
            return ma_cache[key]

        def deadline_ts(ma_df: pd.DataFrame, base_close_ts: int) -> int:
            if allow_after_original and not ma_df.empty:
                return int(ma_df["ts"].max())
            return int(base_close_ts)

        for _, row in g.iterrows():
            side = str(row.get("side") or ""); qty_filled  = float(row.get("execQty") or 0.0)
            price_fill = float(row.get("execPrice") or 0.0); fee = float(row.get("execFee") or 0.0)
            ts_fill = int(row.get("execTime") or 0)
            if qty_filled <= 0 or price_fill <= 0:
                realized_pnl -= fee; continue

            # BUY stänger shorts först
            if side.lower() == "buy":
                qty_to_match = qty_filled; fee_remaining = fee
                while qty_to_match > 0 and shorts:
                    lot = shorts[0]; take_total = min(qty_to_match, lot["qty"])
                    base_close_price = price_fill; base_close_ts = ts_fill

                    candidates: List[Tuple[int,str,float,float]] = []  # (ts, tag, price, pct_of_remaining)

                    if stop_enabled:
                        ma_stop_df = get_ma(symbol, stop_tf, stop_ma_period)
                        d_stop = deadline_ts(ma_stop_df, base_close_ts)
                        hit, sp, st_ts = first_stop_close_condition(ma_stop_df, int(lot["ts"]), d_stop, "short", stop_tf)
                        if hit: candidates.append((int(st_ts), "STOP_ALL", float(sp), 100.0))

                    for idx, tier in enumerate(tp_tiers, start=1):
                        if not tier.enabled: continue
                        if tier.mode.upper() == "ORIGINAL":
                            candidates.append((int(base_close_ts), f"TP{idx}_ORIG", float(base_close_price), float(tier.pct_of_remaining)))
                        else:
                            ma_tp_df = get_ma(symbol, tier.ma_tf, tier.ma_period)
                            d_tp = deadline_ts(ma_tp_df, base_close_ts)
                            hit, tpp, tpts = first_ma_tp_signal(
                                ma_tp_df, int(lot["ts"]), d_tp, "short", tier.ma_tf,
                                entry_price=float(lot["price"]), require_profit_first=bool(tier.require_profit_first)
                            )
                            if hit: candidates.append((int(tpts), f"TP{idx}_MA", float(tpp), float(tier.pct_of_remaining)))

                    if not candidates:
                        candidates.append((int(base_close_ts), "ORIGINAL_ALL", float(base_close_price), 100.0))

                    candidates.sort(key=lambda x: x[0])
                    remaining = take_total; last_ts_seen = None

                    for ev_ts, tag, ev_price, pct_rem in candidates:
                        if remaining <= 1e-12: break
                        if last_ts_seen is not None and ev_ts < last_ts_seen: continue
                        qty_part = remaining if "ALL" in tag else remaining * (pct_rem / 100.0)
                        qty_part = min(qty_part, remaining)
                        if qty_part <= 1e-12: continue

                        entry_notional = lot["price"] * qty_part
                        pnl_core = (lot["price"] - ev_price) * qty_part  # short: entry - exit
                        fee_part = fee * (qty_part / qty_filled)
                        pnl = pnl_core - fee_part
                        realized_pnl += pnl; fee_remaining -= fee_part; closed_notional += entry_notional
                        r = (pnl / entry_notional) if entry_notional > 0 else 0.0

                        event_rows.append({
                            "ts": int(ev_ts), "symbol": symbol, "side_close": "Buy→close short",
                            "qty_closed": qty_part, "price_close": float(ev_price),
                            "pnl_usdt": pnl, "notional_usdt": entry_notional, "r": r,
                            "stop_hit": tag == "STOP_ALL", "tp_hit": tag.startswith("TP"),
                            "after_original": ev_ts > base_close_ts,
                            "exit_type": tag, "original_close_ts": int(base_close_ts), "original_close_price": float(base_close_price)
                        })

                        remaining -= qty_part; last_ts_seen = ev_ts
                        if "ALL" in tag:
                            remaining = 0.0; break

                    used = take_total - remaining
                    lot["qty"] -= used; qty_to_match -= take_total
                    if lot["qty"] <= 1e-12: shorts.pop(0)

                if qty_to_match > 0:
                    realized_pnl -= fee_remaining
                    longs.append({"qty": qty_to_match, "price": price_fill, "ts": ts_fill})

            # SELL stänger longs först
            elif side.lower() == "sell":
                qty_to_match = qty_filled; fee_remaining = fee
                while qty_to_match > 0 and longs:
                    lot = longs[0]; take_total = min(qty_to_match, lot["qty"])
                    base_close_price = price_fill; base_close_ts = ts_fill

                    candidates: List[Tuple[int,str,float,float]] = []

                    if stop_enabled:
                        ma_stop_df = get_ma(symbol, stop_tf, stop_ma_period)
                        d_stop = deadline_ts(ma_stop_df, base_close_ts)
                        hit, sp, st_ts = first_stop_close_condition(ma_stop_df, int(lot["ts"]), d_stop, "long", stop_tf)
                        if hit: candidates.append((int(st_ts), "STOP_ALL", float(sp), 100.0))

                    for idx, tier in enumerate(tp_tiers, start=1):
                        if not tier.enabled: continue
                        if tier.mode.upper() == "ORIGINAL":
                            candidates.append((int(base_close_ts), f"TP{idx}_ORIG", float(base_close_price), float(tier.pct_of_remaining)))
                        else:
                            ma_tp_df = get_ma(symbol, tier.ma_tf, tier.ma_period)
                            d_tp = deadline_ts(ma_tp_df, base_close_ts)
                            hit, tpp, tpts = first_ma_tp_signal(
                                ma_tp_df, int(lot["ts"]), d_tp, "long", tier.ma_tf,
                                entry_price=float(lot["price"]), require_profit_first=bool(tier.require_profit_first)
                            )
                            if hit: candidates.append((int(tpts), f"TP{idx}_MA", float(tpp), float(tier.pct_of_remaining)))

                    if not candidates:
                        candidates.append((int(base_close_ts), "ORIGINAL_ALL", float(base_close_price), 100.0))

                    candidates.sort(key=lambda x: x[0])
                    remaining = take_total; last_ts_seen = None

                    for ev_ts, tag, ev_price, pct_rem in candidates:
                        if remaining <= 1e-12: break
                        if last_ts_seen is not None and ev_ts < last_ts_seen: continue
                        qty_part = remaining if "ALL" in tag else remaining * (pct_rem / 100.0)
                        qty_part = min(qty_part, remaining)
                        if qty_part <= 1e-12: continue

                        entry_notional = lot["price"] * qty_part
                        pnl_core = (ev_price - lot["price"]) * qty_part  # long: exit - entry
                        fee_part = fee * (qty_part / qty_filled)
                        pnl = pnl_core - fee_part
                        realized_pnl += pnl; fee_remaining -= fee_part; closed_notional += entry_notional
                        r = (pnl / entry_notional) if entry_notional > 0 else 0.0

                        event_rows.append({
                            "ts": int(ev_ts), "symbol": symbol, "side_close": "Sell→close long",
                            "qty_closed": qty_part, "price_close": float(ev_price),
                            "pnl_usdt": pnl, "notional_usdt": entry_notional, "r": r,
                            "stop_hit": tag == "STOP_ALL", "tp_hit": tag.startswith("TP"),
                            "after_original": ev_ts > base_close_ts,
                            "exit_type": tag, "original_close_ts": int(base_close_ts), "original_close_price": float(base_close_price)
                        })

                        remaining -= qty_part; last_ts_seen = ev_ts
                        if "ALL" in tag:
                            remaining = 0.0; break

                    used = take_total - remaining
                    lot["qty"] -= used; qty_to_match -= take_total
                    if lot["qty"] <= 1e-12: longs.pop(0)

                if qty_to_match > 0:
                    realized_pnl -= fee_remaining
                    shorts.append({"qty": qty_to_match, "price": price_fill, "ts": ts_fill})
            else:
                realized_pnl -= fee

        roi = (realized_pnl / closed_notional * 100.0) if closed_notional > 0 else 0.0
        summaries.append({"symbol": symbol, "realized_pnl_usdt": round(realized_pnl, 6),
                          "closed_notional_usdt": round(closed_notional, 6), "roi_pct": round(roi, 4)})

    summary_df = pd.DataFrame(summaries).sort_values(by="realized_pnl_usdt", ascending=False).reset_index(drop=True)
    events_df = pd.DataFrame(event_rows)
    if not events_df.empty: events_df = events_df.sort_values(by="ts").reset_index(drop=True)
    else:
        events_df = pd.DataFrame(columns=["ts","symbol","side_close","qty_closed","price_close","pnl_usdt","notional_usdt","r",
                                          "stop_hit","tp_hit","after_original","exit_type","original_close_ts","original_close_price"])
    return summary_df, events_df

# ===== WINRATE =====
def compute_winloss(events_df: pd.DataFrame) -> Tuple[int, int, int, float]:
    if events_df is None or events_df.empty or "pnl_usdt" not in events_df.columns:
        return 0, 0, 0, 0.0
    e = events_df.copy(); e["pnl_usdt"] = pd.to_numeric(e["pnl_usdt"], errors="coerce")
    e = e.dropna(subset=["pnl_usdt"]); e = e[e["pnl_usdt"] != 0]
    if e.empty: return 0, 0, 0, 0.0
    wins = int((e["pnl_usdt"] > 0).sum()); losses = int((e["pnl_usdt"] < 0).sum())
    total = wins + losses; winrate = (wins / total * 100.0) if total > 0 else 0.0
    return wins, losses, total, winrate

# ===== SIMULERING & KURVOR =====
def simulate_equity_from_events(events_df: pd.DataFrame, start_equity: float, allocation_pct: float) -> Tuple[pd.DataFrame, float, float]:
    if events_df.empty or start_equity <= 0 or allocation_pct <= 0:
        curve = pd.DataFrame(columns=["ts","equity"]); return curve, start_equity, 0.0
    eq = float(start_equity); rows = []
    for _, row in events_df.iterrows():
        r = float(row.get("r") or 0.0); eq = eq * (1.0 + allocation_pct * r)
        rows.append({"ts": int(row.get("ts") or 0), "equity": eq})
    curve = pd.DataFrame(rows)
    total_ret = (eq / start_equity - 1.0) * 100.0
    return curve, eq, total_ret

def overlay_curves(base_curve: pd.DataFrame, sim_curve: pd.DataFrame) -> pd.DataFrame:
    df1 = base_curve.rename(columns={"equity": "Original"}); df2 = sim_curve.rename(columns={"equity": "MA-Strategi"})
    merged = pd.merge(df1, df2, on="ts", how="outer").sort_values("ts")
    merged[["Original","MA-Strategi"]] = merged[["Original","MA-Strategi"]].ffill()
    return merged.set_index("ts")

# ===== PROJEKTION =====
def estimate_daily_rate_from_curve(curve_df: pd.DataFrame) -> float:
    if curve_df is None or curve_df.empty or len(curve_df) < 2: return 0.0
    ts0 = int(curve_df["ts"].iloc[0]); ts1 = int(curve_df["ts"].iloc[-1])
    eq0 = float(curve_df["equity"].iloc[0]); eq1 = float(curve_df["equity"].iloc[-1])
    if eq0 <= 0 or eq1 <= 0 or ts1 <= ts0: return 0.0
    days = (ts1 - ts0) / (24*3600*1000); 
    if days <= 0: return 0.0
    r_day = (eq1 / eq0) ** (1.0 / days) - 1.0
    return r_day

def project_equity(current_equity: float, daily_rate: float, horizon_days: int) -> float:
    if current_equity <= 0: return 0.0
    return current_equity * ((1.0 + daily_rate) ** max(0, int(horizon_days)))

# ===== OPTIMERING =====
def _df_fingerprint(df: pd.DataFrame) -> str:
    if df is None or df.empty: return "empty"
    tmin = pd.to_numeric(df.get("execTime", pd.Series([0])), errors="coerce").min()
    tmax = pd.to_numeric(df.get("execTime", pd.Series([0])), errors="coerce").max()
    nids = df.get("execId", pd.Series(dtype=str)).nunique() if "execId" in df.columns else len(df)
    return f"{nids}_{int(tmin)}_{int(tmax)}"

def _tiers_from_scenario(pcts: List[int|float], modes: List[str], specs: List[Optional[Tuple[int,str,bool]]]) -> List[TPTier]:
    tiers: List[TPTier] = []
    for i in range(3):
        en = (i < len(pcts) and pcts[i] and pcts[i] > 0)
        pct = float(pcts[i]) if i < len(pcts) else 0.0
        mode = (modes[i] if i < len(modes) else "MA").upper()
        sp = specs[i] if i < len(specs) else None
        if mode == "ORIGINAL":
            tiers.append(TPTier(enabled=en, pct_of_remaining=pct, mode="ORIGINAL"))
        else:
            per, tf, pf = sp if sp else (50, "15m", True)
            tiers.append(TPTier(enabled=en, pct_of_remaining=pct, mode="MA", ma_period=int(per), ma_tf=str(tf), require_profit_first=bool(pf)))
    return tiers

@st.cache_data(show_spinner=False, ttl=600)
def run_combo(
    df_fpr: str, trades_df: pd.DataFrame,
    stop_period: int, stop_tf: str,
    tp_period: int, tp_tf: str,
    allow_after_original: bool, post_extend_days: int,
    start_equity: float, alloc: float,
    multi_tp_tiers: Optional[List[TPTier]] = None
) -> Dict[str, Any]:
    pnl_df, events_df = compute_linear_fifo_pnl_and_events(
        trades_df,
        stop_enabled=True, stop_ma_period=stop_period, stop_tf=stop_tf,
        tp_enabled=True, tp_ma_period=tp_period, tp_tf=tp_tf,
        allow_after_original=allow_after_original, post_extend_days=post_extend_days,
        tp_tiers=multi_tp_tiers
    )
    curve, final_eq, total_ret_pct = simulate_equity_from_events(events_df, start_equity, alloc)
    total_realized = float(pnl_df["realized_pnl_usdt"].sum()) if not pnl_df.empty else 0.0
    entry_notional = float(pnl_df["closed_notional_usdt"].sum()) if not pnl_df.empty else 0.0
    avg_roi = (total_realized / entry_notional * 100.0) if entry_notional > 0 else 0.0
    wins, losses, total, winrate_pct = compute_winloss(events_df)
    return {"final_eq": final_eq, "total_ret_pct": total_ret_pct, "total_realized": total_realized,
            "avg_roi": avg_roi, "winrate_pct": winrate_pct, "wins": wins, "losses": losses, "total_trades": total,
            "stop_period": stop_period, "stop_tf": stop_tf, "tp_period": tp_period, "tp_tf": tp_tf,
            "post_extend_days": post_extend_days,
            "tiers_repr": str([vars(t) for t in (multi_tp_tiers or [])])}

def optimize_settings(
    trades_df: pd.DataFrame,
    start_equity: float, allocation_pct: float,
    allow_after_original: bool, post_extend_days: int,
    stop_periods: List[int], stop_tfs: List[str],
    tp_periods: List[int], tp_tfs: List[str],
    enable_multi_tp_opt: bool,
    multi_tp_scenarios: List[Tuple[List[float], List[str], List[Optional[Tuple[int,str,bool]]]]]
) -> pd.DataFrame:
    results = []
    if trades_df is None or trades_df.empty or start_equity <= 0 or allocation_pct <= 0:
        return pd.DataFrame(results)
    fpr = _df_fingerprint(trades_df)

    # hur många kombinationer?
    n_base = len(stop_periods)*len(stop_tfs)*len(tp_periods)*len(tp_tfs)
    if enable_multi_tp_opt and multi_tp_scenarios:
        total = n_base * len(multi_tp_scenarios)
    else:
        total = n_base
    prog = st.progress(0); cnt = 0

    for sp in stop_periods:
        for stf in stop_tfs:
            for tp in tp_periods:
                for ttf in tp_tfs:
                    if enable_multi_tp_opt and multi_tp_scenarios:
                        # kör alla Multi-TP-scenarios
                        for (pcts, modes, specs) in multi_tp_scenarios:
                            tiers = _tiers_from_scenario(pcts, modes, specs)
                            res = run_combo(
                                fpr, trades_df, sp, stf, tp, ttf, allow_after_original, post_extend_days,
                                start_equity, allocation_pct, multi_tp_tiers=tiers
                            )
                            # lägg scenario meta
                            res["tp_scenario"] = {"pcts":pcts,"modes":modes,"specs":specs}
                            results.append(res); cnt += 1; prog.progress(min(100, int(cnt/total*100)))
                    else:
                        # endast "single TP" (gamla sättet)
                        res = run_combo(
                            fpr, trades_df, sp, stf, tp, ttf, allow_after_original, post_extend_days,
                            start_equity, allocation_pct, multi_tp_tiers=None
                        )
                        res["tp_scenario"] = None
                        results.append(res); cnt += 1; prog.progress(min(100, int(cnt/total*100)))
    prog.empty()
    return pd.DataFrame(results).sort_values("final_eq", ascending=False).reset_index(drop=True)

# ===== UI ======
st.set_page_config(page_title="Bybit Perps – PnL, MA-Stop & MA/Multi-TP (Close-only)", page_icon="📊", layout="wide")
st.title("📊 Bybit Perps – PnL, MA-Stop & MA/Multi-TP (Close-only)")
st.caption("Jämför Original vs MA/Multi-TP, se exit-markörer, optimera parametrar, projektera framåt, följ winrate – och beräkna positionsstorlek.")

with st.sidebar:
    st.header("Konto")
    acct_type = st.selectbox("Account type", ["UNIFIED", "CONTRACT"], index=0)
    coin_sel = st.text_input("Valuta (t.ex. USDT)", value="USDT")
    colb1, colb2 = st.columns(2)
    with colb1: show_balance_btn = st.button("Hämta saldo")
    with colb2: refresh_balance_btn = st.button("Uppdatera")

if show_balance_btn or refresh_balance_btn:
    try:
        st.session_state['acct_balance'] = fetch_wallet_balance(API_KEY, API_SECRET, acct_type, coin_sel or None)
        st.toast("Saldo uppdaterat.", icon="✅")
    except Exception as e:
        st.error(f"Kunde inte hämta saldo: {e}")

bal = st.session_state.get('acct_balance')
if isinstance(bal, dict) and bal and 'coin' in bal:
    st.subheader("Kontosaldo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"{bal['equity']:,.2f} {bal['coin']}")
    c2.metric("Wallet", f"{bal['walletBalance']:,.2f} {bal['coin']}")
    c3.metric("Available", f"{bal['available']:,.2f} {bal['coin']}")
    c4.metric("Unrealised PnL", f"{bal['unrealisedPnl']:,.2f} {bal['coin']}")
    st.caption(f"Konto: {bal.get('accountType','?')}")
else:
    st.info("Hämta kontosaldo för att använda positionsräknaren.")

with st.sidebar:
    st.header("Tidsintervall (trades)")
    c1, c2 = st.columns(2)
    with c1: days_back_from = st.number_input("Från (dagar bakåt)", 0, 3650, 0, 1)
    with c2: days_back_to   = st.number_input("Till (dagar bakåt)", 0, 3650, 0, 1)
    st.caption("Lämna båda 0 för allt API ger.")

    st.header("Kapital & Allokering (sim-kurvor)")
    start_equity = st.number_input("Startkapital i USDT", min_value=0.0, value=0.0, step=100.0)
    allocation_pct_input = st.number_input("Allokering per event (%)", min_value=0.0, max_value=100.0, value=100.0, step=5.0)

    st.header("MA-Stop (CLOSE)")
    use_ma_stop = st.checkbox("Aktivera MA-Stop", value=True)
    stop_ma_period = st.selectbox("MA-period (Stop)", [10, 20, 50], index=1)
    stop_tf = st.selectbox("Timeframe (Stop)", ["5m","10m","15m","30m","60m"], index=2)

    st.header("MA-TP (CLOSE)")
    use_ma_tp = st.checkbox("Aktivera MA-TP (kräver passage av vinstläge)", value=True)
    tp_ma_period = st.selectbox("MA-period (TP)", [10, 20, 50], index=1)
    tp_tf = st.selectbox("Timeframe (TP)", ["5m","10m","15m","30m","60m"], index=2)

    st.header("Multi-TP (upp till 3 nivåer)")
    use_multi_tp = st.checkbox("Aktivera Multi-TP", value=False)
    tp_tiers_cfg: Optional[List[TPTier]] = None
    if use_multi_tp:
        tp_tiers_cfg = []
        for i in range(1, 4):
            st.markdown(f"**TP{i}**")
            enabled = st.checkbox(f"Aktivera TP{i}", value=(i==1), key=f"tp{i}_en")
            pct = st.number_input(f"TP{i} – % av kvarvarande", 1.0, 100.0, 33.0 if i<3 else 34.0, 1.0, key=f"tp{i}_pct")
            mode = st.selectbox(f"TP{i} – Trigger", ["MA-korsning","Original-close"], index=0, key=f"tp{i}_mode")
            if mode == "MA-korsning":
                per = st.selectbox(f"TP{i} – MA-period", [10,20,50], index=1, key=f"tp{i}_per")
                tfv = st.selectbox(f"TP{i} – Timeframe", ["5m","10m","15m","30m","60m"], index=2, key=f"tp{i}_tf")
                prof = st.checkbox(f"TP{i} kräver vinstläge först", value=True, key=f"tp{i}_pf")
                tp_tiers_cfg.append(TPTier(enabled, float(pct), "MA", int(per), str(tfv), bool(prof)))
            else:
                tp_tiers_cfg.append(TPTier(enabled, float(pct), "ORIGINAL"))

    st.header("Prioritering & sökfönster")
    allow_after = st.toggle("Tillåt exit efter original (håll längre)", value=st.session_state["allow_after_default"])
    st.session_state["allow_after_default"] = allow_after
    post_extend_days = st.number_input("Sökfönster efter original-TP (dagar)", 1, 90, 30, 1,
                                       help="Hur länge vi fortsätter leta efter MA-TP/STOP efter historisk TP-passering/sista event.")

    fetch_btn = st.button("Hämta & simulera trades")

# ===== Position Size-räknare =====
st.divider()
st.markdown("## Position size & Risk (live mot kontosaldo)")

def _to_float(s: str) -> float:
    if s is None: return 0.0
    s = str(s).strip().replace(" ", "").replace(",", ".")
    import re; m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    try: return float(m.group(0)) if m else 0.0
    except Exception: return 0.0

for k in ["entry_price_str","stop_price_str","tp_price_str"]:
    st.session_state.setdefault(k, "")

def _bulk_paste_cb():
    raw = st.session_state.get("bulk_paste_raw","") or ""
    tpl = raw.replace(";", " ").replace("/", " ").replace("|", " ").replace("\t", " ").replace(",", ".")
    import re; nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", tpl)
    if len(nums) >= 2:
        st.session_state["entry_price_str"] = nums[0]; st.session_state["stop_price_str"]  = nums[1]
        if len(nums) >= 3: st.session_state["tp_price_str"] = nums[2]
        st.toast("Klistrade in entry/stop/TP.", icon="✅")

if isinstance(st.session_state.get('acct_balance'), dict) and st.session_state['acct_balance'] and 'coin' in st.session_state['acct_balance']:
    bal = st.session_state['acct_balance']
    side = st.radio("Riktning", ["long", "short"], horizontal=True, index=0)
    st.text_input("Snabbklistra (entry stop [tp]) – t.ex. 0.3726 0.3510 0.4180", key="bulk_paste_raw", on_change=_bulk_paste_cb,
                  placeholder="0.3726 0.3510 0.4180")
    e1, e2, e3, e4 = st.columns(4)
    with e1: st.text_input("Entry", key="entry_price_str", placeholder="t.ex. 0.3726")
    with e2: st.text_input("Stop", key="stop_price_str", placeholder="t.ex. 0.3510")
    with e3: st.text_input("Take Profit (valfritt)", key="tp_price_str", placeholder="t.ex. 0.4180")
    with e4: leverage = st.number_input("Leverage (kontroll)", 1.0, 100.0, 5.0, 1.0)

    entry_price = _to_float(st.session_state.get("entry_price_str",""))
    stop_price  = _to_float(st.session_state.get("stop_price_str",""))
    tp_price    = _to_float(st.session_state.get("tp_price_str",""))
    r1, r2 = st.columns(2)
    with r1: risk_pct = st.number_input("Risk av konto (%)", 0.01, 100.0, 1.0, 0.25)
    with r2: fee_buf = st.number_input("Buffert avgifter/slippage (%)", 0.0, 1.0, 0.10, 0.05,
                                       help="Dras från riskbeloppet.")

    if entry_price > 0 and stop_price > 0 and risk_pct > 0:
        ps = calc_position_size(side, entry_price, stop_price, bal['equity'], risk_pct, fee_buf, leverage, available=bal['available'])
        rr = calc_rr(side, entry_price, stop_price, tp_price if tp_price > 0 else None)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Riskbelopp", f"{ps['risk_amount_usdt']:,.2f} USDT")
        k2.metric("Kontraktsmängd (qty)", f"{ps['qty']:,.6f}")
        k3.metric("Notional", f"{ps['notional']:,.2f} USDT")
        k4.metric("Init. marginal", f"{ps['init_margin']:,.2f} USDT")
        j1, j2, j3 = st.columns(3)
        j1.metric("Risk/kontrakt", f"{rr['risk_per_unit']:,.6f} $")
        j2.metric("Reward/kontrakt", f"{rr['reward_per_unit']:,.6f} $")
        j3.metric("R:R", f"{rr['rr']:,.2f} R")
        if ps["capped"] > 0:
            st.warning("Storleken har kapats till tillgänglig marginal × leverage.")
    else:
        st.info("Klistra in eller skriv entry, stop och risk% och tryck Enter.")
else:
    st.info("Hämta saldo för att aktivera positionsräknaren.")

# ===== HÄMTA TRADES =====
if fetch_btn:
    start_ms = None; end_ms = None; now_ms = int(time.time() * 1000)
    if days_back_from > 0: start_ms = now_ms - int(days_back_from * 24 * 3600 * 1000)
    if days_back_to > 0:
        end_ms = now_ms - int(days_back_to * 24 * 3600 * 1000)
        if start_ms and end_ms and start_ms > end_ms: start_ms, end_ms = end_ms, start_ms
    with st.spinner("Hämtar Bybit-perpetuals…"):
        try: raw_trades = fetch_all_linear_trades(API_KEY, API_SECRET, start_ms, end_ms)
        except Exception as e: st.error(f"Kunde inte hämta trades: {e}"); st.stop()
    if not raw_trades: st.info("Inga trades hittades för valt intervall."); st.stop()
    st.session_state["trades_df"] = pd.DataFrame(raw_trades); st.session_state["opt_results"] = None

# ===== HUVUDVY – TRADES =====
df = st.session_state.get("trades_df")
if df is None:
    st.info("Klicka ”Hämta & simulera trades”."); st.stop()

nice_cols = ["symbol","side","execQty","execPrice","execValue","execFee","feeCurrency","orderId","execId","execType","isMaker","execTime"]
view_cols = [c for c in nice_cols if c in df.columns]
st.subheader("Alla PERP-trades (USDT-linear)")
st.dataframe(df[view_cols] if view_cols else df, use_container_width=True, height=420)

# ===== Original (bas) =====
base_pnl_df, base_events_df = compute_linear_fifo_pnl_and_events(
    df, stop_enabled=False, tp_enabled=False, allow_after_original=False, post_extend_days=post_extend_days
)
base_total_realized = float(base_pnl_df["realized_pnl_usdt"].sum()) if not base_pnl_df.empty else 0.0
base_entry_notional = float(base_pnl_df["closed_notional_usdt"].sum()) if not base_pnl_df.empty else 0.0
base_avg_roi = (base_total_realized / base_entry_notional * 100.0) if base_entry_notional > 0 else 0.0
base_w, base_l, base_n, base_winrate = compute_winloss(base_events_df)

# ===== MA/Multi-TP-strategi =====
if not use_ma_stop and not use_ma_tp and not use_multi_tp:
    st.info("Aktivera minst MA-Stop, MA-TP eller Multi-TP för jämförelse."); st.stop()

sim_pnl_df, sim_events_df = compute_linear_fifo_pnl_and_events(
    df,
    stop_enabled=use_ma_stop, stop_ma_period=int(stop_ma_period), stop_tf=str(stop_tf),
    tp_enabled=use_ma_tp, tp_ma_period=int(tp_ma_period), tp_tf=str(tp_tf),
    allow_after_original=bool(allow_after), post_extend_days=int(post_extend_days),
    tp_tiers=tp_tiers_cfg if use_multi_tp else None
)
sim_total_realized = float(sim_pnl_df["realized_pnl_usdt"].sum()) if not sim_pnl_df.empty else 0.0
sim_entry_notional = float(sim_pnl_df["closed_notional_usdt"].sum()) if not sim_pnl_df.empty else 0.0
sim_avg_roi = (sim_total_realized / sim_entry_notional * 100.0) if sim_entry_notional > 0 else 0.0
sim_w, sim_l, sim_n, sim_winrate  = compute_winloss(sim_events_df)

# ===== Jämförelse =====
st.markdown("## Jämförelse: Original vs MA/Multi-TP")
colL, colR = st.columns(2)
with colL:
    st.markdown("### Original (historiskt)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Realiserad PnL (USDT)", f"{base_total_realized:,.2f}")
    c2.metric("Genomsnittlig ROI", f"{base_avg_roi:,.2f}%")
    if start_equity > 0: c3.metric("Kontoutv. från start", f"{(base_total_realized/start_equity*100):,.2f}%")
    wc1, wc2 = st.columns(2)
    wc1.metric(f"Winrate ({base_w}W / {base_l}L)", f"{base_winrate:,.2f}%")
    st.caption(f"Totalt räknade events: {base_n}")
    st.dataframe(base_pnl_df, use_container_width=True, height=280)
with colR:
    st.markdown("### MA/Multi-TP (aktuella val)")
    d1, d2, d3 = st.columns(3)
    d1.metric("Realiserad PnL (USDT)", f"{sim_total_realized:,.2f}", delta=f"{(sim_total_realized-base_total_realized):,.2f}")
    d2.metric("Genomsnittlig ROI", f"{sim_avg_roi:,.2f}%", delta=f"{(sim_avg_roi-base_avg_roi):,.2f}%")
    if start_equity > 0:
        d3.metric("Kontoutv. från start", f"{(sim_total_realized/start_equity*100):,.2f}%",
                  delta=f"{((sim_total_realized-base_total_realized)/start_equity*100):,.2f}%")
    wd1, wd2 = st.columns(2)
    wd1.metric(f"Winrate ({sim_w}W / {sim_l}L)", f"{sim_winrate:,.2f}%", delta=f"{(sim_winrate-base_winrate):,.2f}%")
    st.caption(f"Totalt räknade events: {sim_n}")
    st.dataframe(sim_pnl_df, use_container_width=True, height=280)

# ===== Kontokurvor =====
st.markdown("### Kontokurva – överlagrad")
alloc = float(allocation_pct_input) / 100.0 if start_equity > 0 else 0.0
if start_equity > 0 and alloc > 0:
    base_curve, base_final_eq, base_total_ret = simulate_equity_from_events(base_events_df, start_equity, alloc)
    sim_curve, sim_final_eq, sim_total_ret = simulate_equity_from_events(sim_events_df, start_equity, alloc)
    over = overlay_curves(base_curve, sim_curve)
    st.line_chart(over, height=280, use_container_width=True)
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Slutligt konto – Original", f"{base_final_eq:,.2f} USDT")
    e2.metric("Total avkastning – Original", f"{base_total_ret:,.2f}%")
    e3.metric("Slutligt konto – MA/Multi-TP", f"{sim_final_eq:,.2f} USDT", delta=f"{(sim_final_eq-base_final_eq):,.2f}")
    e4.metric("Total avkastning – MA/Multi-TP", f"{sim_total_ret:,.2f}%", delta=f"{(sim_total_ret-base_total_ret):,.2f}%")
else:
    st.info("Ange startkapital och allokering > 0% för att se kontokurvor.")
    base_curve, sim_curve = pd.DataFrame(), pd.DataFrame()

# ===== Winrate per symbol =====
st.markdown("### Winrate per symbol (MA/Multi-TP)")
if not sim_events_df.empty:
    tmp = sim_events_df.copy()
    tmp["pnl_usdt"] = pd.to_numeric(tmp["pnl_usdt"], errors="coerce"); tmp = tmp.dropna(subset=["pnl_usdt"])
    tmp = tmp[tmp["pnl_usdt"] != 0]
    if not tmp.empty:
        g = tmp.groupby("symbol")["pnl_usdt"]
        per_symbol = pd.DataFrame({"wins": g.apply(lambda s: int((s > 0).sum())),
                                   "losses": g.apply(lambda s: int((s < 0).sum()))})
        per_symbol["total"] = per_symbol["wins"] + per_symbol["losses"]
        per_symbol["winrate_pct"] = per_symbol.apply(lambda r: (r["wins"]/r["total"]*100.0) if r["total"]>0 else 0.0, axis=1)
        per_symbol = per_symbol.sort_values(["winrate_pct","total"], ascending=[False, False]).reset_index()
        st.dataframe(per_symbol, use_container_width=True, height=260)
    else:
        st.info("Inga icke-noll PnL-events för att beräkna per-symbol-winrate.")
else:
    st.info("Inga händelser i simuleringen för att beräkna per-symbol-winrate.")

# ===== Händelser – etiketter =====
st.markdown("### Händelser (senaste 100) – med exit_type och jämförelse mot original")
if not sim_events_df.empty:
    show_cols = ["ts","symbol","side_close","qty_closed","price_close","pnl_usdt","r","exit_type","after_original",
                 "original_close_price","original_close_ts","stop_hit","tp_hit"]
    st.dataframe(sim_events_df[show_cols].tail(100), use_container_width=True, height=360)
else:
    st.info("Inga händelser i simuleringen.")

# ===== Exit-markörer (TP/STOP) =====
st.divider()
st.markdown("## Exit-markörer (simulerade)")
def build_markers_df(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty: return pd.DataFrame(columns=["symbol","event","ts","time_utc","price","pnl_usdt","r","after_original","side_close"])
    m = events[events["exit_type"].isin(["STOP_ALL","TP1_MA","TP2_MA","TP3_MA","TP1_ORIG","TP2_ORIG","TP3_ORIG","ORIGINAL_ALL"])].copy()
    if m.empty: return pd.DataFrame(columns=["symbol","event","ts","time_utc","price","pnl_usdt","r","after_original","side_close"])
    m["event"] = np.where(m["exit_type"].str.contains("STOP"),"MA_STOP", np.where(m["exit_type"].str.contains("_MA"),"MA_TP","ORIG_TP"))
    m["time_utc"] = pd.to_datetime(m["ts"], unit="ms", utc=True)
    m.rename(columns={"price_close":"price"}, inplace=True)
    cols = ["symbol","event","ts","time_utc","price","pnl_usdt","r","after_original","side_close","exit_type"]
    cols = [c for c in cols if c in m.columns]
    return m[cols].sort_values(["symbol","ts"]).reset_index(drop=True)

markers_df = build_markers_df(sim_events_df)
if markers_df.empty:
    st.info("Inga simulerade MA-TP/STOP att visa för nuvarande inställningar.")
else:
    all_syms = list(markers_df["symbol"].unique())
    sel_syms = st.multiselect("Välj symbol(er) att visa", all_syms, default=all_syms)
    view_markers = markers_df[markers_df["symbol"].isin(sel_syms)].copy()
    st.dataframe(view_markers, use_container_width=True, height=280)
    csv = view_markers.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Ladda ner markörer (CSV)", data=csv, file_name="ma_exit_markers.csv", mime="text/csv")

    st.markdown("### Chart-preview med markörer")
    c1, c2 = st.columns(2)
    with c1: chart_symbol = st.selectbox("Symbol för chart", all_syms)
    with c2:
        chart_tf = st.selectbox("Timeframe för chart", ["5m","10m","15m","30m","60m"], index=2,
                                help="Visningstimeframe (separat från reglernas TF).")
    subm = markers_df[markers_df["symbol"] == chart_symbol]
    tmin = int(subm["ts"].min()); tmax = int(subm["ts"].max())
    pad = timeframe_to_ms(chart_tf) * 200
    start_ms = max(0, tmin - pad); end_ms = tmax + pad

    if chart_tf == "10m":
        base = fetch_klines_linear(chart_symbol, "5", start_ms, end_ms); kdf = resample_5m_to_10m(base)
    else:
        tf_map = {"5m":"5","15m":"15","30m":"30","60m":"60"}
        kdf = fetch_klines_linear(chart_symbol, tf_map[chart_tf], start_ms, end_ms)

    if kdf.empty:
        st.info("Kunde inte hämta klines för vald symbol/tidsfönster.")
    else:
        kdf["time_utc"] = pd.to_datetime(kdf["ts"], unit="ms", utc=True)
        if not PLOTLY_AVAILABLE:
            st.warning("Plotly saknas. Installera `pip install plotly` för candlestick.")
            line_df = kdf[["time_utc","close"]].set_index("time_utc")
            st.line_chart(line_df, use_container_width=True, height=280)
        else:
            fig = go.Figure(data=[go.Candlestick(x=kdf["time_utc"], open=kdf["open"], high=kdf["high"],
                                                 low=kdf["low"], close=kdf["close"], name="Price")])
            subs = subm.copy(); subs["time_utc"] = pd.to_datetime(subs["ts"], unit="ms", utc=True)
            stops = subs[subs["event"] == "MA_STOP"]
            if not stops.empty:
                fig.add_trace(go.Scatter(x=stops["time_utc"], y=stops["price"], mode="markers+text", text=["STOP"]*len(stops),
                                         textposition="top center", name="MA_STOP", marker=dict(symbol="x", size=10)))
            tps = subs[subs["event"].isin(["MA_TP","ORIG_TP"])]
            if not tps.empty:
                fig.add_trace(go.Scatter(x=tps["time_utc"], y=tps["price"], mode="markers+text",
                                         text=tps["event"], textposition="bottom center", name="TPs",
                                         marker=dict(symbol="triangle-up", size=10)))
            fig.update_layout(height=520, xaxis_title="Tid (UTC)", yaxis_title="Pris", legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

# ===== Framåtriktad projektion =====
st.divider()
st.markdown("## Framåtriktad projektion (daglig CAGR)")
if start_equity > 0 and alloc > 0 and (not base_curve.empty or not sim_curve.empty):
    proj_source = st.radio("Vilken kurva vill du projicera från?", ["MA-Strategi", "Original"], index=0, horizontal=True)
    horizon_days = st.number_input("Antal dagar framåt", 1, 2000, 30, 1)
    src_curve = sim_curve if proj_source == "MA-Strategi" else base_curve
    if src_curve.empty or len(src_curve) < 2:
        st.warning("För få datapunkter för att estimera daglig avkastning.")
    else:
        current_eq = float(src_curve["equity"].iloc[-1]); daily_rate = estimate_daily_rate_from_curve(src_curve)
        span_days = (int(src_curve["ts"].iloc[-1]) - int(src_curve["ts"].iloc[0])) / (24*3600*1000)
        if span_days < 7: st.info("Obs: Historiken är kort (<7 dagar).")
        projected_eq = project_equity(current_eq, daily_rate, int(horizon_days))
        exp_change_pct = (projected_eq / current_eq - 1.0) * 100.0; exp_profit = projected_eq - current_eq
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Antagen daglig avkastning", f"{daily_rate*100:,.3f}%")
        c2.metric(f"Förväntad förändring ({horizon_days} dgr)", f"{exp_change_pct:,.2f}%")
        c3.metric("Nuvarande konto", f"{current_eq:,.2f} USDT")
        c4.metric(f"Prognostiserat konto", f"{projected_eq:,.2f} USDT", delta=f"{exp_profit:,.2f} USDT")
else:
    st.info("Ange startkapital + allokering > 0% och kör en simulering.")

# ===== Auto-optimering (grid search) =====
st.divider()
st.markdown("## Auto-optimering (grid search)")

with st.expander("Visa inställningar för optimering", expanded=st.session_state["opt_open"]):
    c1, c2, c3 = st.columns(3)
    with c1:
        opt_stop_periods = st.multiselect("Stop – perioder", [10,20,50], default=st.session_state["opt_stop_periods"])
    with c2:
        opt_stop_tfs = st.multiselect("Stop – timeframes", ["5m","10m","15m","30m","60m"], default=st.session_state["opt_stop_tfs"])
    with c3:
        opt_allow_after = st.checkbox("Tillåt exit efter original (i optimering)", value=st.session_state["allow_after_default"])

    d1, d2 = st.columns(2)
    with d1:
        opt_tp_periods = st.multiselect("TP – perioder (single-TP fallback)", [10,20,50], default=st.session_state["opt_tp_periods"])
    with d2:
        opt_tp_tfs = st.multiselect("TP – timeframes (single-TP fallback)", ["5m","10m","15m","30m","60m"], default=st.session_state["opt_tp_tfs"])

    st.markdown("### Multi-TP – optimering")
    opt_multi_tp_on = st.checkbox("Aktivera Multi-TP i optimering", value=False,
                                  help="Kör grid även över TP1/TP2/TP3-scenarion. Annars används single-TP fallback.")
    if opt_multi_tp_on:
        st.caption("Fördefinierade scenarion som testas (ändra i sidans state om du vill):")
        st.json(st.session_state["opt_multi_tp_scenarios"])

    opt_post_extend_days = st.number_input("Sökfönster i optimering (dagar)", 1, 90, int(post_extend_days), 1)

    st.session_state["opt_stop_periods"] = list(opt_stop_periods)
    st.session_state["opt_stop_tfs"] = list(opt_stop_tfs)
    st.session_state["opt_tp_periods"] = list(opt_tp_periods)
    st.session_state["opt_tp_tfs"] = list(opt_tp_tfs)
    st.session_state["opt_open"] = True

    run_opt = st.button("Kör optimering")

if run_opt:
    if start_equity <= 0 or alloc <= 0:
        st.warning("Ange startkapital och allokering > 0% för optimering.")
    else:
        with st.spinner("Optimerar kombinationer…"):
            opt_df = optimize_settings(
                trades_df=df, start_equity=float(start_equity), allocation_pct=float(alloc),
                allow_after_original=bool(opt_allow_after), post_extend_days=int(opt_post_extend_days),
                stop_periods=list(map(int, st.session_state["opt_stop_periods"])),
