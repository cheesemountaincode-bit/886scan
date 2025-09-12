# 886strat
# 0.886 → 0.618 Scanner (Trend-segment + explosivt filter + EMA-surf 10/20/50)
# Källor: Bybit (ccxt), Yahoo (manuell/.ST-beta), Avanza CSV, Stooq (daglig)
# Inkl: multi-pivot, strikt-regel, kvalitetsdiagnostik, och EMA-surf-filter

import statistics
from datetime import datetime, timezone
import re, requests
import streamlit as st
import ccxt
import pandas as pd
import yfinance as yf
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# =========================
# Hjälpfunktioner (generella)
# =========================

def ts(ms):
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def fetch_ohlcv_safe(ex, symbol, timeframe, limit):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

def load_usdt_perp_symbols(ex):
    ex.load_markets()
    return [s for s, m in ex.markets.items()
            if m.get("active", True) and m.get("swap") and m.get("quote") == "USDT"]

def atr_like_vol(ohlcv, lookback):
    vals = []
    for _, o, h, l, c, v in ohlcv[-lookback:]:
        if c:
            vals.append((h - l) / c)
    return statistics.mean(vals) if vals else 0.0

def pct_dist(a, b):
    if b == 0 or a is None or b is None:
        return None
    return (a - b) / b * 100.0

def build_tradingview_url(ccxt_symbol: str):
    try:
        base = ccxt_symbol.split("/")[0]
        quote_part = ccxt_symbol.split("/")[1]
        quote = quote_part.split(":")[0]
    except Exception:
        return None
    tv_symbol = f"BYBIT:{base}{quote}.P"
    return f"https://www.tradingview.com/chart/?symbol={tv_symbol}"

# =========================
# EMA & trendsegment
# =========================

def ema_series(values, length):
    out = [None] * len(values)
    if not values:
        return out
    k = 2 / (length + 1)
    e = values[0]
    for i, v in enumerate(values):
        if i == 0:
            e = v
        else:
            e = v * k + e * (1 - k)
        if i >= length - 1:
            out[i] = e
    return out

def current_trend_and_segment_start(ohlcv):
    closes = [x[4] for x in ohlcv]
    e50 = ema_series(closes, 50)
    e200 = ema_series(closes, 200)
    n = len(closes)
    if n < 210 or e50[-1] is None or e200[-1] is None:
        return None, None
    up_now = (e50[-1] > e200[-1] and closes[-1] > e200[-1])
    down_now = (e50[-1] < e200[-1] and closes[-1] < e200[-1])
    if not up_now and not down_now:
        return None, None
    side = 'up' if up_now else 'down'
    seg_start = None
    for i in range(n - 2, 0, -1):
        if None in (e50[i], e200[i], e50[i-1], e200[i-1]):
            continue
        crossed_up = (e50[i-1] <= e200[i-1]) and (e50[i] > e200[i])
        crossed_down = (e50[i-1] >= e200[i-1]) and (e50[i] < e200[i])
        if side == 'up' and crossed_up:
            seg_start = i
            break
        if side == 'down' and crossed_down:
            seg_start = i
            break
    if seg_start is None:
        seg_start = max(0, n - 400)
    return side, seg_start

# =========================
# Pivot-detektering (segment)
# =========================

def find_first_red_indices_in_segment(ohlcv, seg_start):
    o=[x[1] for x in ohlcv]; c=[x[4] for x in ohlcv]
    idxs=[]
    for i in range(max(2, seg_start+1), len(ohlcv)-1):
        if c[i] < o[i] and (c[i-1] > o[i-1] or c[i-2] > o[i-2]):
            idxs.append(i)
    return idxs

def find_first_green_indices_in_segment(ohlcv, seg_start):
    o=[x[1] for x in ohlcv]; c=[x[4] for x in ohlcv]
    idxs=[]
    for i in range(max(2, seg_start+1), len(ohlcv)-1):
        if c[i] > o[i] and (c[i-1] < o[i-1] or c[i-2] < o[i-2]):
            idxs.append(i)
    return idxs

# =========================
# Setup från pivot
# =========================

def setup_up_from_index_segment(ohlcv, r_idx, segment_start, include_turn_wick=True,
                                use_segment_extreme=False):
    if r_idx is None or r_idx <= segment_start:
        return None
    o=[x[1] for x in ohlcv]; h=[x[2] for x in ohlcv]; l=[x[3] for x in ohlcv]; c=[x[4] for x in ohlcv]
    high_slice_end = r_idx + (1 if include_turn_wick else 0)
    if use_segment_extreme:
        if high_slice_end <= segment_start:
            return None
        top_price = max(h[segment_start:high_slice_end])
    else:
        j = r_idx - 1
        if j < segment_start:
            return None
        first_green = j
        while first_green >= segment_start and c[first_green] > o[first_green]:
            first_green -= 1
        first_green += 1
        if first_green > j:
            return None
        top_price = max(h[first_green:high_slice_end])
    bottom_low = l[r_idx]
    freeze_idx = r_idx
    k = r_idx + 1
    while k < len(ohlcv) and c[k] < o[k]:
        bottom_low = min(bottom_low, l[k]); freeze_idx = k; k += 1
    span = top_price - bottom_low
    if span <= 0: return None
    level886 = bottom_low + 0.886 * span
    level618 = bottom_low + 0.618 * span
    return {"mode":"up","first_idx":r_idx,"freeze_idx":freeze_idx,
            "top":top_price,"bottom":bottom_low,"span":span,
            "level886":level886,"level618":level618}

def setup_down_from_index_segment(ohlcv, g_idx, segment_start, include_turn_wick=True,
                                  use_segment_extreme=False):
    if g_idx is None or g_idx <= segment_start:
        return None
    o=[x[1] for x in ohlcv]; h=[x[2] for x in ohlcv]; l=[x[3] for x in ohlcv]; c=[x[4] for x in ohlcv]
    low_slice_end = g_idx + (1 if include_turn_wick else 0)
    if use_segment_extreme:
        if low_slice_end <= segment_start: return None
        bottom_low = min(l[segment_start:low_slice_end])
    else:
        j = g_idx - 1
        if j < segment_start: return None
        first_red = j
        while first_red >= segment_start and c[first_red] < o[first_red]:
            first_red -= 1
        first_red += 1
        if first_red > j: return None
        bottom_low = min(l[first_red:low_slice_end])
    top_high = h[g_idx]
    freeze_idx = g_idx
    k = g_idx + 1
    while k < len(ohlcv) and c[k] > o[k]:
        top_high = max(top_high, h[k]); freeze_idx = k; k += 1
    span = top_high - bottom_low
    if span <= 0: return None
    level886 = top_high - 0.886 * span
    level618 = top_high - 0.618 * span
    return {"mode":"down","first_idx":g_idx,"freeze_idx":freeze_idx,
            "top":top_high,"bottom":bottom_low,"span":span,
            "level886":level886,"level618":level618}

# =========================
# Strikt-regel hjälpare
# =========================

def broke_top_since(ohlcv, start_idx, top):
    for i in range(start_idx + 1, len(ohlcv)):
        if ohlcv[i][2] > top:
            return True
    return False

def broke_bottom_since(ohlcv, start_idx, bottom):
    for i in range(start_idx + 1, len(ohlcv)):
        if ohlcv[i][3] < bottom:
            return True
    return False

# =========================
# Sekvensdetektorer (0.886 → 0.618)
# =========================

def detect_up_0886_then_0618(ohlcv, setup,
                             tol_886=0.0006, tol_618=0.0008,
                             invalidate_on_new_low=True,
                             max_bars=0):
    level886=setup["level886"]; level618=setup["level618"]
    bottom=setup["bottom"]; top=setup["top"]; fr=setup["freeze_idx"]
    lo_886 = level886*(1 - tol_886); hi_618 = level618*(1 + tol_618)
    touch_idx=None
    for i in range(fr+1, len(ohlcv)):
        _, o, h, l, c, v = ohlcv[i]
        if h > top: return {"ok":False,"reason":"broke_top_before_0886"}
        if invalidate_on_new_low and l < bottom: return {"ok":False,"reason":"new_low_before_0886"}
        if h >= lo_886: touch_idx=i; break
    if touch_idx is None: return {"ok":False,"reason":"no_0886_touch"}
    retrace_idx=None
    for j in range(touch_idx+1, len(ohlcv)):
        _, o, h, l, c, v = ohlcv[j]
        if h > top: return {"ok":False,"reason":"broke_top_before_0618"}
        if invalidate_on_new_low and l < bottom: return {"ok":False,"reason":"new_low_before_0618"}
        if l <= hi_618: retrace_idx=j; break
        if max_bars and (j - touch_idx) > max_bars:
            return {"ok":False,"reason":"too_many_bars_between_touch_and_retrace"}
    if retrace_idx is None: return {"ok":False,"reason":"no_0618_retrace_after_0886"}
    if broke_top_since(ohlcv, fr, top): return {"ok":False,"reason":"broke_top_after_sequence"}
    return {"ok":True,"touch_idx":touch_idx,"retrace_idx":retrace_idx}

def detect_down_0886_then_0618(ohlcv, setup,
                               tol_886=0.0006, tol_618=0.0008,
                               invalidate_on_new_high=True,
                               max_bars=0):
    level886=setup["level886"]; level618=setup["level618"]
    top=setup["top"]; bottom=setup["bottom"]; fr=setup["freeze_idx"]
    hi_886 = level886*(1 + tol_886); lo_618 = level618*(1 - tol_618)
    touch_idx=None
    for i in range(fr+1, len(ohlcv)):
        _, o, h, l, c, v = ohlcv[i]
        if l < bottom: return {"ok":False,"reason":"broke_bottom_before_0886"}
        if invalidate_on_new_high and h > top: return {"ok":False,"reason":"new_high_before_0886"}
        if l <= hi_886: touch_idx=i; break
    if touch_idx is None: return {"ok":False,"reason":"no_0886_touch"}
    retrace_idx=None
    for j in range(touch_idx+1, len(ohlcv)):
        _, o, h, l, c, v = ohlcv[j]
        if l < bottom: return {"ok":False,"reason":"broke_bottom_before_0618"}
        if invalidate_on_new_high and h > top: return {"ok":False,"reason":"new_high_before_0618"}
        if h >= lo_618: retrace_idx=j; break
        if max_bars and (j - touch_idx) > max_bars:
            return {"ok":False,"reason":"too_many_bars_between_touch_and_retrace"}
    if retrace_idx is None: return {"ok":False,"reason":"no_0618_retrace_after_0886"}
    if broke_bottom_since(ohlcv, fr, bottom): return {"ok":False,"reason":"broke_bottom_after_sequence"}
    return {"ok":True,"touch_idx":touch_idx,"retrace_idx":retrace_idx}

# =========================
# Explosive-move detection
# =========================

def _ema(values, n):
    if not values: return []
    k = 2/(n+1); out=[]; e=None
    for v in values:
        e = v if e is None else v*k + e*(1-k)
        out.append(e)
    return out

def _rolling_atr_like(ohlcv, n=14):
    h=[x[2] for x in ohlcv]; l=[x[3] for x in ohlcv]; c=[x[4] for x in ohlcv]
    tr=[(h[i]-l[i])/max(c[i],1e-12) for i in range(len(ohlcv))]
    return _ema(tr, n)

def detect_explosive_windows_dir(ohlcv, win=3, min_move_pct=6.0, atr_mult=1.8, vol_mult=2.0):
    n=len(ohlcv)
    if n < win+25: return set(), set()
    c=[x[4] for x in ohlcv]; v=[x[5] for x in ohlcv]
    atr14=_rolling_atr_like(ohlcv, 14); v_ema20=_ema(v, 20)
    up=set(); down=set()
    for end in range(win-1, n):
        start=end-(win-1)
        c0=max(c[start],1e-12)
        c_move=(c[end]-c[start])/c0*100.0
        tr_mean=sum((ohlcv[i][2]-ohlcv[i][3])/max(c[i],1e-12) for i in range(start,end+1))/win
        atr_ref = atr14[end-1] if (end-1) < len(atr14) and atr14[end-1] is not None else (atr14[end] if end < len(atr14) else None)
        if atr_ref is None: continue
        vol_sum=sum(v[i] for i in range(start,end+1))
        vol_ref=(v_ema20[end] or 0.0)*win
        if vol_ref<=0: continue
        if abs(c_move) >= min_move_pct and tr_mean >= atr_mult*atr_ref and vol_sum >= vol_mult*vol_ref:
            if c_move > 0: up.add(end)
            elif c_move < 0: down.add(end)
    return up, down

def has_burst_before(index, bursts_set, lookback=20):
    lo=max(0, index-lookback)
    for j in range(lo, index):
        if j in bursts_set: return True
    return False

# =========================
# Yahoo & Stooq OHLCV
# =========================

YF_INTERVAL_MAP = {
    "15m": "15m", "30m": "30m", "1h": "60m", "2h": "60m",
    "4h": "60m", "6h": "60m", "12h": "60m", "1d": "1d"
}

def fetch_ohlcv_yf(symbol, timeframe, limit):
    interval = YF_INTERVAL_MAP.get(timeframe, None)
    if interval is None:
        raise ValueError(f"Timeframe {timeframe} stöds ej av Yahoo (stöd: 15m/30m/1h/1d).")
    period = "60d" if interval.endswith("m") else "10y"
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty: return []
    df = df.dropna()
    if limit and len(df) > limit: df = df.iloc[-limit:]
    out = []
    for ts_idx, row in df.iterrows():
        ms = int(pd.Timestamp(ts_idx).tz_localize(None).timestamp() * 1000)
        o = float(row["Open"]); h = float(row["High"]); l = float(row["Low"]); c = float(row["Close"]); v = float(row["Volume"])
        out.append([ms, o, h, l, c, v])
    return out

def fetch_ohlcv_stooq(symbol, timeframe, limit):
    if pdr is None:
        raise RuntimeError("pandas_datareader saknas. Installera: pip install pandas-datareader")
    if timeframe != "1d":
        raise ValueError("Stooq stöder endast '1d' i denna app.")
    try:
        df = pdr.DataReader(symbol, "stooq")
    except Exception:
        return []
    if df is None or df.empty: return []
    df = df.sort_index()
    if limit and len(df) > limit: df = df.iloc[-limit:]
    out = []
    for ts_idx, row in df.iterrows():
        ms = int(pd.Timestamp(ts_idx).tz_localize(None).timestamp() * 1000)
        o = float(row["Open"]); h = float(row["High"]); l = float(row["Low"]); c = float(row["Close"]); v = float(row.get("Volume", 0.0) or 0.0)
        out.append([ms, o, h, l, c, v])
    return out

# =========================
# Avanza CSV & Yahoo (alla .ST)
# =========================

def load_avanza_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=";")
    except Exception:
        df = pd.read_csv(uploaded_file)
    tickers = []
    for col in ["Beteckning", "Namn", "Ticker", "Symbol"]:
        if col in df.columns:
            tickers = df[col].dropna().astype(str).str.strip().unique().tolist()
            break
    out = []
    for t in tickers:
        t = t.strip()
        if "." in t: out.append(t)
        else: out.append(f"{t}.ST")
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_yahoo_st_tickers():
    urls = [
        "https://en.wikipedia.org/wiki/NASDAQ_OMX_Stockholm",
        "https://en.wikipedia.org/wiki/OMX_Stockholm_30",
        "https://sv.wikipedia.org/wiki/Nasdaq_Stockholm",
        "https://en.wikipedia.org/wiki/OMX_Stockholm_Large_Cap",
        "https://en.wikipedia.org/wiki/OMX_Stockholm_Mid_Cap",
        "https://en.wikipedia.org/wiki/OMX_Stockholm_Small_Cap",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; scanner/1.0)"}
    found = set()
    def norm(x: str) -> str:
        x = x.strip().upper().replace(" ", "")
        if not x: return ""
        if not x.endswith(".ST"): x = f"{x}.ST"
        return x
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200 or not resp.text: continue
            try:
                tables = pd.read_html(resp.text)
                for df in tables:
                    for col in df.columns:
                        col_l = str(col).lower()
                        if any(k in col_l for k in ["ticker","symbol","short","kod","beteckning"]):
                            vals = df[col].dropna().astype(str).str.replace(r"[*\s]", "", regex=True).tolist()
                            for v in vals:
                                if not v: continue
                                if v.endswith(".ST"): found.add(v.upper())
                                else: found.add(norm(v))
            except Exception:
                pass
            for m in re.finditer(r"\b([A-ZÅÄÖ0-9\-]{1,12})\.ST\b", resp.text, flags=re.IGNORECASE):
                found.add(m.group(0).upper())
        except Exception:
            continue
    cleaned = sorted({t for t in found if len(t) > 3 and t.endswith(".ST")})
    return cleaned

# =========================
# Diagnostik / kvalitet
# =========================

def rolling_atr_points(ohlcv, n=14):
    h=[x[2] for x in ohlcv]; l=[x[3] for x in ohlcv]
    tr=[(h[i]-l[i]) for i in range(len(ohlcv))]
    return _ema(tr, n)

def advance_ratio(ohlcv, i0, i1, up=True):
    o=[x[1] for x in ohlcv]; c=[x[4] for x in ohlcv]
    if i1 <= i0: return 0.0
    seg = range(max(0,i0), min(len(ohlcv)-1, i1)+1)
    if up:
        greens = sum(1 for i in seg if c[i] > o[i])
        total  = len(list(seg))
        return greens/total if total else 0.0
    else:
        reds = sum(1 for i in seg if c[i] < o[i])
        total = len(list(seg))
        return reds/total if total else 0.0

def compute_quality_and_debug(ohlcv, setup, seq, bursts_set, burst_win, burst_lookback, mode):
    atr_pts = rolling_atr_points(ohlcv, 14)
    fr = setup["freeze_idx"]; touch = seq["touch_idx"]
    span_pts = abs(setup["top"] - setup["bottom"])
    atr_here = atr_pts[touch] if touch < len(atr_pts) and atr_pts[touch] else (atr_pts[fr] if fr < len(atr_pts) else None)
    span_atr_mult = (span_pts / atr_here) if atr_here and atr_here > 0 else None
    burst_end = None
    if bursts_set:
        for j in range(max(0, setup["first_idx"]-burst_lookback), setup["first_idx"]):
            if j in bursts_set: burst_end = j
    bars_burst_to_pivot = (setup["first_idx"] - burst_end) if burst_end is not None else None
    burst_move = None
    if burst_end is not None:
        c=[x[4] for x in ohlcv]; start = max(0, burst_end-(burst_win-1))
        burst_move = (c[burst_end]-c[start])/max(c[start],1e-12)*100.0
    adv = advance_ratio(ohlcv, fr+1, touch, up=(mode=="up"))
    score = 0.0
    if span_atr_mult is not None: score += max(0.0, min(1.0, (span_atr_mult-1.0)/3.0)) * 40
    if burst_move is not None:    score += max(0.0, min(1.0, (abs(burst_move)-4.0)/6.0)) * 25
    score += max(0.0, min(1.0, adv)) * 20
    score += max(0, 10 - max(0, seq["retrace_idx"] - seq["touch_idx"])) * 1.5
    return {
        "Span/ATR14": None if span_atr_mult is None else float(f"{span_atr_mult:.2f}"),
        "Burst→Pivot bars": bars_burst_to_pivot,
        "Burst move %": None if burst_move is None else float(f"{burst_move:.2f}"),
        "Advance ratio": float(f"{adv:.2f}"),
        "Quality": float(f"{score:.1f}")
    }

# =========================
# EMA-surf (10/20/50) – nytt
# =========================

def check_ema_surf(ohlcv, i0, i1, mode, close_only, allow10, allow20, strict50=True):
    """
    Kollar att pris "surfar" på EMA10/20 (få brott) och aldrig bryter EMA50 om strict50=True.
    mode='up'  : vill ha C(eller L) >= EMA10/20 mestadels; aldrig under EMA50
    mode='down': vill ha C(eller H) <= EMA10/20 mestadels; aldrig över EMA50
    Returnerar (ok, breaches10, breaches20, breaches50)
    """
    if i1 <= i0: return True, 0, 0, 0
    c=[x[4] for x in ohlcv]; h=[x[2] for x in ohlcv]; l=[x[3] for x in ohlcv]
    e10 = ema_series(c, 10); e20 = ema_series(c, 20); e50 = ema_series(c, 50)
    b10=b20=b50=0
    for i in range(max(0,i0), min(len(ohlcv)-1, i1)+1):
        if e50[i] is None or e20[i] is None or e10[i] is None: continue
        if mode == "up":
            ref10 = (c[i] if close_only else l[i]); ref20 = (c[i] if close_only else l[i]); ref50 = (c[i] if close_only else l[i])
            if ref10 < e10[i]: b10 += 1
            if ref20 < e20[i]: b20 += 1
            if strict50 and ref50 < e50[i]: b50 += 1
        else:
            ref10 = (c[i] if close_only else h[i]); ref20 = (c[i] if close_only else h[i]); ref50 = (c[i] if close_only else h[i])
            if ref10 > e10[i]: b10 += 1
            if ref20 > e20[i]: b20 += 1
            if strict50 and ref50 > e50[i]: b50 += 1
    ok = (b10 <= allow10) and (b20 <= allow20) and (b50 == 0 if strict50 else True)
    return ok, b10, b20, b50

# =========================
# Multi-pivot sök (med explosivt filter)
# =========================

def find_best_up_hit_segment(ohlcv, segment_start, include_turn_wick,
                             use_segment_extreme,
                             tol_886, tol_618, invalidate_on_new_low,
                             max_bars, require_under_top_now,
                             use_explosive=False, bursts_up=None, burst_lookback=20):
    closes = [x[4] for x in ohlcv]; last_px = closes[-1] if closes else None
    candidates = find_first_red_indices_in_segment(ohlcv, segment_start)
    for r_idx in reversed(candidates):
        if use_explosive and (not bursts_up or not has_burst_before(r_idx, bursts_up, burst_lookback)):
            continue
        setup = setup_up_from_index_segment(ohlcv, r_idx, segment_start,
                                            include_turn_wick=include_turn_wick,
                                            use_segment_extreme=use_segment_extreme)
        if not setup: continue
        seq = detect_up_0886_then_0618(ohlcv, setup, tol_886=tol_886, tol_618=tol_618,
                                       invalidate_on_new_low=invalidate_on_new_low, max_bars=max_bars)
        if not seq.get("ok"): continue
        if require_under_top_now and (last_px is None or not (last_px < setup["top"])): continue
        return setup, seq
    return None, None

def find_best_down_hit_segment(ohlcv, segment_start, include_turn_wick,
                               use_segment_extreme,
                               tol_886, tol_618, invalidate_on_new_high,
                               max_bars, require_over_bottom_now,
                               use_explosive=False, bursts_down=None, burst_lookback=20):
    closes = [x[4] for x in ohlcv]; last_px = closes[-1] if closes else None
    candidates = find_first_green_indices_in_segment(ohlcv, segment_start)
    for g_idx in reversed(candidates):
        if use_explosive and (not bursts_down or not has_burst_before(g_idx, bursts_down, burst_lookback)):
            continue
        setup = setup_down_from_index_segment(ohlcv, g_idx, segment_start,
                                              include_turn_wick=include_turn_wick,
                                              use_segment_extreme=use_segment_extreme)
        if not setup: continue
        seq = detect_down_0886_then_0618(ohlcv, setup, tol_886=tol_886, tol_618=tol_618,
                                         invalidate_on_new_high=invalidate_on_new_high, max_bars=max_bars)
        if not seq.get("ok"): continue
        if require_over_bottom_now and (last_px is None or not (last_px > setup["bottom"])): continue
        return setup, seq
    return None, None

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="0.886→0.618 Scanner (Bybit/Yahoo/Stooq/Avanza)", layout="wide")
st.title("0.886 → 0.618 Scanner (Trend-segment + explosivt filter + EMA-surf)")

with st.sidebar:
    st.subheader("Inställningar")

    data_source = st.selectbox(
        "Datakälla",
        [
            "Bybit (USDT-perps, ccxt)",
            "Yahoo Finance (aktier/ETF/cert)",
            "Yahoo Finance (alla .ST, beta)",
            "Avanza (CSV-upload)",
            "Stooq (aktier/ETF, daglig)"
        ],
        index=0
    )

    if data_source == "Bybit (USDT-perps, ccxt)":
        timeframe = st.selectbox("Timeframe", ["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"], index=3)
    elif data_source in ("Yahoo Finance (aktier/ETF/cert)", "Yahoo Finance (alla .ST, beta)", "Avanza (CSV-upload)"):
        timeframe = st.selectbox("Timeframe (Yahoo/Avanza)", ["15m","30m","1h","1d"], index=0)
    else:
        timeframe = "1d"
        st.info("Stooq stöder daglig data (1d).")

    # Symboler
    yf_symbols = []; stooq_symbols = []; ranked = []

    if data_source == "Bybit (USDT-perps, ccxt)":
        scan_all = st.checkbox("Skanna ALLA USDT-perps (ignorera vol-ranking)", value=True)
        lookback_vol = st.number_input("Volatilitetsfönster (om vol-ranking används)", 50, 2000, 288, 10)
        top_n = st.number_input("Om ej ALLA: antal mest volatila", 5, 200, 50, 5)

    elif data_source == "Yahoo Finance (aktier/ETF/cert)":
        st.caption("Yahoo-tickers separerade med kommatecken. Ex: VOLV-B.ST, HM-B.ST, XACT-BEAR.ST")
        symbols_text = st.text_area("Tickers (Yahoo)", value="VOLV-B.ST, HM-B.ST", height=70)
        yf_symbols = [s.strip() for s in symbols_text.split(",") if s.strip()]

    elif data_source == "Yahoo Finance (alla .ST, beta)":
        st.caption("Hämtar bred lista .ST (kan ta tid). Begränsa gärna antalet.")
        if st.button("Hämta alla .ST"):
            all_st = get_all_yahoo_st_tickers()
            st.session_state["all_st"] = all_st
            if all_st: st.success(f"Laddade {len(all_st)} tickers.")
            else: st.warning("Hittade inga tickers.")
        max_st = st.number_input("Max antal .ST (0 = alla)", 0, 10000, 500, 50)
        prefilter = st.text_input("Prefixfilter (valfritt)", value="")
        manual_fallback = st.text_area("Fallback (.ST separerade med kommatecken)", value="", height=70)
        current_list = st.session_state.get("all_st", [])
        if manual_fallback.strip():
            current_list = [x.strip().upper() for x in manual_fallback.split(",") if x.strip()]
        if prefilter:
            current_list = [t for t in current_list if t.upper().startswith(prefilter.upper())]
        if max_st and max_st > 0: current_list = current_list[:max_st]
        if current_list: st.caption(f"Aktuell lista: {len(current_list)} tickers")
        yf_symbols = current_list

    elif data_source == "Avanza (CSV-upload)":
        uploaded_file = st.file_uploader("Ladda upp Avanza CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                yf_symbols = load_avanza_csv(uploaded_file)
                st.success(f"Hittade {len(yf_symbols)} tickers i CSV-filen.")
            except Exception as e:
                st.error(f"Kunde inte läsa CSV: {e}")
        st.caption("Vi lägger på .ST om suffix saknas.")

    else:
        st.caption("Stooq är daglig data. Exempel: AAPL.US, MSFT.US")
        symbols_text = st.text_area("Tickers (Stooq, daglig)", value="AAPL.US, MSFT.US", height=70)
        stooq_symbols = [s.strip() for s in symbols_text.split(",") if s.strip()]

    # Top/Bottom-källa
    top_source = st.selectbox(
        "Källa för Top/Bottom",
        ["Lokalt block (precis före vändning)", "Högsta/lägsta i hela trendsegmentet"],
        index=1
    )
    use_segment_extreme = (top_source == "Högsta/lägsta i hela trendsegmentet")

    # Setup/sekvens-parametrar
    top_lookback = st.number_input("Max bakåt för lokalt block (candles)", 5, 2000, 120, 5)
    include_turn_wick = st.checkbox("Inkludera vändnings-wick i Top/Bottom", value=True)

    tol_886 = st.number_input("Tolerans 0.886 (relativ)", 0.0, 0.02, 0.0006, 0.0001, format="%.4f")
    tol_618 = st.number_input("Tolerans 0.618 (relativ)", 0.0, 0.02, 0.0008, 0.0001, format="%.4f")
    max_bars = st.number_input("Max candles touch→retrace (0 = ingen gräns)", 0, 1000, 0, 1)

    # Extraregler
    invalidate_low  = st.checkbox("Upptrend: ogiltig vid ny lägre low innan 0.618", True)
    invalidate_high = st.checkbox("Nedtrend: ogiltig vid ny högre high innan 0.618", True)
    require_under_top_now   = st.checkbox("Krav: Last < Top (upptrend)", True)
    require_over_bottom_now = st.checkbox("Krav: Last > Bottom (nedtrend)", True)

    # Trend-läge
    trend_mode = st.selectbox("Trendläge", ["Auto (EMA50/200)", "Upptrend", "Nedtrend"], index=0)
    allow_range_scan = st.checkbox("Tillåt scanning i range (om ingen trend detekteras)", value=False)

    # Explosivt filter + presets
    st.markdown("— Explosivt filter —")
    use_explosive = st.checkbox("Kräv explosiv rörelse före setup", value=True)
    preset = st.selectbox(
        "Preset för explosiv-filter",
        ["Custom", "BTC/ETH högre TF (8–10% & 2.0–2.2)", "Alts intraday (4–6% & 1.6–1.8)"],
        index=0
    )
    default_min_move = 6.0; default_atr_mult = 1.8
    if preset == "BTC/ETH högre TF (8–10% & 2.0–2.2)":
        default_min_move = 9.0; default_atr_mult = 2.1
    elif preset == "Alts intraday (4–6% & 1.6–1.8)":
        default_min_move = 5.0; default_atr_mult = 1.7
    burst_win = st.number_input("Burst-fönster (bars)", 2, 12, 3, 1)
    burst_min_move = st.number_input("Min % flytt (min_move_pct)", 1.0, 30.0, default_min_move, 0.5)
    burst_atr_mult = st.number_input("ATR-multipel (atr_mult) vs ATR(14)", 0.5, 5.0, default_atr_mult, 0.1)
    burst_vol_mult = st.number_input("Volym-multipel vs EMA(20)", 0.5, 6.0, 2.0, 0.1)
    burst_lookback = st.number_input("Max bars pivot efter burst", 5, 150, 20, 1)
    enforce_min_tf_15m = st.checkbox("Blockera explosivt filter under 15m", value=False)

    # Formfilter / diagnostik
    st.markdown("— Formfilter / diagnostik —")
    show_diag = st.checkbox("Visa diagnostik/quality-kolumner", value=True)
    min_quality = st.number_input("Min quality (0–100)", 0.0, 100.0, 50.0, 1.0)
    min_span_atr = st.number_input("Min Span/ATR14", 0.0, 20.0, 1.2, 0.1)
    min_adv_ratio_up = st.number_input("Min advance ratio (upptrend)", 0.0, 1.0, 0.60, 0.05)
    min_adv_ratio_down = st.number_input("Min advance ratio (nedtrend)", 0.0, 1.0, 0.60, 0.05)
    max_burst_to_pivot = st.number_input("Max bars: burst→pivot (0=ignorera)", 0, 500, 40, 1)

    # EMA-surf-filter (NYTT)
    st.markdown("— EMA-surf (10/20/50) —")
    ema_surf_enable = st.checkbox("Kräv EMA-surf före och efter 0.886-touch", value=True)
    ema_close_only = st.checkbox("Endast closes (tillåt wicks genom EMA)", value=True)
    ema_allow10 = st.number_input("Max brott mot EMA10 (per hel sekvens)", 0, 50, 2, 1)
    ema_allow20 = st.number_input("Max brott mot EMA20 (per hel sekvens)", 0, 50, 1, 1)
    ema_strict50 = st.checkbox("Aldrig bryta EMA50 (strikt)", value=True)

    # Datahämtning
    limit_candles = st.number_input("Candles att hämta per symbol", 300, 3000, 800, 50)
    run_btn = st.button("Skanna nu")

status = st.empty()
progress = st.progress(0, text="Väntar …")

# =========================
# Körning
# =========================

if run_btn:
    try:
        ranked = []; exchange = None
        if data_source == "Bybit (USDT-perps, ccxt)":
            exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})
            exchange.load_markets()
            all_syms = load_usdt_perp_symbols(exchange) or []
            if scan_all:
                ranked = all_syms[:]; st.caption(f"Skannar ALLA USDT-perps ({len(ranked)})")
            else:
                st.info("Rankar volatilitet …")
                vols = []
                for idx, s in enumerate(all_syms):
                    try:
                        ohlcv_tmp = fetch_ohlcv_safe(exchange, s, timeframe, limit=max(lookback_vol, 300))
                        vols.append((atr_like_vol(ohlcv_tmp, lookback_vol), s))
                    except Exception:
                        pass
                    progress.progress(int((idx+1)/max(1,len(all_syms))*100), text=f"Vol-rankar {idx+1}/{len(all_syms)}")
                vols.sort(reverse=True); ranked = [s for _, s in vols[:top_n]]
                st.caption(f"Topp {top_n} mest volatila ({timeframe})")
        elif data_source in ("Yahoo Finance (aktier/ETF/cert)", "Yahoo Finance (alla .ST, beta)", "Avanza (CSV-upload)"):
            ranked = yf_symbols[:]; st.caption(f"Skannar {len(ranked)} Yahoo/Avanza-tickers")
        else:
            ranked = stooq_symbols[:]; st.caption(f"Skannar {len(ranked)} Stooq-tickers (daglig)")

        hits = []
        for i, s in enumerate(ranked):
            try:
                # OHLCV
                if data_source == "Bybit (USDT-perps, ccxt)":
                    ohlcv = fetch_ohlcv_safe(exchange, s, timeframe, limit=limit_candles)
                elif data_source in ("Yahoo Finance (aktier/ETF/cert)", "Yahoo Finance (alla .ST, beta)", "Avanza (CSV-upload)"):
                    ohlcv = fetch_ohlcv_yf(s, timeframe, limit=limit_candles)
                else:
                    ohlcv = fetch_ohlcv_stooq(s, timeframe, limit=limit_candles)
                if not ohlcv or len(ohlcv) < 210:
                    progress.progress(int((i+1)/max(1,len(ranked))*100), text=f"Skannar {i+1}/{len(ranked)}")
                    continue

                closes = [x[4] for x in ohlcv]; last_px = closes[-1]
                side, seg_start = current_trend_and_segment_start(ohlcv)

                if trend_mode == "Upptrend":
                    do_up, do_down = True, False
                elif trend_mode == "Nedtrend":
                    do_up, do_down = False, True
                else:
                    if side == 'up': do_up, do_down = True, False
                    elif side == 'down': do_up, do_down = False, True
                    else:
                        if allow_range_scan:
                            do_up, do_down = True, True; seg_start = max(0, len(ohlcv) - 300)
                        else:
                            do_up, do_down = False, False

                # Bursts
                bursts_up, bursts_down = set(), set()
                if use_explosive:
                    if enforce_min_tf_15m and timeframe in ("1m","3m","5m","10m"):
                        bursts_up, bursts_down = set(), set()
                    else:
                        bursts_up, bursts_down = detect_explosive_windows_dir(
                            ohlcv, win=burst_win, min_move_pct=burst_min_move,
                            atr_mult=burst_atr_mult, vol_mult=burst_vol_mult
                        )

                # === Upptrend ===
                if do_up and seg_start is not None:
                    setup, seq = find_best_up_hit_segment(
                        ohlcv, seg_start, include_turn_wick,
                        use_segment_extreme,
                        tol_886, tol_618, invalidate_on_new_low=invalidate_low,
                        max_bars=max_bars, require_under_top_now=require_under_top_now,
                        use_explosive=use_explosive, bursts_up=bursts_up, burst_lookback=burst_lookback
                    )
                    if setup and seq:
                        fr = setup["freeze_idx"]; t_idx = seq["touch_idx"]; r_idx = seq["retrace_idx"]
                        # EMA-surf filter (både freeze→touch och touch→retrace)
                        ema_ok=True; b10=b20=b50=0
                        if ema_surf_enable:
                            ok1, a10, a20, a50 = check_ema_surf(ohlcv, fr+1, t_idx, "up", ema_close_only, ema_allow10, ema_allow20, ema_strict50)
                            ok2, b10_, b20_, b50_ = check_ema_surf(ohlcv, t_idx+1, r_idx, "up", ema_close_only, ema_allow10, ema_allow20, ema_strict50)
                            ema_ok = ok1 and ok2
                            b10=a10+b10_; b20=a20+b20_; b50=a50+b50_
                        if not ema_ok:
                            pass
                        else:
                            tbar = ohlcv[t_idx]; rbar = ohlcv[r_idx]
                            low_lvl = min(setup["level618"], setup["level886"]); hi_lvl = max(setup["level618"], setup["level886"])
                            tv_url = build_tradingview_url(s) if data_source == "Bybit (USDT-perps, ccxt)" else None
                            diag = compute_quality_and_debug(ohlcv, setup, seq, bursts_up, burst_win, burst_lookback, mode="up")
                            reject = False
                            if diag["Span/ATR14"] is not None and diag["Span/ATR14"] < min_span_atr: reject = True
                            if not reject and max_burst_to_pivot and diag["Burst→Pivot bars"] is not None and diag["Burst→Pivot bars"] > max_burst_to_pivot: reject = True
                            if not reject and diag["Advance ratio"] < min_adv_ratio_up: reject = True
                            if not reject and diag["Quality"] < min_quality: reject = True
                            if not reject:
                                rec = {
                                    "Symbol": s, "TF": timeframe, "Trend": "Up",
                                    "Top": setup["top"], "Bottom": setup["bottom"],
                                    "0.886": setup["level886"], "0.618": setup["level618"],
                                    "Touch 0.886 (UTC)": ts(tbar[0]),
                                    "Retrace 0.618 (UTC)": ts(rbar[0]),
                                    "Bars touch→retrace": r_idx - t_idx,
                                    "Last": last_px,
                                    "Dist% till 0.886": pct_dist(last_px, setup["level886"]),
                                    "Dist% till 0.618": pct_dist(last_px, setup["level618"]),
                                    "Mellan 0.618–0.886 nu": bool(low_lvl <= last_px <= hi_lvl),
                                    "Under TOP nu": (last_px < setup["top"]),
                                    "Span/ATR14": diag["Span/ATR14"],
                                    "Advance ratio": diag["Advance ratio"],
                                    "Quality": diag["Quality"],
                                    "EMA surf OK": ema_ok,
                                    "Breach10": b10, "Breach20": b20, "Breach50": b50
                                }
                                if show_diag:
                                    rec["Burst→Pivot bars"] = diag["Burst→Pivot bars"]
                                    rec["Burst move %"] = diag["Burst move %"]
                                if tv_url: rec = {"TradingView": tv_url, **rec}
                                hits.append(rec)

                # === Nedtrend ===
                if do_down and seg_start is not None:
                    setup, seq = find_best_down_hit_segment(
                        ohlcv, seg_start, include_turn_wick,
                        use_segment_extreme,
                        tol_886, tol_618, invalidate_on_new_high=invalidate_high,
                        max_bars=max_bars, require_over_bottom_now=require_over_bottom_now,
                        use_explosive=use_explosive, bursts_down=bursts_down, burst_lookback=burst_lookback
                    )
                    if setup and seq:
                        fr = setup["freeze_idx"]; t_idx = seq["touch_idx"]; r_idx = seq["retrace_idx"]
                        ema_ok=True; b10=b20=b50=0
                        if ema_surf_enable:
                            ok1, a10, a20, a50 = check_ema_surf(ohlcv, fr+1, t_idx, "down", ema_close_only, ema_allow10, ema_allow20, ema_strict50)
                            ok2, b10_, b20_, b50_ = check_ema_surf(ohlcv, t_idx+1, r_idx, "down", ema_close_only, ema_allow10, ema_allow20, ema_strict50)
                            ema_ok = ok1 and ok2
                            b10=a10+b10_; b20=a20+b20_; b50=a50+b50_
                        if not ema_ok:
                            pass
                        else:
                            tbar = ohlcv[t_idx]; rbar = ohlcv[r_idx]
                            low_lvl = min(setup["level618"], setup["level886"]); hi_lvl = max(setup["level618"], setup["level886"])
                            tv_url = build_tradingview_url(s) if data_source == "Bybit (USDT-perps, ccxt)" else None
                            diag = compute_quality_and_debug(ohlcv, setup, seq, bursts_down, burst_win, burst_lookback, mode="down")
                            reject = False
                            if diag["Span/ATR14"] is not None and diag["Span/ATR14"] < min_span_atr: reject = True
                            if not reject and max_burst_to_pivot and diag["Burst→Pivot bars"] is not None and diag["Burst→Pivot bars"] > max_burst_to_pivot: reject = True
                            if not reject and diag["Advance ratio"] < min_adv_ratio_down: reject = True
                            if not reject and diag["Quality"] < min_quality: reject = True
                            if not reject:
                                rec = {
                                    "Symbol": s, "TF": timeframe, "Trend": "Down",
                                    "Top": setup["top"], "Bottom": setup["bottom"],
                                    "0.886": setup["level886"], "0.618": setup["level618"],
                                    "Touch 0.886 (UTC)": ts(tbar[0]),
                                    "Retrace 0.618 (UTC)": ts(rbar[0]),
                                    "Bars touch→retrace": r_idx - t_idx,
                                    "Last": last_px,
                                    "Dist% till 0.886": pct_dist(last_px, setup["level886"]),
                                    "Dist% till 0.618": pct_dist(last_px, setup["level618"]),
                                    "Mellan 0.618–0.886 nu": bool(low_lvl <= last_px <= hi_lvl),
                                    "Över BOTTOM nu": (last_px > setup["bottom"]),
                                    "Span/ATR14": diag["Span/ATR14"],
                                    "Advance ratio": diag["Advance ratio"],
                                    "Quality": diag["Quality"],
                                    "EMA surf OK": ema_ok,
                                    "Breach10": b10, "Breach20": b20, "Breach50": b50
                                }
                                if show_diag:
                                    rec["Burst→Pivot bars"] = diag["Burst→Pivot bars"]
                                    rec["Burst move %"] = diag["Burst move %"]
                                if tv_url: rec = {"TradingView": tv_url, **rec}
                                hits.append(rec)

            except Exception:
                pass

            progress.progress(int((i+1)/max(1,len(ranked))*100), text=f"Skannar {i+1}/{len(ranked)}")

        # Resultat
        if hits:
            df = pd.DataFrame(hits)
            float_cols = ["Top","Bottom","0.886","0.618","Last","Dist% till 0.886","Dist% till 0.618",
                          "Burst move %","Span/ATR14","Advance ratio","Quality"]
            for col in float_cols:
                if col in df.columns:
                    if col in ("Burst move %","Span/ATR14","Advance ratio","Quality"):
                        df[col] = df[col].map(lambda x: None if x is None else float(x))
                    else:
                        df[col] = df[col].map(lambda x: None if x is None else float(f"{x:.6f}"))
            sort_cols = [c for c in ["EMA surf OK","Quality","Mellan 0.618–0.886 nu","Dist% till 0.886"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(by=sort_cols, ascending=[False, False, False, True], kind="mergesort")
            preferred_order = ["TradingView","Symbol","TF","Trend","Top","Bottom","0.886","0.618",
                               "Touch 0.886 (UTC)","Retrace 0.618 (UTC)","Bars touch→retrace",
                               "Last","Dist% till 0.886","Dist% till 0.618",
                               "EMA surf OK","Breach10","Breach20","Breach50",
                               "Span/ATR14","Advance ratio","Quality","Burst move %","Burst→Pivot bars",
                               "Mellan 0.618–0.886 nu","Under TOP nu","Över BOTTOM nu"]
            cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
            df = df[cols]
            st.success(f"Klar. Träffar: {len(df)}")
            col_config = {}
            if "TradingView" in df.columns:
                col_config["TradingView"] = st.column_config.LinkColumn("TV", help="Öppna i TradingView (BYBIT perps)", display_text="📈")
            st.data_editor(df, hide_index=True, use_container_width=True, column_config=col_config, disabled=True)
        else:
            st.info("Inga symboler uppfyllde villkoren i vald källa/timeframe.")
    except Exception as e:
        st.error(f"Fel: {e}")
