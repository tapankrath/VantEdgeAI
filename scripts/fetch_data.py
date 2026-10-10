"""
SignalHQ nightly data builder.

Pulls end-of-day options data from Yahoo Finance (via the unofficial `yfinance`
library — free, no API key, but not officially supported by Yahoo and can break
or rate-limit without warning) and computes the fields the SignalHQ UI expects,
writing them to data.json at the repo root.

IMPORTANT — read before trusting the numbers:
- `iv` (implied volatility) comes directly from Yahoo's option chain.
- `delta` is computed here via Black-Scholes, assuming 0% dividend yield and a
  flat risk-free rate (RISK_FREE_RATE below). Real delta from a broker may differ.
- `pot` (probability of touch) uses the common trader heuristic pot ≈ 2 × |delta|,
  not a rigorous barrier-option calculation. Treat it as a rough guide.
- `ivr` (IV Rank) is NOT true IV rank (which needs a year of historical *option*
  IV data, which isn't freely available). It's a proxy built from the percentile
  of recent 20-day realized volatility vs. the past year — correlated with real
  IV rank but not the same number your broker would show.
- `score` (composite rating, 0-10) is an illustrative weighted blend of the above.
  It is not a validated trading signal. Adjust the weights in `composite_score()`
  to match what you actually care about.
- Strategy/strike selection targets a ~0.20 delta short leg, a common informal
  "20-delta" premium-selling convention — not personalized to any risk tolerance.

This script is a starting point, not a finished quant model. Treat every number
it produces as directional, not authoritative, and verify anything before
acting on it.
"""

import json
import math
import os
import re
import sys
import zlib
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import norm

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed — run: pip install -r scripts/requirements.txt", file=sys.stderr)
    raise

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("vaderSentiment not installed — run: pip install -r scripts/requirements.txt", file=sys.stderr)
    raise

# VADER's stock lexicon is tuned for general/social text and badly misreads financial
# headlines — e.g. out of the box it scores "faces lawsuit... shares tumble" as
# slightly POSITIVE, because words like "tumble" and "beats" aren't in its default
# dictionary. This augments it with common financial-news vocabulary so it actually
# reads headlines the way a finance-literate person would. Not exhaustive — extend
# FINANCE_LEXICON below if you notice it missing common terms.
FINANCE_LEXICON = {
    # positive
    "beat": 3.0, "beats": 3.0, "beating": 3.0, "exceeded": 2.8, "exceeds": 2.8,
    "upgrade": 2.5, "upgraded": 2.5, "upgrades": 2.5, "outperform": 2.5,
    "raises": 1.8, "raised": 1.8, "surge": 3.0, "surged": 3.0, "surges": 3.0,
    "rally": 2.5, "rallied": 2.5, "rallies": 2.5, "soar": 3.2, "soared": 3.2, "soars": 3.2,
    "bullish": 2.5, "buyback": 1.8, "record high": 2.8, "accelerate": 1.5,
    "accelerated": 1.5, "breakthrough": 2.5, "guidance raised": 2.5,
    "strong demand": 2.2, "blowout": 3.0, "jumps": 2.2, "jumped": 2.2,
    # negative
    "miss": -3.0, "misses": -3.0, "missed": -3.0, "downgrade": -2.5, "downgraded": -2.5,
    "downgrades": -2.5, "cut": -1.8, "cuts": -1.8, "plunge": -3.2, "plunged": -3.2,
    "plunges": -3.2, "tumble": -3.0, "tumbled": -3.0, "tumbles": -3.0,
    "slump": -2.5, "slumped": -2.5, "bearish": -2.5, "lawsuit": -2.2,
    "investigation": -2.5, "recall": -2.2, "layoffs": -2.5, "bankruptcy": -3.5,
    "default": -3.0, "delisted": -3.0, "fraud": -3.5, "scandal": -3.0,
    "warning": -1.8, "weak demand": -2.2, "slowdown": -1.8, "guidance cut": -2.8,
    "sinks": -2.5, "sank": -2.5, "slides": -1.8,
}

_sentiment_analyzer = SentimentIntensityAnalyzer()
_sentiment_analyzer.lexicon.update(FINANCE_LEXICON)

# --- Configuration -----------------------------------------------------------

UNIVERSE_PATH = "universe.json"
UNIVERSE_STALE_DAYS = 7      # warn (but still use it) past this age
UNIVERSE_META = None         # set by load_universe(); echoed into data.json


def load_universe():
    """
    Reads universe.json (written daily by scripts/build_universe.py): the dynamic
    slice of the watchlist — large US stocks picked by market cap / liquidity
    rather than typed in by hand. Returns a list of symbols, or [] if the file
    is missing or unreadable, so a bad or absent universe can never break the
    run: the pinned tickers.json list is always enough on its own.
    """
    global UNIVERSE_META
    try:
        with open(UNIVERSE_PATH) as f:
            cfg = json.load(f)
        symbols = [t.strip().upper() for t in cfg.get("tickers", []) if isinstance(t, str) and t.strip()]
        generated = cfg.get("generated_at")
        UNIVERSE_META = {
            "generated_at": generated,
            "size": len(symbols),
            "criteria": cfg.get("criteria"),
            "momentum": [m.get("symbol") for m in cfg.get("momentum") or [] if isinstance(m, dict)],
            "pinnedWeak": [p.get("symbol") for p in cfg.get("pinnedReport") or []
                           if isinstance(p, dict) and p.get("options") in ("thin", "none")],
        }
        try:
            age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(generated)).days
            if age_days > UNIVERSE_STALE_DAYS:
                print(f"{UNIVERSE_PATH} is {age_days} days old — is the universe workflow still running?", file=sys.stderr)
        except (TypeError, ValueError):
            pass
        return symbols
    except FileNotFoundError:
        print(f"{UNIVERSE_PATH} not found — running the pinned tickers.json list only", file=sys.stderr)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"{UNIVERSE_PATH} unreadable ({e}) — running the pinned tickers.json list only", file=sys.stderr)
    return []


def load_tickers():
    """
    Builds the run list: the pinned watchlist from tickers.json (repo root — edit
    by hand on GitHub, or via the "Manage Tickers" panel in the app) FOLLOWED BY
    the dynamic names from universe.json that aren't already pinned.

    Order matters: build_trade_for_ticker() picks each ticker's strategy from its
    index, so pinned tickers keep their positions (and therefore their existing
    strategy assignments) and dynamic names get a stable, symbol-derived index
    instead of a positional one — otherwise every day's ranking shuffle would
    flip strategies on names that didn't change.

    Falls back to a small built-in default set if tickers.json is missing or
    invalid, so a bad edit here can't break the run entirely.
    """
    default_tickers = ["AAPL", "MSFT", "NVDA", "XOM", "JPM", "SPY", "META", "TSLA", "AMD"]
    default_etfs = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "GLD"]
    try:
        with open("tickers.json") as f:
            cfg = json.load(f)
        pinned = cfg.get("tickers") or default_tickers
        etfs = set(cfg.get("etfs") or default_etfs)
        pinned = [t.strip().upper() for t in pinned if t.strip()]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"tickers.json missing or invalid ({e}) — using built-in defaults", file=sys.stderr)
        pinned, etfs = default_tickers, set(default_etfs)

    strategy_index = {}
    for i, sym in enumerate(pinned):
        strategy_index.setdefault(sym, i)

    merged = list(dict.fromkeys(pinned))
    for sym in load_universe():
        if sym not in strategy_index:
            merged.append(sym)
            strategy_index[sym] = zlib.crc32(sym.encode()) % 1200  # 1200 = multiple of 3, 4 and 2
    print(f"Run list: {len(pinned)} pinned + {len(merged) - len(set(pinned))} from universe = {len(merged)} tickers")
    return merged, etfs, strategy_index


TICKERS, KNOWN_ETFS, STRATEGY_INDEX = load_tickers()

TARGET_DTE_MIN = 7          # was 5, then 14 originally — narrowed to 7-45 so the
                             # picker searches a real window and optimizes within it,
                             # rather than drifting into 0-6 DTE territory this nightly
                             # EOD tool shouldn't be making picks for anyway.
TARGET_DTE_MAX = 45
TARGET_SHORT_DELTA = 0.20   # informal "20-delta" premium-selling target
TARGET_LONG_DELTA = 0.45    # long call/put target delta — near-the-money; balances
                             # cost against probability of profit, instead of either
                             # a cheap far-OTM "lottery ticket" or an expensive
                             # deep-ITM stock-replacement play

# --- Double Diagonal (added 2026-09-22) -------------------------------------
# The one strategy in this file that spans TWO expirations, so it can't go
# through try_strategy_pick/evaluate_expiration_candidate's single-chain
# signature — see build_double_diagonal() below for its own dedicated
# expiration-selection + chain-fetch logic instead.
DIAGONAL_NEAR_DTE_MIN = 7    # short legs: same near-dated window the rest of the
DIAGONAL_NEAR_DTE_MAX = 21   # screen's short strikes already live in, just capped
                             # tighter at the top end (21d) so the near legs are
                             # genuinely the faster-decaying side of the trade.
DIAGONAL_FAR_DTE_MIN = 35    # long legs: comfortably past the near window so
DIAGONAL_FAR_DTE_MAX = 60    # there's real calendar spread to the structure —
                             # same range HEDGE_TARGET_DTE_MIN/MAX already uses
                             # for the same "slower-decaying, longer horizon" reason.
DIAGONAL_MIN_GAP_DAYS = 14   # reject a near/far pairing that ends up too close
                             # together to behave like a real diagonal.
DIAGONAL_LONG_DELTA = 0.12   # further OTM than the near legs' 0.20-delta shorts —
                             # standard "double diagonal" shape: the bought wings
                             # are wider than the sold ones, not the same strikes
                             # (same strikes would make it a double CALENDAR, a
                             # different, not-yet-supported structure).
MAX_DIAGONAL_NEAR_CANDIDATES = 3   # how many near-expiration candidates to try
MAX_DIAGONAL_FAR_CANDIDATES = 3    # ...times how many far-expiration candidates,
                                    # per near candidate, before giving up on the
                                    # ticker — bounded so a name with unusually many
                                    # listed expirations doesn't blow up chain fetches.
RISK_FREE_RATE = 0.045      # flat approximation; update periodically
OUTPUT_PATH = "data.json"

# --- Portfolio hedge candidate ---------------------------------------------
# Every current strategy in this file sells premium (short vol) — a portfolio
# built entirely from them is exposed to the same bad day: a sudden move that
# spikes IV and moves the underlying against several short strikes at once.
# The common, simple offset is a small, cheap, far-OTM index put bought as
# tail-risk insurance — it's not trying to make money, it's there to pay off
# specifically when everything else is hurting at the same time. QQQ (not a
# single name) so this doesn't overlap with any one ticker's own short strikes.
HEDGE_TICKER = "QQQ"
HEDGE_TARGET_DTE_MIN = 30    # longer-dated than the 7-45d strategy window on
HEDGE_TARGET_DTE_MAX = 60    # purpose — a hedge you're re-checking every EOD run
                             # doesn't need to be rolled as often as a premium-
                             # selling trade does, and a 30-60d put decays slower.
HEDGE_TARGET_DELTA = 0.15    # further OTM than the 0.20-delta short-strike
                             # convention above — cheaper per contract, which
                             # matters since this is meant to cost a small,
                             # known amount, not be a large directional bet.
MAX_PLAUSIBLE_ROC = 35      # raw period ROC (%) sanity ceiling — deliberately NOT applied to
MAX_PLAUSIBLE_DEBIT_SPREAD_ROC = 500   # ceiling at/above a ~21-day debit spread; see
                                        # debit_spread_roc_ceiling() below for how this
                                        # scales down at shorter DTE — a cheap, far-OTM
                                        # debit spread can legitimately return several
                                        # hundred percent if the stock gets there,
                                        # unlike a credit strategy's premium/collateral
                                        # ratio, which the tighter MAX_PLAUSIBLE_ROC
                                        # above is actually calibrated for. Still finite
                                        # so a near-zero premium from a broken/wide
                                        # quote can't silently win the expiration-
                                        # candidate ranking on a bogus number.
DEBIT_SPREAD_ROC_CEILING_FLOOR = 80    # scaled ceiling never drops below this even at
                                        # the shortest DTE this reaches (see
                                        # debit_spread_roc_ceiling()) — a legitimately
                                        # cheap, deep-OTM short-dated spread can still
                                        # post a real (if unusual) triple-digit return;
                                        # the floor keeps the scaling from rejecting
                                        # everything at the 7-day edge of the window.
MIN_DEBIT_SPREAD_PREMIUM_PCT_OF_WIDTH = 0.30  # separate from debit_spread_roc_ceiling
                                        # above (added 2026-09-23) — that ceiling exists
                                        # to catch thin/wide-market DATA QUALITY problems
                                        # at short DTE and is deliberately loose (up to
                                        # 500%) at normal DTE, so it never caught a
                                        # perfectly legitimate but lottery-ticket-shaped
                                        # spread: long leg near the money, short leg far
                                        # enough out that the premium paid is a small
                                        # sliver of the width. That's not bad data, just
                                        # a structural shape this screen shouldn't
                                        # surface by default. Requiring the premium be at
                                        # least 30% of the width caps "Return" at roughly
                                        # 233% (width/premium - 1) regardless of DTE —
                                        # a moneyness/quality bar, not a data-quality one,
                                        # so it applies at every DTE, not just short ones.
                             # the annualized figure, since annualizing amplifies short-DTE
                             # trades by up to 365/DTE (60x+ at 6 DTE), which used to make
                             # legitimate short-dated premium look "implausible" and get
                             # rejected. Raw ROC (premium/collateral, pre-annualization) means
                             # the same thing regardless of DTE, so it's the correct thing to
                             # sanity-check for thin/wide-market quotes.


# --- Math helpers --------------------------------------------------------------

def bs_delta(spot, strike, dte_days, iv, option_type, r=RISK_FREE_RATE):
    """Black-Scholes delta. option_type: 'call' or 'put'. Assumes 0% dividend yield."""
    if dte_days <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    t = dte_days / 365.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * t) / (iv * math.sqrt(t))
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


def safe_float(v, default=0.0):
    """float() that treats NaN (and bad input) as `default` instead of propagating NaN.
    Needed because Python's `x or default` idiom does NOT catch NaN — NaN is truthy —
    and NaN silently passes any `<= 0` / `> 0` comparison (all NaN comparisons are False).
    Both of those gaps let real Yahoo data (which frequently has NaN IV/price fields
    on illiquid strikes) sail past guards that looked like they should have caught it.
    """
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def probability_of_touch(delta):
    """Rough trader heuristic, not a rigorous barrier-option calculation."""
    if delta is None or math.isnan(delta):
        return 50  # neutral fallback rather than crashing the whole ticker
    return min(100, round(abs(delta) * 2 * 100))


def probability_of_profit(delta):
    """
    Rough proxy for probability of finishing beyond a point AT expiration — used
    for long option breakeven, where a HIGHER number is favorable (the opposite
    reading from probability_of_touch's short-strike-touch risk metric above;
    see the "potIsProfitProb" flag on Long Call/Long Put trades, which tells the
    frontend which meaning applies).

    Deliberately NOT doubled like probability_of_touch: that ×2 approximates
    probability of touching a barrier ANY TIME before expiration, a different
    (larger) quantity than probability of finishing beyond a point AT
    expiration — delta itself is the standard rough proxy for the latter.
    """
    if delta is None or math.isnan(delta):
        return 50  # neutral fallback rather than crashing the whole ticker
    return min(100, round(abs(delta) * 100))


def compute_ema(closes, span):
    return closes.ewm(span=span, adjust=False).mean()


def _try_fast_info_price(tk):
    """
    Best-effort fresher price via yfinance's fast_info, which taps a
    different, live-quote-oriented endpoint than history()'s historical-
    chart data (see resolve_current_price() below for why that distinction
    matters). fast_info's exact attribute/key names have shifted across
    yfinance versions and requirements.txt only pins a lower bound
    (yfinance>=0.2.40), so this tries several known spellings defensively
    rather than assuming one — returns None (never raises) if none work,
    so a yfinance version this wasn't written against just falls back to
    the caller's existing history()-based price instead of crashing.
    """
    try:
        fi = tk.fast_info
    except Exception:
        return None
    for key in ("last_price", "lastPrice", "regular_market_price", "regularMarketPrice"):
        val = None
        try:
            val = fi[key]
        except Exception:
            val = getattr(fi, key, None)
        fval = safe_float(val, default=None)
        if fval is not None and fval > 0:
            return fval
    return None


def resolve_current_price(tk, history, ticker_symbol, context, now_et=None):
    """
    `history["Close"].iloc[-1]` (the price source every caller in this file
    uses) can lag the real most-recent close by a full trading day — this
    isn't theoretical, it's confirmed against real data: a run at 11:19pm ET
    on 2026-09-15 (long after that day's 4pm close, well past normal
    settlement) returned SPY at $760.88 and QQQ at $709.18 — both exactly
    2026-09-14's close, not 2026-09-15's real close ($757.39 / $704.54, per
    stockanalysis.com's published history). That's Yahoo's own historical-
    chart data lagging its live-quote data, not something a caching
    parameter here controls — a shorter period= on the same history() call
    hits the same backend pipeline and would show the same lag.

    Heuristic: if it's a weekday evening (past 5pm ET — safely past close
    and normal settlement) and the most recent daily bar isn't from today,
    treat the close as suspect and try fast_info, which isn't affected by
    the same lag, as a fresher cross-check. This is a heuristic, not an
    exhaustive fix (it won't catch every possible staleness window, e.g.
    one spanning a long holiday weekend) — it directly targets the exact
    failure mode confirmed above rather than trying to be a full market
    calendar. Returns (price, is_stale): is_stale is True when this is
    still using the (known-suspect) history() price because fast_info
    wasn't available or didn't look any better — callers can use that to
    flag the number rather than presenting it as confidently current.

    `now_et` is exposed purely so tests can pin "now" instead of depending
    on the real clock — production callers should always leave it as None.
    """
    price = float(history["Close"].iloc[-1])
    last_bar_date = history.index[-1]
    try:
        last_bar_date = last_bar_date.date()
    except AttributeError:
        pass  # already a plain date, or an unexpected index type — comparison below just won't match, which is safe (treated as not-stale)

    if now_et is None:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    looks_stale = now_et.weekday() < 5 and now_et.hour >= 17 and last_bar_date < now_et.date()
    if not looks_stale:
        return price, False

    fresher = _try_fast_info_price(tk)
    if fresher is not None and price > 0 and abs(fresher - price) / price > 0.001:
        print(f"  {ticker_symbol} ({context}): history() close ({price}) looks stale "
              f"(last bar {last_bar_date}, but it's {now_et:%H:%M} ET) — using fast_info "
              f"price ({fresher}) instead")
        return fresher, False

    print(f"  {ticker_symbol} ({context}): history() close ({price}) looks stale "
          f"(last bar {last_bar_date}, but it's {now_et:%H:%M} ET) and no better fast_info "
          f"price was available — using it anyway, flagged as stale")
    return price, True


def compute_atr(history, period=14):
    high, low, close = history["High"], history["Low"], history["Close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(period).mean().iloc[-1]


IV_RANK_MIN_HISTORY_DAYS = 200   # ~9-10 months of trading days. Below this, the
                                  # percentile below is ranked against a materially
                                  # shorter, more recent-biased sample than the
                                  # ~1-year window it's meant to cover — typically
                                  # a recent IPO (tk.history(period="1y") just
                                  # returns however much history actually exists,
                                  # it doesn't pad a young listing out to a year).
                                  # Flagged via the returned `limited_history` bool
                                  # instead of silently shown as an equally-
                                  # confident number next to every other ticker's.


def iv_rank_proxy(history, window=252, vol_window=20):
    """
    Proxy for IV rank using realized volatility percentile, since a year of
    historical *implied* volatility isn't freely available. Correlated with
    real IV rank but not equivalent to it.
    Returns (percentile, current_realized_vol_pct, limited_history) — the second
    value is used separately to compare against the option's actual IV (see
    classify_vol_regime); the third flags a too-short price history (see
    IV_RANK_MIN_HISTORY_DAYS above).
    """
    closes = history["Close"].tail(window + vol_window)
    limited_history = len(closes) < IV_RANK_MIN_HISTORY_DAYS
    log_returns = np.log(closes / closes.shift(1)).dropna()
    realized_vol = log_returns.rolling(vol_window).std() * math.sqrt(252)
    realized_vol = realized_vol.dropna()
    if len(realized_vol) < 20:
        return 50, None, True  # not enough history yet — neutral fallback, and
                                # definitionally a limited-history case too
    current = realized_vol.iloc[-1]
    percentile = (realized_vol < current).sum() / len(realized_vol) * 100
    return round(percentile), round(current * 100, 1), limited_history


def classify_vol_regime(iv_pct, realized_vol_pct):
    """
    Compares an option's implied volatility against the stock's own recent
    realized volatility. All of SignalHQ's current strategies are premium
    SELLING strategies (short put, covered call, credit spreads), which do
    better when IV is "rich" relative to what the stock has actually been
    doing — you're being paid more than the recent movement would justify.
    "Cheap" IV doesn't make a selling strategy wrong, but the edge is thinner.
    Returns (regime_label, ratio) — ratio is IV/RV, None if RV unavailable.
    """
    if realized_vol_pct is None or realized_vol_pct <= 0 or iv_pct is None or iv_pct <= 0:
        return "Unknown", None
    ratio = round(iv_pct / realized_vol_pct, 2)
    if ratio >= 1.15:
        return "Rich", ratio
    if ratio <= 0.85:
        return "Cheap", ratio
    return "Fair", ratio


MIN_SECTOR_PEERS_FOR_VALUATION = 3  # need at least this many same-sector
    # tickers in THIS run's universe before comparing one against the others
    # means anything — see compute_sector_valuations() below. Below this,
    # a ticker gets no valuation rather than a "peer" comparison against 1-2
    # names that happens to be misleading.
VALUATION_RICH_CHEAP_THRESHOLD_PCT = 15  # growth-adjusted P/E must be at
    # least this far from the sector median (either direction) to earn a
    # Cheap/Rich label instead of Fair — a small gap is noise, not a signal.


def compute_sector_valuations(equities):
    """
    Peer-relative valuation — added 2026-09-24. Requires NO additional
    yfinance calls: sector and P/E were already pulled inside
    build_equity_snapshot's existing tk.info fetch, just unused until now.

    IMPORTANT — this compares each ticker only against the OTHER tickers in
    THIS run's ~50-90 name universe that share its sector, not the whole
    market. A "Cheap" label means "cheaper than its peers currently being
    screened," not "cheap by any market-wide standard" — the frontend
    surfaces the peer count alongside the label specifically so this reads
    as what it is, not as a market-wide valuation call.

    P/E alone conflates "expensive" with "fast-growing," so this also pulls
    in pegRatio when available (P/E ÷ expected earnings growth) to soften
    the raw P/E signal for names where the multiple is arguably justified —
    a rough, deliberately mild adjustment, not a full growth-adjusted model.

    Returns {ticker_symbol: {valuationLabel, peVsSectorPct, sectorMedianPE,
    sectorPeerCount, sector}} for every ticker with a usable P/E, a reported
    sector, and enough same-sector peers in this run (see
    MIN_SECTOR_PEERS_FOR_VALUATION). Tickers that don't qualify are simply
    absent from the returned dict — callers treat a missing entry as "no
    valuation available," same as any other optional field in this file.
    """
    by_sector = {}
    for e in equities:
        sector = e.get("sector")
        pe = e.get("peRatio")
        if sector and isinstance(pe, (int, float)) and pe > 0:
            by_sector.setdefault(sector, []).append(e)

    out = {}
    for sector, members in by_sector.items():
        if len(members) < MIN_SECTOR_PEERS_FOR_VALUATION:
            continue
        pes = sorted(m["peRatio"] for m in members)
        mid = len(pes) // 2
        sector_median_pe = pes[mid] if len(pes) % 2 else (pes[mid - 1] + pes[mid]) / 2

        for m in members:
            pe_vs_sector_pct = round((m["peRatio"] - sector_median_pe) / sector_median_pe * 100, 1)
            growth_adjusted_pct = pe_vs_sector_pct
            peg = m.get("pegRatio")
            if isinstance(peg, (int, float)) and peg > 0:
                # A rich-looking P/E backed by strong expected growth (low
                # PEG) shouldn't be flagged "Rich" the same as one that
                # isn't — soften, don't cancel out, the raw P/E signal.
                if peg < 1.5:
                    growth_adjusted_pct -= 15
                elif peg < 2.0:
                    growth_adjusted_pct -= 7

            if growth_adjusted_pct <= -VALUATION_RICH_CHEAP_THRESHOLD_PCT:
                label = "Cheap"
            elif growth_adjusted_pct >= VALUATION_RICH_CHEAP_THRESHOLD_PCT:
                label = "Rich"
            else:
                label = "Fair"

            out[m["sym"]] = {
                "valuationLabel": label,
                "peVsSectorPct": pe_vs_sector_pct,
                "sectorMedianPE": round(sector_median_pe, 1),
                "sectorPeerCount": len(members),
                "sector": sector,
            }
    return out


def valuation_score_component(valuation_info, side):
    """
    Direction-aware scoring input for composite_score()/equity_composite_score()
    below — a "Cheap" stock supports a BULLISH thesis (upside room) but
    argues AGAINST a bearish one (why bet against something already priced
    below peers?), and vice versa for "Rich." A neutral-side trade (Iron
    Condor, Double Diagonal, or a plain equity view) doesn't lean either way
    on valuation direction, so it gets the same neutral component regardless
    of label. Returns 5 (neutral, a no-op on the blended score) when there's
    no valuation available for this ticker — same "don't let a missing
    optional input silently zero out the score" pattern already used for
    ivr/pot elsewhere in this file.
    """
    if not valuation_info:
        return 5
    label = valuation_info.get("valuationLabel")
    if side == "bull":
        return {"Cheap": 8, "Fair": 5, "Rich": 2}.get(label, 5)
    if side == "bear":
        return {"Cheap": 2, "Fair": 5, "Rich": 8}.get(label, 5)
    return 5  # neutral side (or a plain equity view) — valuation doesn't favor a direction


def composite_score(ann_profit, pot, ivr, valuation_component=5):
    """Illustrative 0-10 blend — adjust weights to match your priorities.
    valuation_component (0-10, direction-aware — see valuation_score_component())
    defaults to 5 (neutral/no-op) when no peer valuation is available for
    this ticker, so this stays backward-compatible with every existing caller."""
    profit_component = min(10, max(0, ann_profit / 5))      # ~50% ann. profit -> 10
    safety_component = min(10, max(0, (100 - pot) / 10))     # lower POT -> higher score
    ivr_component = min(10, max(0, ivr / 10))
    score = (0.35 * profit_component + 0.28 * safety_component
             + 0.17 * ivr_component + 0.20 * valuation_component)
    return round(min(10, max(1, score)), 1)


def composite_score_long_option(pot_profit, ivr, breakeven_move_pct, valuation_component=5):
    """
    Illustrative 0-10 blend for Long Call/Long Put — scored differently from
    composite_score() above because none of its inputs carry over cleanly:
    there's no annualized-profit figure (unlimited/floor-at-zero upside can't
    be reduced to one number without inventing a price target), `pot` here
    already means probability of PROFIT so higher is better (the opposite of
    composite_score's touch-probability reading), and cheap IV (low ivr) is
    what a BUYER wants — the inverse of composite_score's ivr_component,
    which rewards rich premium for a seller.
    """
    profit_prob_component = min(10, max(0, pot_profit / 10))
    cheap_iv_component = min(10, max(0, (100 - ivr) / 10))
    move_component = min(10, max(0, 10 - (breakeven_move_pct or 0) / 2))  # smaller
                                                                            # required
                                                                            # move -> higher
    score = (0.32 * profit_prob_component + 0.24 * cheap_iv_component
             + 0.24 * move_component + 0.20 * valuation_component)
    return round(min(10, max(1, score)), 1)


def composite_score_double_diagonal(pot_touch, ivr):
    """
    Illustrative 0-10 blend for Double Diagonal — no ap (see build_double_diagonal's
    docstring for why a clean profit figure isn't computed there), and pot here
    is touch probability of either short strike like Iron Condor's (higher =
    worse), NOT Long Call/Put's profit-probability reading (higher = better) —
    don't reuse composite_score_long_option, its pot/ivr directions are both
    inverted from what this needs. A double diagonal's real edge comes from
    near-term IV being rich RELATIVE TO far-term IV (term-structure skew), not
    absolute IV level the way a simple premium-seller's does — that comparison
    isn't computed here, so ivr is weighted lightly, as a loose "is there
    premium worth collecting in this name at all" signal rather than the main
    driver the way it is in composite_score().
    """
    safety_component = min(10, max(0, (100 - pot_touch) / 10))
    ivr_component = min(10, max(0, ivr / 10))
    score = 0.70 * safety_component + 0.30 * ivr_component
    return round(min(10, max(1, score)), 1)


def chain_diagnostics(df, spot=None):
    """
    Describes what a chain actually contained, for logging when strike-picking
    fails. The distinction matters a lot: 0 rows means Yahoo likely blocked or
    rate-limited the request (a known risk running yfinance from shared CI IP
    ranges); rows present but no valid IV means a stale/garbage snapshot; rows
    with valid IV but still no match means the target delta genuinely wasn't
    available that day, which is a data problem, not a request problem;
    strikes wildly inconsistent with spot means a stale/un-adjusted chain,
    most commonly following a real stock split the chain hasn't caught up to
    (confirmed against ServiceNow/NOW after its Dec 2025 5-for-1 split).
    """
    if df is None or len(df) == 0:
        return "chain came back with 0 rows — likely blocked/rate-limited by Yahoo, not a data-quality issue"
    ivs = df["impliedVolatility"].apply(lambda v: safe_float(v, default=float("nan")))
    valid = ivs[(ivs > 0) & (~ivs.isna())]
    if len(valid) == 0:
        return f"{len(df)} rows but none had valid IV — likely a stale/blocked Yahoo snapshot"
    if spot and spot > 0:
        strikes = df["strike"].apply(lambda v: safe_float(v, default=0.0))
        sane = strikes[(strikes / spot >= 0.25) & (strikes / spot <= 4.0)]
        if len(sane) == 0:
            return (f"{len(df)} rows with valid IV, but every strike is wildly inconsistent with "
                    f"spot ${spot:.2f} (strike range {strikes.min():.0f}-{strikes.max():.0f}) — "
                    f"likely a stale/un-adjusted chain, check for a recent stock split")
    return f"{len(df)} rows, {len(valid)} with valid IV (range {valid.min():.2f}-{valid.max():.2f})"


# --- Leg liquidity (recorded only; added 2026-09-30) --------------------------
# Strike selection and pricing are unchanged (delta-based picks, priced at the
# mid). Each trade just carries its legs' liquidity so the page's "Min OI" and
# "Max bid-ask %" filters can screen on it. FILL_SLIPPAGE_FRACTION only feeds
# the informational "est. fill" figure in the trade details.
FILL_SLIPPAGE_FRACTION = 0.25


def leg_quote(row):
    """(bid, ask, mid, width, open_interest) for one chain row; mid is 0 when
    there isn't a real two-sided market."""
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    oi = safe_float(row.get("openInterest"))
    two_sided = bid > 0 and ask > 0 and ask >= bid
    mid = (bid + ask) / 2 if two_sided else 0.0
    return bid, ask, mid, (ask - bid) if two_sided else 0.0, oi


def liquidity_summary(legs, calls, puts):
    """
    Worst-leg numbers for data.json:
      minOI           lowest open interest across the legs
      worstSpreadPct  widest leg's bid-ask as % of its mid — None when any
                      leg has no two-sided quote (the page treats that as
                      failing a Max bid-ask filter)
      midPremium      net premium at the mid (what the trade is priced at)
      fillPremium     same, giving up FILL_SLIPPAGE_FRACTION of each width
    """
    min_oi, worst, no_quote = None, 0.0, False
    mid_net, fill_net = 0.0, 0.0
    weak_leg, leg_quotes = None, []
    for leg in legs or []:
        chain = calls if leg.get("type") == "call" else puts
        match = chain[chain["strike"] == leg.get("strike")]
        if match.empty:
            no_quote = True
            continue
        bid, ask, mid, width, oi = leg_quote(match.iloc[0])
        # per-leg quote for the trade's order ticket (2026-10-05)
        leg_quotes.append({"type": leg.get("type"), "strike": leg.get("strike"), "action": leg.get("action"),
                           "bid": round(bid, 2), "ask": round(ask, 2), "oi": int(oi)})
        if min_oi is None or oi < min_oi:
            weak_leg = {"type": leg.get("type"), "strike": leg.get("strike"), "action": leg.get("action")}
        min_oi = oi if min_oi is None else min(min_oi, oi)
        if mid <= 0:
            no_quote = True
            continue
        worst = max(worst, width / mid)
        sign = 1 if leg.get("action") == "sell" else -1
        mid_net += sign * mid
        fill_net += sign * mid - FILL_SLIPPAGE_FRACTION * width
    return {
        "minOI": int(min_oi) if min_oi is not None else None,
        "worstSpreadPct": None if no_quote else round(worst * 100, 1),
        "midPremium": None if no_quote else round(abs(mid_net), 2),
        "fillPremium": None if no_quote else round(abs(fill_net), 2),
        "weakLeg": weak_leg,       # which leg the minOI belongs to
        "legQuotes": leg_quotes,   # bid / ask / OI per leg
    }


def pick_strike_by_delta(chain_df, spot, dte_days, target_delta, option_type):
    """Return the chain row whose computed delta is closest to target_delta."""
    best_row, best_diff = None, None
    for _, row in chain_df.iterrows():
        iv = safe_float(row.get("impliedVolatility"), default=0.0)
        if iv <= 0:
            continue
        strike = safe_float(row.get("strike"), default=0.0)
        if strike <= 0:
            continue
        # Sanity guard: a ~20-delta strike should always land within a fairly
        # narrow band of spot under any realistic market condition. A strike
        # wildly outside that band (e.g. 5-10x spot) means the options chain
        # itself is bad data — most commonly a real stock split that yfinance's
        # free/unofficial chain hasn't caught up to adjusting for yet, so the
        # strikes still reflect pre-split contract prices while `spot` (from
        # price history) correctly reflects the post-split price. Confirmed
        # this exact scenario against ServiceNow (NOW): a real 5-for-1 split
        # effective Dec 18, 2025 left the chain showing ~$1200 strikes against
        # a genuine ~$134 spot — an 8.96x mismatch a real 0.20-delta pick would
        # never produce. Skip rather than publish an internally-inconsistent
        # trade (strike and breakeven implying two different stock prices).
        if not (0.25 <= strike / spot <= 4.0):
            continue
        delta = bs_delta(spot, strike, dte_days, iv, option_type)
        if math.isnan(delta):
            continue
        diff = abs(abs(delta) - target_delta)
        if best_diff is None or diff < best_diff:
            best_diff, best_row = diff, (row, delta)
    return best_row  # (row, delta) or None


def mid_price(row):
    bid, ask = safe_float(row.get("bid")), safe_float(row.get("ask"))
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return safe_float(row.get("lastPrice"))


# --- Per-ticker trade construction --------------------------------------------

def rank_expirations(expirations, today):
    """
    Returns (in_window, outside_window) — each a list of (exp_str, dte) tuples,
    closest-to-target-window first. Returned as two SEPARATE lists rather than
    one pre-concatenated one (changed 2026-09-23): outside_window exists so a
    ticker with fewer than MAX_CANDIDATES_TO_EVALUATE in-window expirations
    (common for names without a full weekly cycle) still has somewhere to fall
    back to if EVERY in-window candidate turns out unusable (e.g. every
    relevant strike has NaN IV that day) — a true last resort, not a routine
    top-up. A single concatenated list couldn't tell those apart: the caller
    would slice the first MAX_CANDIDATES_TO_EVALUATE regardless of how many
    were in-window, silently mixing in a sub-7-day candidate whenever a
    ticker's in-window count ran short — and since annualizing (ap = roc *
    365/dte) inflates short-DTE returns by 100x+, that candidate would then
    win the ranking almost automatically over every legitimate 7-45d
    alternative, not because it was a better trade but because it was
    shorter-dated. Keeping the buckets separate lets the caller exhaust
    in_window on its own merits first.
    """
    target_mid = (TARGET_DTE_MIN + TARGET_DTE_MAX) / 2
    listed = set(expirations)
    in_window, off_cycle, outside_window = [], [], []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte <= 0:
            continue
        diff = abs(dte - target_mid)
        if TARGET_DTE_MIN <= dte <= TARGET_DTE_MAX:
            # Friday-cycle preference (2026-10-05): mega-caps now list Monday
            # and Wednesday expirations with a fraction of the open interest
            # of the Friday weeklies/monthlies — and because candidates are
            # ranked by annualized profit, the shortest-dated (often a Mon/Wed)
            # kept winning, giving trades with single-digit OI on a leg.
            # Fridays (or a Thursday in a holiday week whose Friday isn't
            # listed) are the real candidates; Mon/Wed dates are only a
            # fallback, tried before going outside the 7-45d window.
            if is_friday_cycle(exp_date, listed):
                in_window.append((diff, exp_str, dte))
            else:
                off_cycle.append((diff, exp_str, dte))
        else:
            outside_window.append((diff, exp_str, dte))
    if not in_window:
        in_window, off_cycle = off_cycle, []   # no Friday-cycle date in window: use what's there
    in_window.sort(key=lambda x: x[0])
    off_cycle.sort(key=lambda x: x[0])
    outside_window.sort(key=lambda x: x[0])
    return ([(exp_str, dte) for _, exp_str, dte in in_window],
            [(exp_str, dte) for _, exp_str, dte in off_cycle + outside_window])


def is_friday_cycle(exp_date, listed):
    """Friday expiration, or a Thursday whose following Friday isn't listed
    (exchange holiday, e.g. Good Friday) — the standard weekly/monthly cycle."""
    if exp_date.weekday() == 4:
        return True
    if exp_date.weekday() == 3:
        from datetime import timedelta
        return (exp_date + timedelta(days=1)).strftime("%Y-%m-%d") not in listed
    return False


# --- Tradeability + momentum rules (2026-10-10) -------------------------------
# Options-liquidity gate, applied to every expiration candidate. A candidate
# that fails is skipped like any other unusable expiration, and a ticker with
# no passing candidate gets no trade at all — better no idea than one you
# can't fill.
MIN_EXPIRY_TOTAL_OI = 5000     # calls + puts open interest across the whole expiration
MIN_LEG_OI = 100               # every leg of the trade needs at least this much OI
MAX_LEG_SPREAD_PCT = 25        # widest leg bid-ask as % of mid — checked only while
                               # the market is open, since Yahoo's after-hours quotes
                               # go stale/one-sided and would reject everything
# Momentum regime (replaces the old index-based strategy rotation):
#   bull    = price above its 50-day EMA, 8-day EMA above 20-day, and up over ~3 months
#   bear    = the mirror image
#   neutral = anything mixed — the only case that gets an Iron Condor
MOMENTUM_LOOKBACK_DAYS = 63    # ~3 trading months
# Credit vs debit: credit spreads (sell premium) by default; switch to the
# direction's debit spread only when options are clearly cheap, i.e. implied
# vol is well under the stock's own realized vol.
CHEAP_OPTIONS_IV_RV = 0.90
# Leveraged/inverse ETFs move too fast for a range bet: never Iron Condor them.
LEVERAGED_ETFS = {"SOXL", "SOXS", "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TNA", "TZA",
                  "LABU", "LABD", "FAS", "FAZ", "NVDL", "TSLL", "UVXY", "SVXY", "TECL"}


def market_is_open(now=None):
    """True between 9:35 and 16:00 ET on a weekday (holidays not modeled —
    on a holiday the quotes are stale, which only makes the spread check skip)."""
    try:
        from zoneinfo import ZoneInfo
        now = now or datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return False
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 35 <= minutes <= 16 * 60


def momentum_regime(history, spot):
    """('bull' | 'bear' | 'neutral', details) from the 8/20/50 EMAs plus the
    ~3-month return, so a short bounce inside a downtrend (8 > 20 but price
    still under the 50 and down on the quarter) no longer reads as bullish."""
    closes = history["Close"].dropna()
    ema8 = compute_ema(closes, 8).iloc[-1]
    ema20 = compute_ema(closes, 20).iloc[-1]
    ema50 = compute_ema(closes, 50).iloc[-1]
    ret3m = None
    if len(closes) > MOMENTUM_LOOKBACK_DAYS:
        base = float(closes.iloc[-MOMENTUM_LOOKBACK_DAYS - 1])
        if base > 0:
            ret3m = (spot / base - 1) * 100
    details = {"ema50": round(float(ema50), 2), "ret3m": None if ret3m is None else round(ret3m, 1)}
    if ret3m is None:
        return "neutral", details
    if spot > ema50 and ema8 > ema20 and ret3m > 0:
        return "bull", details
    if spot < ema50 and ema8 < ema20 and ret3m < 0:
        return "bear", details
    return "neutral", details


MAX_CANDIDATES_TO_EVALUATE = 8  # how many expirations within the target window to
                                 # actually price and compare before picking whichever
                                 # produces the best annualized profit. Bounded so a
                                 # name with unusually many listed expirations doesn't
                                 # blow up the number of chain fetches per ticker.


def debit_spread_roc_ceiling(dte):
    """
    Scaled version of MAX_PLAUSIBLE_DEBIT_SPREAD_ROC (added 2026-09-23, after
    a near-miss: a real 2-DTE Bear Put Spread posted a 488% raw ROC, just
    under the flat 500% ceiling, and — because outside-window fallback used
    to mix short-DTE candidates into the same evaluated pool as legitimate
    7-45d ones — very nearly won a ranking it had no business winning once
    annualized). The flat 500% ceiling was calibrated with ~21-45 day debit
    spreads in mind, where a rich max-profit/premium ratio is normal for a
    cheap, far-OTM spread. At very short DTE, thin extrinsic value across the
    ENTIRE chain (not just the strikes picked) can produce that same raw
    ratio from a fundamentally different, less legitimate cause — a thin or
    wide-market quote, not a genuinely cheap spread. Scales linearly down to
    DEBIT_SPREAD_ROC_CEILING_FLOOR at TARGET_DTE_MIN (the 7-day edge of the
    normal target window — see rank_expirations for why anything shorter
    than that should now only ever appear as a last-resort fallback anyway),
    full value at 21+ days.
    """
    if dte >= 21:
        return MAX_PLAUSIBLE_DEBIT_SPREAD_ROC
    span = max(1, 21 - TARGET_DTE_MIN)
    frac = max(0.0, dte - TARGET_DTE_MIN) / span
    return DEBIT_SPREAD_ROC_CEILING_FLOOR + frac * (MAX_PLAUSIBLE_DEBIT_SPREAD_ROC - DEBIT_SPREAD_ROC_CEILING_FLOOR)


def evaluate_expiration_candidate(tk, strat, side, spot, cand_exp, cand_dte, atr):
    """
    Builds a complete, fully-priced trade for ONE specific expiration, so multiple
    expirations can be compared against each other on actual economics (annualized
    profit) rather than just picking whichever is closest to a fixed target date.
    Returns (result_dict, None) on success or (None, reason_string) on failure —
    result_dict covers everything that varies per-expiration; the caller fills in
    the per-ticker fields (news, earnings, composite score) that don't depend on
    which expiration ends up winning.
    """
    try:
        chain = tk.option_chain(cand_exp)
    except Exception as e:
        return None, f"chain fetch failed: {e}"
    calls, puts = chain.calls, chain.puts

    fields, reason = try_strategy_pick(strat, calls, puts, spot, cand_dte)
    if not fields:
        return None, reason
    liquidity = liquidity_summary(fields.get("legs"), calls, puts)

    # Tradeability gate (2026-10-10) — see MIN_EXPIRY_TOTAL_OI and friends.
    expiry_oi = float(calls["openInterest"].fillna(0).sum() + puts["openInterest"].fillna(0).sum())
    if expiry_oi < MIN_EXPIRY_TOTAL_OI:
        return None, f"thin expiration (total OI {int(expiry_oi):,} < {MIN_EXPIRY_TOTAL_OI:,})"
    if liquidity.get("minOI") is None or liquidity["minOI"] < MIN_LEG_OI:
        return None, f"thin leg (OI {liquidity.get('minOI')} < {MIN_LEG_OI})"
    if market_is_open():
        spr = liquidity.get("worstSpreadPct")
        if spr is None or spr > MAX_LEG_SPREAD_PCT:
            return None, f"wide market (worst leg spread {spr}% > {MAX_LEG_SPREAD_PCT}%)"
    liquidity["expiryOI"] = int(expiry_oi)

    premium = fields["premium"]
    collateral = fields["collateral"]
    if premium <= 0 or collateral <= 0:
        return None, "unusable premium/collateral"

    # Long Call/Long Put have uncapped (call) or floor-at-zero (put) upside — a
    # "% return on capital" figure is either undefined or requires inventing a
    # made-up price target, so these two skip roc/ap entirely. breakevenMovePct
    # (below) is the honest substitute: the % move actually required to profit.
    is_uncapped_debit = strat in ("Long Call", "Long Put")
    # Bull Call Spread/Bear Put Spread ARE defined-risk (capped profit, capped
    # loss) despite being debit strategies, so unlike Long Call/Long Put they
    # DO get a real roc/ap — just measured against the net debit paid instead
    # of a credit collected (see try_strategy_pick's "max_profit" field).
    is_debit_spread = strat in ("Bull Call Spread", "Bear Put Spread")
    if is_uncapped_debit:
        if premium >= spot:
            return None, f"premium (${premium:.2f}) implausibly >= spot (${spot:.2f}), likely a thin/wide-market quote"
        roc = None
        ann_profit = None
    elif is_debit_spread:
        roc = round((fields["max_profit"] / premium) * 100, 2)
        ann_profit = round(roc * (365 / cand_dte), 1)
        if roc > debit_spread_roc_ceiling(cand_dte):
            return None, f"implausible raw ROC ({roc}%), likely a thin/wide-market quote"
        _width = premium + fields["max_profit"]
        _premium_pct_of_width = premium / _width if _width > 0 else 0
        if _premium_pct_of_width < MIN_DEBIT_SPREAD_PREMIUM_PCT_OF_WIDTH:
            return None, (f"premium (${premium:.2f}) is only {_premium_pct_of_width*100:.0f}% of the "
                           f"${_width:.2f} spread width — too cheap/wide relative to width for a "
                           f"moderate return profile (min {MIN_DEBIT_SPREAD_PREMIUM_PCT_OF_WIDTH*100:.0f}%)")
    else:
        roc = round((premium / collateral) * 100, 2)
        ann_profit = round(roc * (365 / cand_dte), 1)

        # sanity guard: check the RAW period ROC, not the annualized figure — annualizing
        # amplifies short-DTE trades by 365/dte (60x+ at 6 DTE), so a flat cap on the
        # annualized number would reject perfectly legitimate short-dated premium just
        # for being short-dated. Raw ROC means the same thing regardless of DTE.
        if roc > MAX_PLAUSIBLE_ROC:
            return None, f"implausible raw ROC ({roc}%), likely a thin/wide-market quote"

    total_call_oi = calls["openInterest"].fillna(0).sum()
    total_put_oi = puts["openInterest"].fillna(0).sum()
    total_call_vol = calls["volume"].fillna(0).sum()
    total_put_vol = puts["volume"].fillna(0).sum()
    pc_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi else 0
    pc_vol = round(total_put_vol / total_call_vol, 2) if total_call_vol else 0

    iv = fields["iv"]
    # $/day only means "income per day" for a credit strategy — for a debit
    # strategy premium is money PAID, so this is left unset rather than
    # published under a label that implies the opposite of what it means.
    daily_return = None if (is_uncapped_debit or is_debit_spread) else round(premium * 100 / cand_dte, 2)

    if strat == "Iron Condor":
        # Two strikes at risk instead of one — use the WORSE (higher) of the
        # two individual touch probabilities as the overall risk reading,
        # since the position is only as safe as its more-threatened wing.
        pot_put = probability_of_touch(bs_delta(spot, fields["put_short_strike"], cand_dte, iv / 100 if iv else 0.3, "put"))
        pot_call = probability_of_touch(bs_delta(spot, fields["call_short_strike"], cand_dte, iv / 100 if iv else 0.3, "call"))
        pot = max(pot_put, pot_call)
        margin_of_safety = bool(atr and min(
            abs(spot - fields["put_short_strike"]), abs(spot - fields["call_short_strike"])
        ) >= atr)
        breakeven_out = fields["breakeven_low"]  # single-field fallback; breakevenLow/High carry the real range below
        breakeven_low_out = fields["breakeven_low"]
        breakeven_high_out = fields["breakeven_high"]
        breakeven_move_pct_out = None
        pot_is_profit_prob = False
    elif is_uncapped_debit:
        option_type = "call" if strat == "Long Call" else "put"
        breakeven_delta = bs_delta(spot, fields["breakeven"], cand_dte, iv / 100 if iv else 0.3, option_type)
        pot = probability_of_profit(breakeven_delta)
        margin_of_safety = bool(atr and abs(spot - fields["strike_for_pot"]) >= atr)
        breakeven_out = round(fields["breakeven"], 2)
        breakeven_low_out = None
        breakeven_high_out = None
        move = (fields["breakeven"] - spot) if strat == "Long Call" else (spot - fields["breakeven"])
        breakeven_move_pct_out = round(move / spot * 100, 2)
        pot_is_profit_prob = True
    elif is_debit_spread:
        # Same probability-of-profit math as Long Call/Long Put (breakeven
        # delta), since these are still bought positions that need the stock
        # to clear a breakeven — just with a capped profit above it instead
        # of running unbounded, so roc/ap are populated above where Long
        # Call/Long Put's aren't. breakevenMovePct is skipped here (unlike
        # the uncapped branch) since roc/ap already give an honest number.
        option_type = "call" if strat == "Bull Call Spread" else "put"
        breakeven_delta = bs_delta(spot, fields["breakeven"], cand_dte, iv / 100 if iv else 0.3, option_type)
        pot = probability_of_profit(breakeven_delta)
        margin_of_safety = bool(atr and abs(spot - fields["strike_for_pot"]) >= atr)
        breakeven_out = round(fields["breakeven"], 2)
        breakeven_low_out = None
        breakeven_high_out = None
        breakeven_move_pct_out = None
        pot_is_profit_prob = True
    else:
        pot = probability_of_touch(bs_delta(spot, fields["strike_for_pot"], cand_dte, iv / 100 if iv else 0.3, "put" if side == "bull" else "call"))
        margin_of_safety = bool(atr and abs(spot - fields["strike_for_pot"]) >= atr)
        breakeven_out = round(fields["breakeven"], 2)
        breakeven_low_out = None
        breakeven_high_out = None
        breakeven_move_pct_out = None
        pot_is_profit_prob = False

    exp_label = datetime.strptime(cand_exp, "%Y-%m-%d").strftime("%b %-d") if sys.platform != "win32" else datetime.strptime(cand_exp, "%Y-%m-%d").strftime("%b %d").replace(" 0", " ")

    return {
        "exp": exp_label,
        "dte": cand_dte,
        "pot": pot,
        "ap": ann_profit,
        "dailyReturn": daily_return,
        "roc": roc,
        "marginOfSafety": margin_of_safety,
        "delta": round(fields["delta_for_output"], 2),
        "iv": round(iv, 1),
        "premium": round(premium, 2),
        "breakeven": breakeven_out,
        "breakevenLow": breakeven_low_out,
        "breakevenHigh": breakeven_high_out,
        "breakevenMovePct": breakeven_move_pct_out,
        "maxLoss": fields["max_loss"],
        "pcOI": pc_oi,
        "pcVol": pc_vol,
        "strike": fields["strike_label"],
        "hedge": fields.get("hedge"),
        "debitStrategy": strat in ("Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread"),
        "potIsProfitProb": pot_is_profit_prob,
        # Raw ISO expiration + per-leg strike/type/action data — not used by
        # the recommendation display at all, only carried through so a
        # tracked position (see update_tracked_positions()) can later
        # re-identify and re-quote this exact contract combination.
        "expDate": cand_exp,
        "legs": fields.get("legs"),
        "liquidity": liquidity,
    }, None


def _rank_diagonal_expirations(expirations, today, dte_min, dte_max):
    """Same closest-to-target-window ranking as rank_expirations(), parameterized
    for the near/far windows a double diagonal needs instead of the single
    7-45d window every other strategy shares."""
    target_mid = (dte_min + dte_max) / 2
    in_window, outside_window = [], []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte <= 0:
            continue
        diff = abs(dte - target_mid)
        (in_window if dte_min <= dte <= dte_max else outside_window).append((diff, exp_str, dte))
    in_window.sort(key=lambda x: x[0])
    outside_window.sort(key=lambda x: x[0])
    return [(exp_str, dte) for _, exp_str, dte in (in_window + outside_window)]


def build_double_diagonal(tk, spot, expirations, today, atr):
    """
    The only strategy here that spans two expirations — sell a near-dated
    20-delta strangle, buy a further-OTM (12-delta) far-dated strangle as its
    defined-risk wings. Bypasses evaluate_expiration_candidate/try_strategy_pick
    entirely (both assume one chain/one expiration) with its own near+far
    candidate search instead. Tries the first near/far pairing that prices
    successfully rather than exhaustively ranking every combination — keeps
    the chain-fetch count bounded the same way MAX_CANDIDATES_TO_EVALUATE does
    for every other strategy.

    Deliberately does NOT compute roc/ap (see the "no roc/ap" note below) —
    same honest omission Long Call/Long Put already make for a different
    reason. Returns a dict shaped like evaluate_expiration_candidate's,
    plus backExp/backExpDate (the far leg's expiration — see the existing
    "exp": "back" convention _reprice_position()/the frontend already handle
    for tracked positions), or (None, reason) on failure.
    """
    near_candidates = _rank_diagonal_expirations(expirations, today, DIAGONAL_NEAR_DTE_MIN, DIAGONAL_NEAR_DTE_MAX)
    far_candidates = _rank_diagonal_expirations(expirations, today, DIAGONAL_FAR_DTE_MIN, DIAGONAL_FAR_DTE_MAX)
    if not near_candidates or not far_candidates:
        return None, "no usable near+far expiration pairing (chain doesn't list enough distinct expirations)"

    chain_cache = {}
    def get_chain(exp_str):
        if exp_str not in chain_cache:
            try:
                chain_cache[exp_str] = tk.option_chain(exp_str)
            except Exception as e:
                chain_cache[exp_str] = None
                print(f"    double diagonal chain fetch failed for {exp_str}: {e}")
        return chain_cache[exp_str]

    failure_reasons = []
    for near_exp, near_dte in near_candidates[:MAX_DIAGONAL_NEAR_CANDIDATES]:
        near_chain = get_chain(near_exp)
        if near_chain is None or near_chain.calls.empty or near_chain.puts.empty:
            failure_reasons.append(f"{near_exp} (near, {near_dte}d): chain unusable")
            continue
        near_calls, near_puts = near_chain.calls, near_chain.puts

        near_short_call = pick_strike_by_delta(near_calls, spot, near_dte, TARGET_SHORT_DELTA, "call")
        near_short_put = pick_strike_by_delta(near_puts, spot, near_dte, TARGET_SHORT_DELTA, "put")
        if not near_short_call or not near_short_put:
            failure_reasons.append(f"{near_exp} (near, {near_dte}d): no strike near target delta on one side — {chain_diagnostics(near_calls, spot)}")
            continue
        nc_row, nc_delta = near_short_call
        np_row, np_delta = near_short_put
        near_call_strike = float(nc_row["strike"])
        near_put_strike = float(np_row["strike"])
        near_credit = mid_price(nc_row) + mid_price(np_row)

        tried_far_this_near = 0
        for far_exp, far_dte in far_candidates[:MAX_DIAGONAL_FAR_CANDIDATES]:
            if far_dte - near_dte < DIAGONAL_MIN_GAP_DAYS:
                continue
            tried_far_this_near += 1
            far_chain = get_chain(far_exp)
            if far_chain is None or far_chain.calls.empty or far_chain.puts.empty:
                failure_reasons.append(f"{far_exp} (far, {far_dte}d): chain unusable")
                continue
            far_calls, far_puts = far_chain.calls, far_chain.puts

            far_long_call = pick_strike_by_delta(far_calls, spot, far_dte, DIAGONAL_LONG_DELTA, "call")
            far_long_put = pick_strike_by_delta(far_puts, spot, far_dte, DIAGONAL_LONG_DELTA, "put")
            if not far_long_call or not far_long_put:
                failure_reasons.append(f"{far_exp} (far, {far_dte}d): no strike near target delta on one side — {chain_diagnostics(far_calls, spot)}")
                continue
            fc_row, _ = far_long_call
            fp_row, _ = far_long_put
            far_call_strike = float(fc_row["strike"])
            far_put_strike = float(fp_row["strike"])

            # The defined-risk shape requires the bought wings to sit OUTSIDE
            # the sold ones — normally guaranteed by targeting a lower delta
            # on the far legs, but a thin/unusual chain can still invert this.
            if not (far_call_strike >= near_call_strike and far_put_strike <= near_put_strike):
                failure_reasons.append(f"{near_exp}/{far_exp}: far strikes aren't wider than near strikes, skipping")
                continue

            far_debit = mid_price(fc_row) + mid_price(fp_row)
            net_debit = far_debit - near_credit
            if net_debit <= 0:
                failure_reasons.append(f"{near_exp}/{far_exp}: net debit isn't positive (${net_debit:.2f}), unusable quote")
                continue

            avg_iv = (safe_float(nc_row.get("impliedVolatility")) + safe_float(np_row.get("impliedVolatility"))) / 2 * 100

            # No roc/ap: a double diagonal's max profit is reached somewhere
            # between the short strikes at NEAR expiration and depends on the
            # far legs' remaining time value there — not closed-form the way
            # a vertical spread's width-minus-debit is. Rather than publish a
            # number built on an unstated pricing-model assumption, this
            # follows Long Call/Long Put's precedent: no roc/ap, pot (below)
            # carries the quality signal instead. maxLoss uses net debit paid
            # as a conservative upper bound — actual worst case is usually
            # somewhat less, since the far legs retain some value even then.
            pot_put = probability_of_touch(bs_delta(spot, near_put_strike, near_dte, avg_iv / 100 if avg_iv else 0.3, "put"))
            pot_call = probability_of_touch(bs_delta(spot, near_call_strike, near_dte, avg_iv / 100 if avg_iv else 0.3, "call"))
            pot = max(pot_put, pot_call)
            margin_of_safety = bool(atr and min(abs(spot - near_put_strike), abs(spot - near_call_strike)) >= atr)

            total_call_oi = near_calls["openInterest"].fillna(0).sum()
            total_put_oi = near_puts["openInterest"].fillna(0).sum()
            total_call_vol = near_calls["volume"].fillna(0).sum()
            total_put_vol = near_puts["volume"].fillna(0).sum()

            near_exp_label = (datetime.strptime(near_exp, "%Y-%m-%d").strftime("%b %-d") if sys.platform != "win32"
                               else datetime.strptime(near_exp, "%Y-%m-%d").strftime("%b %d").replace(" 0", " "))
            far_exp_label = (datetime.strptime(far_exp, "%Y-%m-%d").strftime("%b %-d") if sys.platform != "win32"
                              else datetime.strptime(far_exp, "%Y-%m-%d").strftime("%b %d").replace(" 0", " "))

            return {
                "exp": near_exp_label, "expDate": near_exp, "dte": near_dte,
                "backExp": far_exp_label, "backExpDate": far_exp,
                "pot": pot, "potIsProfitProb": False,
                "ap": None, "roc": None, "dailyReturn": None,
                "marginOfSafety": margin_of_safety,
                "delta": round(nc_delta + np_delta, 2),  # net delta -- roughly market-neutral by construction
                "iv": round(avg_iv, 1),
                "premium": round(net_debit, 2),
                "breakeven": round(near_put_strike, 2),  # single-field fallback; Low/High carry the real range
                "breakevenLow": round(near_put_strike, 2),
                "breakevenHigh": round(near_call_strike, 2),
                "breakevenMovePct": None,
                "maxLoss": round(net_debit * 100, 2),
                "pcOI": round(total_put_oi / total_call_oi, 2) if total_call_oi else 0,
                "pcVol": round(total_put_vol / total_call_vol, 2) if total_call_vol else 0,
                "strike": f"${far_put_strike:.0f}/{near_put_strike:.0f}/{near_call_strike:.0f}/{far_call_strike:.0f}",
                "hedge": None,
                "debitStrategy": True,
                "legs": [
                    {"type": "call", "strike": near_call_strike, "action": "sell", "qty": 1},
                    {"type": "put",  "strike": near_put_strike,  "action": "sell", "qty": 1},
                    {"type": "call", "strike": far_call_strike,  "action": "buy",  "qty": 1, "exp": "back"},
                    {"type": "put",  "strike": far_put_strike,   "action": "buy",  "qty": 1, "exp": "back"},
                ],
            }, None
        if tried_far_this_near == 0:
            failure_reasons.append(f"{near_exp} (near, {near_dte}d): no far expiration far enough past it")

    return None, "; ".join(failure_reasons) if failure_reasons else "no usable near/far expiration pairing"


MAX_NEWS_HEADLINES = 3       # how many recent headlines to pull and score per ticker


_RAW_NEWS_CACHE = {}   # symbol -> list of raw news items; the same ticker is fetched
                       # twice per run (options trade + equity snapshot), so this
                       # halves the news requests and the rate-limit exposure.


def _collect_raw_news(tk, ticker_symbol):
    """
    Gathers raw news items from several Yahoo endpoints, in order, and returns
    the first non-empty result as (items, source_label). Added because a single
    source (tk.news) silently returning [] made EVERY ticker show "no recent
    headlines" with no way to tell why:
      1. tk.news / tk.get_news() — newer yfinance nests items under "content"
         and generally has NO relatedTickers, so relevance falls back to a
         text match against the headline.
      2. yf.Search(symbol).news — a different endpoint whose items DO carry
         relatedTickers (ground truth for relevance), and which often still
         works when the first one is empty or blocked.
    Every attempt is logged so an empty result is diagnosable from the run log.
    """
    if ticker_symbol in _RAW_NEWS_CACHE:
        return _RAW_NEWS_CACHE[ticker_symbol]

    attempts = []
    result = ([], "none")

    def try_source(label, fn):
        try:
            items = fn() or []
            attempts.append(f"{label}={len(items)}")
            return list(items)
        except Exception as e:
            attempts.append(f"{label}=ERR({type(e).__name__}: {str(e)[:80]})")
            return []

    items = try_source("tk.news", lambda: tk.news)
    if items:
        result = (items, "tk.news")
    elif hasattr(tk, "get_news"):
        items = try_source("get_news(all)", lambda: tk.get_news(count=20, tab="all"))
        if items:
            result = (items, "get_news(all)")

    if not result[0]:
        items = try_source("Search.news", lambda: yf.Search(ticker_symbol, news_count=15, max_results=1).news)
        if items:
            result = (items, "Search.news")

    if not result[0]:
        print(f"    {ticker_symbol} news: ALL sources empty ({', '.join(attempts)}) — "
              f"Yahoo is likely blocking/rate-limiting this runner's IP")
    elif len(attempts) > 1:
        print(f"    {ticker_symbol} news: first source(s) empty, recovered via {result[1]} ({', '.join(attempts)})")

    # Don't cache total failures — a later call in the same run may succeed.
    if result[0]:
        _RAW_NEWS_CACHE[ticker_symbol] = result
    return result


def fetch_news_and_sentiment(tk, ticker_symbol):
    """
    Pulls recent headlines via yfinance's free .news property and scores them with
    a finance-augmented VADER analyzer. This is lexicon-based sentiment on
    headlines only — not full-article analysis, and not an LLM reading the story
    for context. Treat it as a rough "does the recent press read positive or
    negative" gauge, not a rigorous signal.

    IMPORTANT: yfinance's .news is NOT reliably scoped to the requested ticker —
    since March 2024 it can return general Yahoo Finance homepage/trending stories
    mixed in with genuinely ticker-specific ones (documented upstream:
    github.com/ranaroussi/yfinance/issues/1956). Trusting it blindly means a
    completely unrelated headline (e.g. a Costco story under an Amazon trade) can
    end up attached to the wrong stock.

    Yahoo's own news items carry a `relatedTickers` field naming which symbols
    an article is actually about — used as ground truth when present. But in
    practice that field is often just not populated at all, so when it's missing
    this falls back to checking whether the ticker symbol or company brand name
    appears in the headline. That fallback matches on NORMALIZED text (all
    whitespace/punctuation stripped from both the candidate name and the
    headline) rather than an exact substring — confirmed necessary in practice:
    Yahoo's info can list a company as "JP Morgan Chase & Co." (with a space),
    while headlines write it as "JPMorgan" (no space) — a literal substring
    match fails there even though the two obviously refer to the same company.

    Returns (avg_compound_score, label, headlines_list). Fails soft — a ticker
    with no news, or if Yahoo's news endpoint has a bad day, just gets neutral
    sentiment and an empty headline list rather than sinking the whole ticker.
    """
    FILLER_WORDS = {"inc", "incorporated", "corp", "corporation", "co", "ltd", "limited",
                     "plc", "llc", "holdings", "holding", "group", "company", "com", "net",
                     "org", "the", "and"}

    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    raw_news, _news_source = _collect_raw_news(tk, ticker_symbol)

    # Best-effort brand-name candidates, used only for the text-match fallback
    # below — a failure here just means the fallback relies on the ticker
    # symbol alone. Builds normalized candidates from the company's short/long
    # name: the first meaningful word alone ("JPMorgan"), and the first two
    # words concatenated ("JP"+"Morgan" -> "jpmorgan"), covering both
    # single-word and space-separated brand name styles.
    brand_candidates = []
    try:
        info = tk.info or {}
        raw_name = (info.get("shortName") or info.get("longName") or "").strip()
        words = re.findall(r"[A-Za-z0-9']+", raw_name)
        brand_words = [w for w in words if w.lower() not in FILLER_WORDS]
        if brand_words:
            brand_candidates.append(normalize(brand_words[0]))
            if len(brand_words) >= 2:
                brand_candidates.append(normalize(brand_words[0] + brand_words[1]))
        brand_candidates = [c for c in brand_candidates if len(c) >= 3]
        if raw_news and not brand_candidates:
            print(f"    {ticker_symbol} news: tk.info had no usable shortName/longName (company-name fallback unavailable)")
    except Exception as e:
        if raw_news:
            print(f"    {ticker_symbol} news: tk.info raised an exception: {e} (company-name fallback unavailable)")

    def is_relevant(title, related_upper):
        if related_upper:
            sym = ticker_symbol.upper()
            return sym in related_upper or sym.replace("-", ".") in related_upper or sym.replace(".", "-") in related_upper
        title_lower = title.lower()
        if re.search(r"\b" + re.escape(ticker_symbol.lower()) + r"\b", title_lower):
            return True
        norm_title = normalize(title)
        if any(cand in norm_title for cand in brand_candidates):
            return True
        return False

    headlines = []
    scores = []
    rejected_examples = []
    for item in raw_news:
        if len(headlines) >= MAX_NEWS_HEADLINES:
            break
        # yfinance's news item shape has shifted across versions; handle both
        # the flat dict style and the newer nested {"content": {...}} style.
        content = item.get("content", item)

        title = content.get("title") or content.get("headline")
        if not title:
            continue

        related = content.get("relatedTickers") or item.get("relatedTickers") or []
        related_upper = [str(r).upper() for r in related]
        if not is_relevant(title, related_upper):
            if len(rejected_examples) < 3:
                reason = f"relatedTickers={related_upper}" if related_upper else "no relatedTickers, no text match"
                rejected_examples.append(f'"{title[:50]}..." ({reason})')
            continue

        publisher = (content.get("provider") or {}).get("displayName") if isinstance(content.get("provider"), dict) else content.get("publisher")
        link = (content.get("canonicalUrl") or {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else content.get("link")
        compound = _sentiment_analyzer.polarity_scores(title)["compound"]
        scores.append(compound)
        headlines.append({
            "title": title,
            "publisher": publisher or "Unknown source",
            "link": link or "",
            "sentiment": round(compound, 2),
        })

    if raw_news and not headlines:
        print(f"    {ticker_symbol} news: {len(raw_news)} raw items came back but ALL were filtered out as irrelevant. Examples:")
        for ex in rejected_examples:
            print(f"      - {ex}")

    if not scores:
        return 0.0, "Neutral", []

    avg = sum(scores) / len(scores)
    if avg >= 0.15:
        label = "Positive"
    elif avg <= -0.15:
        label = "Negative"
    else:
        label = "Neutral"
    return round(avg, 2), label, headlines


def try_strategy_pick(strat, calls, puts, spot, dte):
    """
    Attempts to pick strikes for `strat` against one expiration's chain.
    Returns (fields_dict, None) on success, or (None, reason_string) on failure —
    the reason gets logged by the caller, and used to try the next expiration
    candidate rather than silently giving up on the whole ticker.
    """
    if strat == "Bull Put Spread":
        short_row = pick_strike_by_delta(puts, spot, dte, TARGET_SHORT_DELTA, "put")
        if not short_row:
            return None, f"no put near target delta for short leg — {chain_diagnostics(puts, spot)}"
        s_row, s_delta = short_row
        short_strike = float(s_row["strike"])
        lower_strikes = puts[puts["strike"] < short_strike].sort_values("strike", ascending=False)
        if lower_strikes.empty:
            return None, "no further-OTM strike available for the long leg"
        long_row = lower_strikes.iloc[min(1, len(lower_strikes) - 1)]
        long_strike = float(long_row["strike"])
        premium = mid_price(s_row) - mid_price(long_row)
        width = short_strike - long_strike
        return {
            "premium": premium, "strike_for_pot": short_strike, "collateral": width,
            "breakeven": short_strike - premium, "max_loss": round((width - premium) * 100, 2),
            "strike_label": f"${short_strike:.0f}/{long_strike:.0f}", "iv": safe_float(s_row.get("impliedVolatility")) * 100,
            "delta_for_output": s_delta,
            "legs": [
                {"type": "put", "strike": short_strike, "action": "sell", "qty": 1},
                {"type": "put", "strike": long_strike, "action": "buy", "qty": 1},
            ],
        }, None

    if strat == "Iron Condor":
        # Both sides reuse the exact same leg-selection rule as the standalone
        # Bull Put Spread / Bear Call Spread above (2nd-next further-OTM
        # strike) — an iron condor IS those two spreads, run together on the
        # same ticker and expiration, not a different construction method.
        put_short_row = pick_strike_by_delta(puts, spot, dte, TARGET_SHORT_DELTA, "put")
        if not put_short_row:
            return None, f"no put near target delta for condor's put side — {chain_diagnostics(puts, spot)}"
        ps_row, ps_delta = put_short_row
        put_short_strike = float(ps_row["strike"])
        lower_strikes = puts[puts["strike"] < put_short_strike].sort_values("strike", ascending=False)
        if lower_strikes.empty:
            return None, "no further-OTM strike available for condor's put long leg"
        put_long_row = lower_strikes.iloc[min(1, len(lower_strikes) - 1)]
        put_long_strike = float(put_long_row["strike"])
        put_premium = mid_price(ps_row) - mid_price(put_long_row)
        put_width = put_short_strike - put_long_strike

        call_short_row = pick_strike_by_delta(calls, spot, dte, TARGET_SHORT_DELTA, "call")
        if not call_short_row:
            return None, f"no call near target delta for condor's call side — {chain_diagnostics(calls, spot)}"
        cs_row, cs_delta = call_short_row
        call_short_strike = float(cs_row["strike"])
        higher_strikes = calls[calls["strike"] > call_short_strike].sort_values("strike")
        if higher_strikes.empty:
            return None, "no further-OTM strike available for condor's call long leg"
        call_long_row = higher_strikes.iloc[min(1, len(higher_strikes) - 1)]
        call_long_strike = float(call_long_row["strike"])
        call_premium = mid_price(cs_row) - mid_price(call_long_row)
        call_width = call_long_strike - call_short_strike

        total_premium = put_premium + call_premium
        # The stock can't simultaneously be above the call spread AND below
        # the put spread at expiration — only one side can ever be breached —
        # so max loss uses the WORSE single-side width, not the sum of both.
        worst_width = max(put_width, call_width)
        avg_iv = (safe_float(ps_row.get("impliedVolatility")) + safe_float(cs_row.get("impliedVolatility"))) / 2 * 100

        return {
            "premium": total_premium, "collateral": worst_width,
            "max_loss": round((worst_width - total_premium) * 100, 2),
            "strike_label": f"${put_long_strike:.0f}/{put_short_strike:.0f}/{call_short_strike:.0f}/{call_long_strike:.0f}",
            "iv": avg_iv,
            "delta_for_output": ps_delta + cs_delta,  # net delta -- roughly market-neutral by construction
            "put_short_strike": put_short_strike, "call_short_strike": call_short_strike,
            "breakeven_low": round(put_short_strike - total_premium, 2),
            "breakeven_high": round(call_short_strike + total_premium, 2),
            # Iron Condor is already defined-risk on both sides by construction —
            # nothing to hedge, same as the standalone spreads above.
            "hedge": None,
            "legs": [
                {"type": "put", "strike": put_short_strike, "action": "sell", "qty": 1},
                {"type": "put", "strike": put_long_strike, "action": "buy", "qty": 1},
                {"type": "call", "strike": call_short_strike, "action": "sell", "qty": 1},
                {"type": "call", "strike": call_long_strike, "action": "buy", "qty": 1},
            ],
        }, None

    if strat == "Bear Call Spread":
        short_row = pick_strike_by_delta(calls, spot, dte, TARGET_SHORT_DELTA, "call")
        if not short_row:
            return None, f"no call near target delta for short leg — {chain_diagnostics(calls, spot)}"
        s_row, s_delta = short_row
        short_strike = float(s_row["strike"])
        higher_strikes = calls[calls["strike"] > short_strike].sort_values("strike")
        if higher_strikes.empty:
            return None, "no further-OTM strike available for the long leg"
        long_row = higher_strikes.iloc[min(1, len(higher_strikes) - 1)]
        long_strike = float(long_row["strike"])
        premium = mid_price(s_row) - mid_price(long_row)
        width = long_strike - short_strike
        return {
            "premium": premium, "strike_for_pot": short_strike, "collateral": width,
            "breakeven": short_strike + premium, "max_loss": round((width - premium) * 100, 2),
            "strike_label": f"${short_strike:.0f}/{long_strike:.0f}", "iv": safe_float(s_row.get("impliedVolatility")) * 100,
            "delta_for_output": s_delta,
            "legs": [
                {"type": "call", "strike": short_strike, "action": "sell", "qty": 1},
                {"type": "call", "strike": long_strike, "action": "buy", "qty": 1},
            ],
        }, None

    if strat == "Long Call":
        picked_row = pick_strike_by_delta(calls, spot, dte, TARGET_LONG_DELTA, "call")
        if not picked_row:
            return None, f"no call near target delta — {chain_diagnostics(calls, spot)}"
        row, delta = picked_row
        premium = mid_price(row)
        strike = float(row["strike"])
        return {
            "premium": premium, "strike_for_pot": strike, "collateral": premium,
            "breakeven": strike + premium, "max_loss": round(premium * 100, 2),
            "strike_label": f"${strike:.0f} C", "iv": safe_float(row.get("impliedVolatility")) * 100,
            "delta_for_output": delta,
            "legs": [{"type": "call", "strike": strike, "action": "buy", "qty": 1}],
        }, None

    if strat == "Long Put":
        picked_row = pick_strike_by_delta(puts, spot, dte, TARGET_LONG_DELTA, "put")
        if not picked_row:
            return None, f"no put near target delta — {chain_diagnostics(puts, spot)}"
        row, delta = picked_row
        premium = mid_price(row)
        strike = float(row["strike"])
        return {
            "premium": premium, "strike_for_pot": strike, "collateral": premium,
            "breakeven": strike - premium, "max_loss": round(premium * 100, 2),
            "strike_label": f"${strike:.0f} P", "iv": safe_float(row.get("impliedVolatility")) * 100,
            "delta_for_output": delta,
            "legs": [{"type": "put", "strike": strike, "action": "buy", "qty": 1}],
        }, None

    if strat == "Bull Call Spread":
        # Defined-risk, cheaper alternative to Long Call: buy near the same
        # target delta Long Call uses, sell the 2nd-next further-OTM strike
        # to fund part of it — the same "further-OTM short leg" convention
        # Bull Put Spread/Bear Call Spread already use, just on the debit side.
        long_picked = pick_strike_by_delta(calls, spot, dte, TARGET_LONG_DELTA, "call")
        if not long_picked:
            return None, f"no call near target delta for long leg — {chain_diagnostics(calls, spot)}"
        l_row, l_delta = long_picked
        long_strike = float(l_row["strike"])
        higher_strikes = calls[calls["strike"] > long_strike].sort_values("strike")
        if higher_strikes.empty:
            return None, "no further-OTM strike available for the short leg"
        short_row = higher_strikes.iloc[min(1, len(higher_strikes) - 1)]
        short_strike = float(short_row["strike"])
        premium = mid_price(l_row) - mid_price(short_row)  # net debit paid
        width = short_strike - long_strike
        max_profit = width - premium
        if max_profit <= 0:
            return None, "no positive max profit (net debit exceeds spread width)"
        return {
            "premium": premium, "strike_for_pot": long_strike, "collateral": premium,
            "breakeven": long_strike + premium, "max_loss": round(premium * 100, 2),
            "max_profit": round(max_profit, 2),
            "strike_label": f"${long_strike:.0f}/{short_strike:.0f}", "iv": safe_float(l_row.get("impliedVolatility")) * 100,
            "delta_for_output": l_delta,
            "legs": [
                {"type": "call", "strike": long_strike, "action": "buy", "qty": 1},
                {"type": "call", "strike": short_strike, "action": "sell", "qty": 1},
            ],
        }, None

    if strat == "Bear Put Spread":
        # Mirror of Bull Call Spread, with puts: buy near Long Put's target
        # delta, sell the 2nd-next further-OTM (lower) strike to fund it.
        long_picked = pick_strike_by_delta(puts, spot, dte, TARGET_LONG_DELTA, "put")
        if not long_picked:
            return None, f"no put near target delta for long leg — {chain_diagnostics(puts, spot)}"
        l_row, l_delta = long_picked
        long_strike = float(l_row["strike"])
        lower_strikes = puts[puts["strike"] < long_strike].sort_values("strike", ascending=False)
        if lower_strikes.empty:
            return None, "no further-OTM strike available for the short leg"
        short_row = lower_strikes.iloc[min(1, len(lower_strikes) - 1)]
        short_strike = float(short_row["strike"])
        premium = mid_price(l_row) - mid_price(short_row)  # net debit paid
        width = long_strike - short_strike
        max_profit = width - premium
        if max_profit <= 0:
            return None, "no positive max profit (net debit exceeds spread width)"
        return {
            "premium": premium, "strike_for_pot": long_strike, "collateral": premium,
            "breakeven": long_strike - premium, "max_loss": round(premium * 100, 2),
            "max_profit": round(max_profit, 2),
            "strike_label": f"${long_strike:.0f}/{short_strike:.0f}", "iv": safe_float(l_row.get("impliedVolatility")) * 100,
            "delta_for_output": l_delta,
            "legs": [
                {"type": "put", "strike": long_strike, "action": "buy", "qty": 1},
                {"type": "put", "strike": short_strike, "action": "sell", "qty": 1},
            ],
        }, None

    return None, f"unknown strategy '{strat}'"


def build_trade_for_ticker(ticker_symbol, index):
    try:
        tk = yf.Ticker(ticker_symbol)
        history = tk.history(period="1y")
        if history.empty:
            print(f"  skip {ticker_symbol}: no price history")
            return None

        spot, spot_is_stale = resolve_current_price(tk, history, ticker_symbol, "options")
        if math.isnan(spot) or spot <= 0:
            # the most recent bar is occasionally incomplete/NaN right after close —
            # a NaN spot silently poisons every single strike's delta calculation
            # downstream (spot<=0 doesn't catch NaN; NaN just propagates through the
            # math with no error), which looks like "no strike matched" across the
            # entire chain rather than the actual, single-point root cause. Try
            # falling back a day before giving up.
            if len(history) >= 2:
                spot = float(history["Close"].iloc[-2])
                spot_is_stale = True  # admittedly using an even older bar now
            if math.isnan(spot) or spot <= 0:
                print(f"  skip {ticker_symbol}: spot price is invalid/NaN (most recent close data looks broken)")
                return None

        ema8 = compute_ema(history["Close"], 8).iloc[-1]
        ema20 = compute_ema(history["Close"], 20).iloc[-1]
        uptrend = ema8 > ema20
        near_ema = (abs(spot - ema8) / spot < 0.015) or (abs(spot - ema20) / spot < 0.015)
        atr = compute_atr(history)
        ivr, realized_vol, ivr_limited = iv_rank_proxy(history)

        today = datetime.now(timezone.utc).date()
        expirations = tk.options
        if not expirations:
            print(f"  skip {ticker_symbol}: no options listed")
            return None

        in_window_candidates, outside_window_candidates = rank_expirations(expirations, today)
        if not in_window_candidates and not outside_window_candidates:
            print(f"  skip {ticker_symbol}: no usable expiration (all listed dates are in the past or unparsable)")
            return None

        is_etf = ticker_symbol in KNOWN_ETFS

        # strategy selection: 1-in-4 tickers try a neutral, range-bound bet
        # (Iron Condor) regardless of trend, since it isn't directional the way
        # the rest are. The remaining tickers still split by trend: uptrend ->
        # bullish rotation, downtrend -> bearish rotation. Butterfly and
        # Calendar Spread were removed 2026-09-20 — their payoff shape didn't
        # fit the profit-target intent of this screen. Cash-Secured Put, Bull
        # Call Spread and Bear Put Spread added 2026-09-21 to round out each
        # side with a defined-risk debit alternative. Covered Call and
        # Cash-Secured Put were removed 2026-09-30: their 1-3% per-trade
        # return on full share/cash collateral never fit this screen's
        # profit-target filters, so they were generated but never shown.
        # 2026-10-01: Long Call, Long Put and Double Diagonal are no longer
        # generated — the page had dropped them from its strategy list, so
        # ~1 in 3 tickers each run produced a trade nobody could see. Every
        # slot now gets one of the five strategies the page shows: neutral ->
        # Iron Condor; trending tickers alternate between their direction's
        # credit and debit spread. (The Double Diagonal builder is left in
        # place, just unused.)
        # Strategy from momentum, not list position (2026-10-10). See
        # momentum_regime(): trending names get their direction's credit
        # spread (a debit spread is tried later only if options are cheap);
        # only mixed / flat names get an Iron Condor, and never a leveraged ETF.
        regime, regime_info = momentum_regime(history, spot)
        if regime == "neutral":
            if ticker_symbol in LEVERAGED_ETFS:
                print(f"  skip {ticker_symbol}: no clear trend, and leveraged ETFs don't get range (Iron Condor) trades")
                return None
            strat, side = "Iron Condor", "neutral"
        elif regime == "bull":
            strat, side = "Bull Put Spread", "bull"
        else:
            strat, side = "Bear Call Spread", "bear"

        # Evaluate every expiration candidate within the target window (up to the
        # cap), and keep whichever produces the best annualized profit — instead
        # of just taking the first usable one. This is also why different tickers
        # naturally land on different expiration dates now, rather than every name
        # converging on "whichever Friday is closest to 30 days out": each ticker's
        # own IV term structure and chain liquidity determines its own best pick.
        #
        # Ranks by RAW roc, not annualized ap (changed 2026-09-23 — see the
        # MAX_PLAUSIBLE_ROC/debit_spread_roc_ceiling comments elsewhere in this
        # file for why raw ROC, not ap, is the number that's actually
        # comparable across different DTEs). Annualizing multiplies by
        # 365/dte, which grows as dte shrinks — so ranking BY ap systematically
        # favors whichever surviving candidate has the shortest DTE, even
        # when a longer-dated one has a strictly better raw return. That bias
        # doesn't require an extreme, easily-flagged DTE to bite: it applies
        # at any DTE differential, just more visibly at the extremes — which
        # is exactly why capping ROC at short DTE (debit_spread_roc_ceiling)
        # alone wasn't enough; every ticker just piled onto the next-shortest
        # surviving candidate instead of actually being compared fairly.
        # Long Call/Long Put have no "roc"/"ap" at all (see
        # evaluate_expiration_candidate) — for those two, rank by probability
        # of profit instead, the only comparable-across-candidates number
        # they do produce.
        def _rank_key(result):
            return result["roc"] if result["roc"] is not None else result["pot"]

        best = None
        failure_reasons = []

        if strat == "Double Diagonal":
            # Spans two expirations — needs its own near+far candidate search,
            # not the single-expiration loop below (see build_double_diagonal's
            # docstring for why this can't go through evaluate_expiration_candidate).
            best, reason = build_double_diagonal(tk, spot, expirations, today, atr)
            if not best:
                print(f"  skip {ticker_symbol}: Double Diagonal unusable — {reason}")
                return None
            print(f"    {ticker_symbol} [Double Diagonal] {best['exp']}({best['dte']}d)/{best['backExp']}({(datetime.strptime(best['backExpDate'], '%Y-%m-%d').date() - today).days}d): pot:{best['pot']}%")
        else:
            def _search(strat, side):
                best = None
                failure_reasons = []
                evaluated_log = []  # every candidate's outcome, logged regardless of win/loss —
                                     # needed to see WHY a ticker keeps landing on the same
                                     # expiration: genuinely winning on merit vs. every
                                     # alternative failing validation outright.
                for cand_exp, cand_dte in in_window_candidates[:MAX_CANDIDATES_TO_EVALUATE]:
                    result, reason = evaluate_expiration_candidate(tk, strat, side, spot, cand_exp, cand_dte, atr)
                    if result:
                        log_val = f"ap:{result['ap']}%" if result['ap'] is not None else f"potProfit:{result['pot']}%"
                        evaluated_log.append(f"{cand_exp}({cand_dte}d)={log_val}")
                        if best is None or _rank_key(result) > _rank_key(best):
                            best = result
                    else:
                        evaluated_log.append(f"{cand_exp}({cand_dte}d)=FAILED:{reason}")
                        failure_reasons.append(f"{cand_exp} ({cand_dte}d): {reason}")

                # Only reach outside the 7-45d window if NOTHING in-window priced —
                # a true last resort (see rank_expirations' docstring for why this
                # can't just be "whichever 8 candidates come first regardless of
                # bucket": a short-DTE candidate's annualized profit would win the
                # ranking almost automatically against legitimate longer-dated
                # ones, not on merit, just because it's short-dated.
                if not best and in_window_candidates:
                    for cand_exp, cand_dte in outside_window_candidates[:MAX_CANDIDATES_TO_EVALUATE]:
                        result, reason = evaluate_expiration_candidate(tk, strat, side, spot, cand_exp, cand_dte, atr)
                        if result:
                            log_val = f"ap:{result['ap']}%" if result['ap'] is not None else f"potProfit:{result['pot']}%"
                            evaluated_log.append(f"{cand_exp}({cand_dte}d,outside-window)={log_val}")
                            if best is None or _rank_key(result) > _rank_key(best):
                                best = result
                        else:
                            evaluated_log.append(f"{cand_exp}({cand_dte}d,outside-window)=FAILED:{reason}")
                            failure_reasons.append(f"{cand_exp} ({cand_dte}d, outside window): {reason}")

                print(f"    {ticker_symbol} [{strat}] evaluated {len(evaluated_log)} candidate(s): {' | '.join(evaluated_log)}")
                return best, failure_reasons

            best, failure_reasons = _search(strat, side)

            # Credit vs debit (2026-10-10): the credit spread is the default.
            # If its options turn out clearly cheap (implied vol well under the
            # stock's realized vol) the direction's debit spread is tried and
            # kept when it prices; it's also the fallback when no credit
            # spread passes the liquidity / pricing checks at all.
            if side in ("bull", "bear"):
                debit = "Bull Call Spread" if side == "bull" else "Bear Put Spread"
                cheap = bool(best and realized_vol and best.get("iv")
                             and best["iv"] / realized_vol < CHEAP_OPTIONS_IV_RV)
                if cheap or not best:
                    why = "options cheap (IV/RV %.2f)" % (best["iv"] / realized_vol) if cheap else "no credit spread passed"
                    print(f"    {ticker_symbol}: trying {debit} — {why}")
                    d_best, d_fail = _search(debit, side)
                    if d_best:
                        best, strat = d_best, debit
                    elif not best:
                        failure_reasons += d_fail

            if not best:
                tried = len(failure_reasons)
                print(f"  skip {ticker_symbol}: {strat} unusable across {tried} expiration(s) tried — {'; '.join(failure_reasons)}")
                return None

        dte = best["dte"]

        # earnings within the chosen expiration's window? also capture days-until
        # regardless of window, since "next earnings in 4 days" is useful context
        # even for a trade that isn't flagged as earnings-risky.
        earnings_soon = False
        days_to_earnings = None
        try:
            edates = tk.get_earnings_dates(limit=4)
            if edates is not None and not edates.empty:
                for dt in edates.index:
                    d = dt.date() if hasattr(dt, "date") else dt
                    delta_days = (d - today).days
                    if delta_days >= 0 and (days_to_earnings is None or delta_days < days_to_earnings):
                        days_to_earnings = delta_days
                    if 0 <= delta_days <= dte:
                        earnings_soon = True
        except Exception:
            pass  # earnings calendar not always available — leave as None/False

        news_sentiment, news_sentiment_label, news_headlines = fetch_news_and_sentiment(tk, ticker_symbol)

        if strat in ("Long Call", "Long Put"):
            score = composite_score_long_option(best["pot"], ivr, best["breakevenMovePct"])
        elif strat == "Double Diagonal":
            score = composite_score_double_diagonal(best["pot"], ivr)
        else:
            score = composite_score(best["ap"], best["pot"], ivr)
        vol_regime, iv_rv_ratio = classify_vol_regime(best["iv"], realized_vol)

        return {
            "sym": ticker_symbol,
            "strat": strat,
            "side": side,
            "isETF": is_etf,
            "spot": round(spot, 2),
            "spotPriceStale": spot_is_stale,
            "strike": best["strike"],
            "hedge": best["hedge"],
            "exp": best["exp"],
            "dte": best["dte"],
            "backExp": best.get("backExp"),
            "backExpDate": best.get("backExpDate"),
            "pot": best["pot"],
            "ap": best["ap"],
            "ivr": ivr,
            "ivRankLimited": ivr_limited,
            "dailyReturn": best["dailyReturn"],
            "roc": best["roc"],
            "score": score,
            "buy": bool(uptrend),
            "momentum": regime,                       # bull / bear / neutral (see momentum_regime)
            "ret3m": regime_info.get("ret3m"),        # ~3-month % change
            "sell": bool(not uptrend),
            "ema": bool(near_ema),
            "earningsSoon": earnings_soon,
            "daysToEarnings": days_to_earnings,
            "marginOfSafety": best["marginOfSafety"],
            "delta": best["delta"],
            "iv": best["iv"],
            "premium": best["premium"],
            "breakeven": best["breakeven"],
            "breakevenLow": best["breakevenLow"],
            "breakevenHigh": best["breakevenHigh"],
            "breakevenMovePct": best.get("breakevenMovePct"),
            "maxLoss": best["maxLoss"],
            "pcOI": best["pcOI"],
            "pcVol": best["pcVol"],
            "debitStrategy": best.get("debitStrategy", False),
            "liquidity": best.get("liquidity"),
            "potIsProfitProb": best.get("potIsProfitProb", False),
            "newsSentiment": news_sentiment,
            "newsSentimentLabel": news_sentiment_label,
            "newsHeadlines": news_headlines,
            "volRegime": vol_regime,
            "ivRvRatio": iv_rv_ratio,
            "realizedVol": realized_vol,
            "expDate": best.get("expDate"),
            "legs": best.get("legs"),
        }

    except Exception as e:
        print(f"  skip {ticker_symbol}: {e}")
        return None


# --- On-demand single-ticker lookup ------------------------------------------
# Added 2026-09-19 so the UI can offer a "look up any symbol" box on top of
# the fixed nightly watchlist above, triggered on demand (see run_lookup()
# and the --ticker CLI flag at the bottom of this file). Reuses every pricing/
# validation helper the batch path above uses — the only real difference is
# how the strategy gets picked.

def pick_lookup_strategy(uptrend, near_ema):
    """
    Picks ONE strategy for an on-demand lookup, from trend alone. This is
    deliberately NOT the nightly batch's `index % 4` / `index % 3` rotation in
    build_trade_for_ticker() above — that rotation exists purely to keep the
    fixed WATCHLIST diversified across strategy types across many tickers, and
    has no meaning applied to a single ad-hoc symbol (a user typing in one
    ticker doesn't have a "list position" to rotate on).

    Also deliberately narrower than "evaluate every strategy and keep the
    best": each strategy evaluated costs up to MAX_CANDIDATES_TO_EVALUATE
    separate options-chain fetches against Yahoo's free, rate-limit-prone
    feed (see the module docstring and chain_diagnostics() above) — trying
    several strategies per click would multiply that cost every time someone
    uses the lookup box. One well-established, defined-risk pick per trend
    bucket keeps a single lookup roughly as expensive as one ticker in the
    nightly batch, not several.
    """
    if near_ema:
        return "Iron Condor", "neutral"
    if uptrend:
        return "Bull Put Spread", "bull"
    return "Bear Call Spread", "bear"


def build_lookup_trade(ticker_symbol):
    """
    On-demand equivalent of build_trade_for_ticker() above, for a single
    symbol typed into the UI rather than a slot in the fixed watchlist.
    Returns (trade_dict, None) on success, or (None, reason_string) on
    failure — unlike the batch path (which only logs skip reasons to stderr
    and silently omits the ticker from data.json), the reason here is shown
    directly to whoever typed the symbol in, so it needs to read as an
    explanation, not a log line.
    """
    try:
        tk = yf.Ticker(ticker_symbol)
        history = tk.history(period="1y")
        if history.empty:
            return None, "No price history found for this symbol — double-check the ticker."

        spot, spot_is_stale = resolve_current_price(tk, history, ticker_symbol, "lookup")
        if math.isnan(spot) or spot <= 0:
            if len(history) >= 2:
                spot = float(history["Close"].iloc[-2])
                spot_is_stale = True
            if math.isnan(spot) or spot <= 0:
                return None, "This symbol's current price looks invalid — Yahoo's data may be broken for it right now."

        ema8 = compute_ema(history["Close"], 8).iloc[-1]
        ema20 = compute_ema(history["Close"], 20).iloc[-1]
        uptrend = bool(ema8 > ema20)
        near_ema = bool((abs(spot - ema8) / spot < 0.015) or (abs(spot - ema20) / spot < 0.015))
        atr = compute_atr(history)
        ivr, realized_vol, ivr_limited = iv_rank_proxy(history)

        today = datetime.now(timezone.utc).date()
        expirations = tk.options
        if not expirations:
            return None, "This symbol doesn't have listed options."

        in_window_candidates, outside_window_candidates = rank_expirations(expirations, today)
        if not in_window_candidates and not outside_window_candidates:
            return None, "No usable (future-dated) options expiration is listed for this symbol."

        # Same momentum rule as the batch (2026-10-10), so a looked-up ticker
        # gets the strategy its trend supports.
        regime, _ = momentum_regime(history, spot)
        if regime == "neutral" and ticker_symbol in LEVERAGED_ETFS:
            return None, "No clear trend right now, and leveraged ETFs don't get range (Iron Condor) trades."
        strat, side = {"bull": ("Bull Put Spread", "bull"), "bear": ("Bear Call Spread", "bear")}.get(regime, ("Iron Condor", "neutral"))
        is_etf = ticker_symbol in KNOWN_ETFS

        # See the main pipeline's _rank_key for why this ranks by raw roc, not ap.
        def _rank_key(result):
            return result["roc"] if result["roc"] is not None else result["pot"]

        best = None
        failure_reasons = []
        for cand_exp, cand_dte in in_window_candidates[:MAX_CANDIDATES_TO_EVALUATE]:
            result, reason = evaluate_expiration_candidate(tk, strat, side, spot, cand_exp, cand_dte, atr)
            if result:
                if best is None or _rank_key(result) > _rank_key(best):
                    best = result
            else:
                failure_reasons.append(f"{cand_exp} ({cand_dte}d): {reason}")

        # Same true-last-resort fallback as the main pipeline (see
        # rank_expirations' docstring) — only reach outside the window if
        # nothing in-window could be priced at all.
        if not best and in_window_candidates:
            for cand_exp, cand_dte in outside_window_candidates[:MAX_CANDIDATES_TO_EVALUATE]:
                result, reason = evaluate_expiration_candidate(tk, strat, side, spot, cand_exp, cand_dte, atr)
                if result:
                    if best is None or _rank_key(result) > _rank_key(best):
                        best = result
                else:
                    failure_reasons.append(f"{cand_exp} ({cand_dte}d, outside window): {reason}")

        if not best:
            reasons = "; ".join(failure_reasons[:3]) if failure_reasons else "no usable expirations"
            return None, f"Couldn't build a {strat} for this symbol right now — {reasons}"

        today_iso = today
        earnings_soon = False
        days_to_earnings = None
        try:
            edates = tk.get_earnings_dates(limit=4)
            if edates is not None and not edates.empty:
                for dt in edates.index:
                    d = dt.date() if hasattr(dt, "date") else dt
                    delta_days = (d - today_iso).days
                    if delta_days >= 0 and (days_to_earnings is None or delta_days < days_to_earnings):
                        days_to_earnings = delta_days
                    if 0 <= delta_days <= best["dte"]:
                        earnings_soon = True
        except Exception:
            pass

        news_sentiment, news_sentiment_label, news_headlines = fetch_news_and_sentiment(tk, ticker_symbol)

        if strat in ("Long Call", "Long Put"):
            score = composite_score_long_option(best["pot"], ivr, best["breakevenMovePct"])
        elif strat == "Double Diagonal":
            score = composite_score_double_diagonal(best["pot"], ivr)
        else:
            score = composite_score(best["ap"], best["pot"], ivr)
        vol_regime, iv_rv_ratio = classify_vol_regime(best["iv"], realized_vol)

        return {
            "sym": ticker_symbol,
            "strat": strat,
            "side": side,
            "isETF": is_etf,
            "spot": round(spot, 2),
            "spotPriceStale": spot_is_stale,
            "strike": best["strike"],
            "hedge": best["hedge"],
            "exp": best["exp"],
            "dte": best["dte"],
            "backExp": best.get("backExp"),
            "backExpDate": best.get("backExpDate"),
            "pot": best["pot"],
            "ap": best["ap"],
            "ivr": ivr,
            "ivRankLimited": ivr_limited,
            "dailyReturn": best["dailyReturn"],
            "roc": best["roc"],
            "score": score,
            "buy": uptrend,
            "sell": not uptrend,
            "ema": near_ema,
            "earningsSoon": earnings_soon,
            "daysToEarnings": days_to_earnings,
            "marginOfSafety": best["marginOfSafety"],
            "delta": best["delta"],
            "iv": best["iv"],
            "premium": best["premium"],
            "breakeven": best["breakeven"],
            "breakevenLow": best["breakevenLow"],
            "breakevenHigh": best["breakevenHigh"],
            "breakevenMovePct": best.get("breakevenMovePct"),
            "maxLoss": best["maxLoss"],
            "pcOI": best["pcOI"],
            "pcVol": best["pcVol"],
            "debitStrategy": best.get("debitStrategy", False),
            "liquidity": best.get("liquidity"),
            "potIsProfitProb": best.get("potIsProfitProb", False),
            "newsSentiment": news_sentiment,
            "newsSentimentLabel": news_sentiment_label,
            "newsHeadlines": news_headlines,
            "volRegime": vol_regime,
            "ivRvRatio": iv_rv_ratio,
            "realizedVol": realized_vol,
            "expDate": best.get("expDate"),
            "legs": best.get("legs"),
        }, None

    except Exception as e:
        return None, f"Unexpected error while building a trade for this symbol: {e}"


LOOKUP_OUTPUT_PATH = "lookup.json"


def run_lookup(raw_symbol):
    """
    Entry point for `--ticker SYMBOL`. Always writes lookup.json — even on
    failure — so the frontend polling for a result never waits forever on a
    file that never changes; a "status": "error" response is itself the
    answer. Never raises: an on-demand run failing loudly would still leave
    the workflow's commit step with nothing new to commit, which is the same
    "frontend polls forever" problem from the other direction.
    """
    ticker_symbol = re.sub(r"[^A-Za-z0-9.\-]", "", (raw_symbol or "")).strip().upper()
    output = {
        "requested_ticker": ticker_symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance (unofficial, free, EOD)",
        "status": "error",
        "error": None,
        "trade": None,
        "equity": None,
    }

    if not ticker_symbol or not (1 <= len(ticker_symbol) <= 10):
        output["error"] = "Enter a valid ticker symbol (letters/numbers, up to 10 characters)."
        with open(LOOKUP_OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Lookup rejected: invalid symbol {raw_symbol!r}")
        return

    print(f"On-demand lookup: {ticker_symbol}")
    try:
        trade, trade_error = build_lookup_trade(ticker_symbol)
    except Exception as e:
        trade, trade_error = None, f"Unexpected error: {e}"

    try:
        equity = build_equity_snapshot(ticker_symbol, ticker_symbol in KNOWN_ETFS)
    except Exception as e:
        print(f"  lookup equity snapshot failed for {ticker_symbol}: {e}")
        equity = None

    output["trade"] = trade
    output["equity"] = equity

    if trade is None and equity is None:
        output["status"] = "error"
        output["error"] = trade_error or "Couldn't find usable data for this symbol — check it's a valid, actively-traded ticker."
        print(f"  lookup failed: {output['error']}")
    else:
        output["status"] = "ok"
        # Surface a soft warning even on partial success (e.g. equity data came
        # back but no usable options trade) rather than silently dropping it.
        if trade is None:
            output["error"] = trade_error

    with open(LOOKUP_OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote lookup.json for {ticker_symbol} (status={output['status']})")


def compute_rsi(closes, period=14):
    """Standard 14-day RSI, using an EWM approximation of Wilder's smoothing."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def classify_trend(ema8, ema20, ema50):
    """Simple moving-average-alignment trend classification — not predictive,
    just describes the current shape of the average, same spirit as the
    options side's EMA8/20 buy/sell signal."""
    if ema8 > ema20 > ema50:
        return "Strong Uptrend"
    if ema8 > ema20:
        return "Uptrend"
    if ema8 < ema20 < ema50:
        return "Strong Downtrend"
    if ema8 < ema20:
        return "Downtrend"
    return "Neutral"


def equity_composite_score(trend, rsi, chg_6m, valuation_component=5):
    """Illustrative 0-10 blend for a plain stock/ETF buy-candidate view — same
    honesty caveat as the options side's composite_score: this is a heuristic
    blend I picked, not a validated signal. Adjust the weights to your taste.
    valuation_component (0-10 — see valuation_score_component()) is passed
    with side="bull" for this function specifically: a plain equity view is
    implicitly "would I want to own this," which is a bullish framing, unlike
    an options trade whose side varies with the strategy picked. Defaults to
    5 (neutral/no-op) when no peer valuation is available for this ticker."""
    trend_map = {"Strong Uptrend": 10, "Uptrend": 7, "Neutral": 5, "Downtrend": 3, "Strong Downtrend": 0}
    trend_component = trend_map.get(trend, 5)

    # reward "healthy" momentum (45-65 RSI), penalize extreme overbought/oversold
    if rsi is None:
        rsi_component = 5
    elif 45 <= rsi <= 65:
        rsi_component = 8
    elif rsi > 80 or rsi < 20:
        rsi_component = 2
    else:
        rsi_component = 5

    momentum_component = min(10, max(0, 5 + (chg_6m or 0) / 4))  # +20% over 6mo -> 10

    score = (0.35 * trend_component + 0.23 * rsi_component
             + 0.22 * momentum_component + 0.20 * valuation_component)
    return round(min(10, max(0, score)), 1)


def build_equity_snapshot(ticker_symbol, is_etf):
    """
    Evaluates a ticker as a plain stock/ETF buy candidate — the same free
    yfinance data source as the options pipeline above, but scored on
    equity-appropriate criteria (trend, momentum, valuation, dividend yield)
    instead of options Greeks. Returns a dict, or None if the ticker's data
    is unusable.

    IMPORTANT — same spirit as the options docstring at the top of this file:
    - `peRatio` / `dividendYield` / `marketCap` come from yfinance's .info,
      which is frequently incomplete, especially for ETFs — expense ratio,
      holdings, and AUM aren't reliably available for free at all, so they're
      not included here. A missing value means "not reported," not "zero."
    - `rsi` is a standard 14-day RSI (Wilder-style), a momentum indicator, not
      a prediction.
    - `trend` is a moving-average-alignment heuristic (8/20/50-day EMA).
    - `equityScore` is an illustrative weighted blend — see
      `equity_composite_score()` above to change what it weighs.
    """
    try:
        tk = yf.Ticker(ticker_symbol)
        history = tk.history(period="1y")
        if history.empty or len(history) < 60:
            print(f"  skip {ticker_symbol} (equity): insufficient price history")
            return None

        closes = history["Close"]
        price, price_is_stale = resolve_current_price(tk, history, ticker_symbol, "equity")
        if math.isnan(price) or price <= 0:
            # same fallback as build_trade_for_ticker above — the most recent
            # bar is occasionally incomplete/NaN right after close. Confirmed
            # in production: every equity snapshot failed with "invalid
            # current price" in the exact same run where every options trade
            # for the same tickers succeeded, because that function already
            # had this fallback and this one didn't.
            if len(closes) >= 2:
                price = float(closes.iloc[-2])
                price_is_stale = True  # admittedly using an even older bar now
            if math.isnan(price) or price <= 0:
                print(f"  skip {ticker_symbol} (equity): current price is invalid/NaN (most recent close data looks broken)")
                return None

        high_52wk = float(history["High"].max())
        low_52wk = float(history["Low"].min())
        pct_from_high = round((price - high_52wk) / high_52wk * 100, 1) if high_52wk > 0 else None
        pct_from_low = round((price - low_52wk) / low_52wk * 100, 1) if low_52wk > 0 else None

        ema8 = compute_ema(closes, 8).iloc[-1]
        ema20 = compute_ema(closes, 20).iloc[-1]
        ema50 = compute_ema(closes, 50).iloc[-1] if len(closes) >= 50 else ema20
        trend = classify_trend(ema8, ema20, ema50)

        rsi_series = compute_rsi(closes)
        last_rsi = rsi_series.iloc[-1]
        rsi = round(float(last_rsi), 1) if not math.isnan(last_rsi) else None

        def pct_change(days_back):
            if len(closes) <= days_back:
                return None
            past = closes.iloc[-days_back - 1]
            if past is None or math.isnan(past) or past <= 0:
                return None
            return round((price - past) / past * 100, 1)

        chg_1m = pct_change(21)
        chg_3m = pct_change(63)
        chg_6m = pct_change(126)
        chg_1y = pct_change(min(252, len(closes) - 1))

        vol_ratio = None
        if "Volume" in history.columns:
            avg_vol_30 = history["Volume"].tail(30).mean()
            latest_vol = history["Volume"].iloc[-1]
            if avg_vol_30 and avg_vol_30 > 0 and not math.isnan(avg_vol_30) and not math.isnan(latest_vol):
                vol_ratio = round(latest_vol / avg_vol_30, 2)

        # best-effort fundamentals — often missing, especially for ETFs
        pe_ratio, dividend_yield, market_cap = None, None, None
        sector, industry, peg_ratio = None, None, None
        try:
            info = tk.info or {}
            raw_pe = info.get("trailingPE")
            pe_ratio = round(raw_pe, 1) if isinstance(raw_pe, (int, float)) else None
            raw_dy = info.get("dividendYield")
            # yfinance's dividendYield is already a percentage value (e.g. 0.73
            # meaning 0.73%), NOT a fraction needing *100 — confirmed against
            # real known yields after an earlier version of this code multiplied
            # by 100 and produced "AAPL: 34%", "JPM: 167%" etc. Sanity-clamp
            # anyway: a legitimate stock/ETF yield above ~20% is essentially
            # never real (usually a units mismatch or a data glitch), so treat
            # it as unreliable rather than display an obviously-wrong number.
            if isinstance(raw_dy, (int, float)) and 0 <= raw_dy <= 20:
                dividend_yield = round(raw_dy, 2)
            market_cap = info.get("marketCap")
            # sector/industry/pegRatio, added for compute_sector_valuations()
            # (2026-09-24) — same tk.info call as the fields above, so this
            # costs nothing extra. ETFs generally don't report a sector, which
            # is fine: they just won't get a peer valuation (no basis to pick
            # peers for one anyway).
            sector = info.get("sector") or None
            industry = info.get("industry") or None
            raw_peg = info.get("pegRatio") or info.get("trailingPegRatio")
            peg_ratio = round(raw_peg, 2) if isinstance(raw_peg, (int, float)) else None
        except Exception:
            pass  # fundamentals unavailable — leave as None, not zero

        # reuse the same news/sentiment pipeline as the options side — already
        # hardened against yfinance's .news not being reliably ticker-scoped
        news_sentiment, news_sentiment_label, news_headlines = fetch_news_and_sentiment(tk, ticker_symbol)

        score = equity_composite_score(trend, rsi, chg_6m)

        return {
            "sym": ticker_symbol,
            "isETF": is_etf,
            "price": round(price, 2),
            "priceStale": price_is_stale,
            "high52wk": round(high_52wk, 2),
            "low52wk": round(low_52wk, 2),
            "pctFromHigh": pct_from_high,
            "pctFromLow": pct_from_low,
            "trend": trend,
            "rsi": rsi,
            "chg1m": chg_1m,
            "chg3m": chg_3m,
            "chg6m": chg_6m,
            "chg1y": chg_1y,
            "volRatio": vol_ratio,
            "peRatio": pe_ratio,
            "sector": sector,
            "industry": industry,
            "pegRatio": peg_ratio,
            "dividendYield": dividend_yield,
            "marketCap": market_cap,
            "score": score,
            "newsSentiment": news_sentiment,
            "newsSentimentLabel": news_sentiment_label,
            "newsHeadlines": news_headlines,
        }
    except Exception as e:
        print(f"  skip {ticker_symbol} (equity): {e}")
        return None


# --- Market snapshot (added 2026-09-29) --------------------------------------
# Four headline gauges for the page's market-condition tiles: the QQQ / SPY /
# IWM ETFs (Nasdaq-100, S&P 500, Russell 2000 small caps) plus the VIX index,
# each with the move vs. the previous close. Switched from index levels
# (^IXIC/^GSPC/^DJI) to tradeable ETFs on 2026-09-30. On an
# intraday run yfinance's last daily bar is today's in-progress session, so
# the change is "today so far"; on the post-close run it's the full day.
# "inverse" marks gauges where a rise is bad for stocks (VIX) so the frontend
# can color them the right way round.
MARKET_GAUGES = [
    {"key": "qqq", "label": "QQQ", "symbol": "QQQ"},
    {"key": "spy", "label": "SPY", "symbol": "SPY"},
    {"key": "iwm", "label": "IWM", "symbol": "IWM"},
    {"key": "vix", "label": "VIX", "symbol": "^VIX", "inverse": True},
]


def build_market_snapshot():
    """
    Returns {"asOf": iso, "gauges": [...]} or None. Best-effort per gauge: one
    index failing to fetch just drops that tile; the frontend hides the whole
    strip only when every gauge failed. Never raises.
    """
    gauges = []
    for g in MARKET_GAUGES:
        try:
            hist = yf.Ticker(g["symbol"]).history(period="5d")
            closes = hist["Close"].dropna() if not hist.empty else []
            if len(closes) < 2:
                print(f"  market: not enough history for {g['symbol']}, skipping")
                continue
            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2])
            if math.isnan(last) or math.isnan(prev) or prev <= 0:
                continue
            gauges.append({
                "key": g["key"],
                "label": g["label"],
                "symbol": g["symbol"],
                "value": round(last, 2),
                "change": round(last - prev, 2),
                "changePct": round((last - prev) / prev * 100, 2),
                "inverse": bool(g.get("inverse", False)),
            })
        except Exception as e:
            print(f"  market: {g['symbol']} failed: {e}")
    if not gauges:
        return None
    return {"asOf": datetime.now(timezone.utc).isoformat(), "gauges": gauges}


def build_hedge_candidate():
    """
    Suggests one small tail-risk hedge — a single OTM QQQ put — to sit
    alongside the strategies above, all of which sell premium (short vol) and
    so share the same bad-day exposure: a sudden move that spikes IV and goes
    against several short strikes at once. Deliberately minimal: this returns
    one contract's terms and cost, not a recommended size or an auto-sized
    position — how many (if any) to actually buy is a sizing decision the
    frontend leaves to the person, same as every other number in this file.
    See the HEDGE_* constants above for the selection rules. Returns a dict,
    or None if QQQ's chain isn't usable right now — the frontend treats a
    missing "hedge" key as "not shown," not an error.
    """
    try:
        tk = yf.Ticker(HEDGE_TICKER)
        history = tk.history(period="5d")
        if history.empty:
            print(f"  skip hedge candidate: no price history for {HEDGE_TICKER}")
            return None
        spot = float(history["Close"].iloc[-1])
        if math.isnan(spot) or spot <= 0:
            print(f"  skip hedge candidate: {HEDGE_TICKER} spot price invalid")
            return None

        expirations = tk.options
        if not expirations:
            print(f"  skip hedge candidate: no options listed for {HEDGE_TICKER}")
            return None

        # Same closest-to-target-window ranking as rank_expirations() above,
        # but against HEDGE_TARGET_DTE_MIN/MAX rather than the strategy
        # picker's 7-45d window — kept as a local copy rather than
        # parameterizing the shared helper, so this doesn't risk changing
        # behavior for every existing strategy.
        today = datetime.now(timezone.utc).date()
        target_mid = (HEDGE_TARGET_DTE_MIN + HEDGE_TARGET_DTE_MAX) / 2
        in_window, outside_window = [], []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte <= 0:
                continue
            diff = abs(dte - target_mid)
            bucket = in_window if HEDGE_TARGET_DTE_MIN <= dte <= HEDGE_TARGET_DTE_MAX else outside_window
            bucket.append((diff, exp_str, dte))
        in_window.sort(key=lambda x: x[0])
        outside_window.sort(key=lambda x: x[0])
        candidates = [(exp_str, dte) for _, exp_str, dte in (in_window + outside_window)]
        if not candidates:
            print(f"  skip hedge candidate: no usable {HEDGE_TICKER} expiration")
            return None

        # Fall back to the next-ranked expiration if a chain turns out
        # unusable (all-NaN IV on a bad data day) — same pattern the
        # strategy picker uses instead of giving up on the first failure.
        for exp_str, dte in candidates[:5]:
            try:
                chain = tk.option_chain(exp_str)
            except Exception as e:
                print(f"  hedge candidate: {HEDGE_TICKER} {exp_str} chain fetch failed: {e}")
                continue
            puts = chain.puts
            if puts is None or puts.empty:
                continue
            picked = pick_strike_by_delta(puts, spot, dte, HEDGE_TARGET_DELTA, "put")
            if not picked:
                continue
            row, delta = picked
            premium = mid_price(row)
            if premium <= 0:
                continue
            iv = safe_float(row.get("impliedVolatility"), default=0.0) * 100
            strike = safe_float(row.get("strike"), default=0.0)
            exp_label = (datetime.strptime(exp_str, "%Y-%m-%d").strftime("%b %-d") if sys.platform != "win32"
                         else datetime.strptime(exp_str, "%Y-%m-%d").strftime("%b %d").replace(" 0", " "))
            return {
                "symbol": HEDGE_TICKER,
                "spot": round(spot, 2),
                "strike": round(strike, 2),
                "otmPct": round((spot - strike) / spot * 100, 1),
                "exp": exp_label,
                "expDate": exp_str,
                "dte": dte,
                "delta": round(delta, 2),
                "iv": round(iv, 1),
                "premium": round(premium, 2),
                "costPerContract": round(premium * 100, 2),
            }

        print(f"  skip hedge candidate: no usable put found across {HEDGE_TICKER} candidates")
        return None
    except Exception as e:
        print(f"  skip hedge candidate: {e}")
        return None


# --- Tracked position P&L (user-selected "backtest" tracking) -----------------
#
# Not part of the recommendation pipeline above — this re-prices whatever
# trades the person has chosen to track from the frontend (a "Track" button
# on each recommendation, saved to their browser and hand-committed to the
# repo as tracked_positions.json; see the frontend's Backtest tab). Every
# leg captured at tracking time (see the "legs"/"expDate"/"backExpDate" keys
# added throughout the strategy builders above) is re-quoted here against
# the CURRENT live chain, so the P&L shown is a real mark-to-market number,
# not a modeled/estimated one — with one unavoidable catch: yfinance only
# quotes chains for expirations that haven't happened yet, so a position
# past its expiration date can no longer be re-priced at all (options
# quotes aren't a historical data series the way stock closes are). Once a
# position's expiration passes, its last known mark-to-market snapshot is
# kept as-is and it's flagged "expired" rather than silently going stale.
TRACKED_POSITIONS_PATH = "tracked_positions.json"
TRACKED_PNL_PATH = "tracked_pnl.json"


def _price_leg(chain_cache, tk, exp_date, leg):
    """
    Looks up one leg's current mid price against the live chain for exp_date,
    fetching+caching that chain (chain_cache keyed by exp_date) at most once
    even when several legs of the same position share an expiration.
    Returns None (not an exception) if the chain can't be fetched or the
    exact strike is no longer listed — the caller treats that as "can't
    update this position right now," not a crash.
    """
    if exp_date not in chain_cache:
        try:
            chain_cache[exp_date] = tk.option_chain(exp_date)
        except Exception as e:
            print(f"    tracked-position chain fetch failed for {exp_date}: {e}")
            chain_cache[exp_date] = None
    chain = chain_cache[exp_date]
    if chain is None:
        return None
    frame = chain.calls if leg["type"] == "call" else chain.puts
    match = frame[(frame["strike"] - float(leg["strike"])).abs() < 0.01]
    if match.empty:
        return None
    return mid_price(match.iloc[0])


def _reprice_position(pos):
    """
    Re-quotes every leg of one tracked position and returns a snapshot dict,
    or None if any leg couldn't be priced this run (missing chain, delisted
    strike, etc.) — the caller keeps the position's prior history untouched
    in that case rather than recording a bogus/partial number.
    """
    legs = pos.get("legs")
    exp_date = pos.get("expDate")
    if not legs or not exp_date:
        return None  # tracked before the legs/expDate fields existed, or malformed

    tk = yf.Ticker(pos["sym"])
    chain_cache = {}
    net_credit = 0.0
    for leg in legs:
        leg_exp = pos.get("backExpDate") if leg.get("exp") == "back" else exp_date
        price = _price_leg(chain_cache, tk, leg_exp, leg)
        if price is None:
            return None
        qty = leg.get("qty", 1)
        sign = 1 if leg["action"] == "sell" else -1
        net_credit += price * qty * sign

    # Same orientation as the entry "premium" stored at tracking time: for a
    # credit strategy premium = money received (net_credit as-is); for a
    # debit strategy premium = money paid (net_credit flipped, since a debit
    # position's legs are stored the same buy/sell way a credit one's are).
    current_premium = net_credit if not pos.get("debitStrategy") else -net_credit
    entry_premium = pos.get("premium")
    if entry_premium in (None, 0):
        pnl_dollars, pnl_pct = None, None
    elif pos.get("debitStrategy"):
        pnl_dollars = round((current_premium - entry_premium) * 100, 2)
        pnl_pct = round((current_premium - entry_premium) / entry_premium * 100, 2)
    else:
        pnl_dollars = round((entry_premium - current_premium) * 100, 2)
        pnl_pct = round((entry_premium - current_premium) / entry_premium * 100, 2)

    return {
        "currentPremium": round(current_premium, 2),
        "pnlDollars": pnl_dollars,
        "pnlPct": pnl_pct,
    }


def update_tracked_positions():
    """
    Reads tracked_positions.json (absent = nothing tracked yet = no-op),
    re-prices every still-open position, and writes tracked_pnl.json with an
    appended history entry per position. Every failure mode here (missing
    file, malformed JSON, one bad position, a network error) is caught and
    logged rather than raised, since this feature must never be able to
    take down the core recommendation run in main() below.
    """
    if not os.path.exists(TRACKED_POSITIONS_PATH):
        return  # nothing tracked yet — not an error, just nothing to do

    try:
        with open(TRACKED_POSITIONS_PATH) as f:
            tracked = json.load(f)
    except Exception as e:
        print(f"  tracked positions: couldn't read {TRACKED_POSITIONS_PATH}: {e}")
        return

    prior_by_id = {}
    if os.path.exists(TRACKED_PNL_PATH):
        try:
            with open(TRACKED_PNL_PATH) as f:
                prior_by_id = {p["id"]: p for p in json.load(f).get("positions", [])}
        except Exception as e:
            print(f"  tracked positions: couldn't read prior {TRACKED_PNL_PATH}, starting fresh: {e}")

    today = datetime.now(timezone.utc).date()
    out_positions = []
    for pos in tracked:
        pos_id = pos.get("id")
        if not pos_id or not pos.get("sym") or not pos.get("expDate"):
            print(f"  tracked positions: skipping malformed entry {pos.get('id', '?')}")
            continue

        prior = prior_by_id.get(pos_id, {})
        history = list(prior.get("history", []))
        exp_date = datetime.strptime(pos["expDate"], "%Y-%m-%d").date()
        dte_remaining = (exp_date - today).days

        if dte_remaining < 0:
            # Past expiration — yfinance no longer has a chain to quote, so
            # this is the last snapshot this position will ever get. Keep
            # whatever P&L was last recorded and just flip the status.
            out_positions.append({
                **{k: v for k, v in prior.items() if k not in ("dteRemaining", "status", "lastUpdated")},
                "id": pos_id, "dteRemaining": dte_remaining, "status": "expired",
                "lastUpdated": prior.get("lastUpdated"), "history": history,
            })
            continue

        print(f"  re-pricing tracked position {pos_id} ({pos['sym']} {pos.get('strat', '')})...")
        try:
            snap = _reprice_position(pos)
        except Exception as e:
            snap = None
            print(f"    failed: {e}")

        if snap is None:
            # Couldn't get a fresh quote this run — surface the last good
            # snapshot rather than a blank/broken row on the frontend.
            out_positions.append({
                "id": pos_id, "status": "error", "dteRemaining": dte_remaining,
                "lastUpdated": prior.get("lastUpdated"),
                "currentPremium": prior.get("currentPremium"),
                "pnlDollars": prior.get("pnlDollars"), "pnlPct": prior.get("pnlPct"),
                "history": history,
            })
            continue

        today_iso = today.isoformat()
        if not history or history[-1].get("date") != today_iso:
            history.append({
                "date": today_iso, "dte": dte_remaining,
                "premium": snap["currentPremium"],
                "pnlDollars": snap["pnlDollars"], "pnlPct": snap["pnlPct"],
            })
        out_positions.append({
            "id": pos_id, "status": "open", "dteRemaining": dte_remaining,
            "lastUpdated": datetime.now(timezone.utc).isoformat(),
            "currentPremium": snap["currentPremium"],
            "pnlDollars": snap["pnlDollars"], "pnlPct": snap["pnlPct"],
            "history": history,
        })

    output = {"generated_at": datetime.now(timezone.utc).isoformat(), "positions": out_positions}
    with open(TRACKED_PNL_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Wrote {len(out_positions)} tracked position(s) to {TRACKED_PNL_PATH}")


def main():
    trades = []
    equities = []
    for i, ticker in enumerate(TICKERS):
        print(f"Fetching {ticker}...")
        trade = build_trade_for_ticker(ticker, STRATEGY_INDEX.get(ticker, i))
        if trade:
            trades.append(trade)

        equity = build_equity_snapshot(ticker, ticker in KNOWN_ETFS)
        if equity:
            equities.append(equity)

    if not trades and not equities:
        print("No trades or equities were built — leaving existing data.json untouched.", file=sys.stderr)
        sys.exit(1)

    # --- Peer-relative valuation (added 2026-09-24) ---------------------
    # Second pass, pure in-memory — no additional yfinance calls. Sector
    # medians can't be known until every equity in the run has been built
    # (see compute_sector_valuations' docstring), so trades/equities above
    # were built with composite scores that hadn't seen valuation yet
    # (valuation_component defaults to 5/neutral in every scoring function).
    # This recomputes each score now that peer data is available, using
    # only fields already sitting on the built dicts — no re-fetching.
    valuations = compute_sector_valuations(equities)
    print(f"  valuation: {len(valuations)} of {len(equities)} equities got a peer comparison "
          f"(need >= {MIN_SECTOR_PEERS_FOR_VALUATION} same-sector tickers in this run)")

    for e in equities:
        v = valuations.get(e["sym"])
        if v:
            e.update(v)
        val_component = valuation_score_component(v, "bull")  # see equity_composite_score's
        e["score"] = equity_composite_score(e["trend"], e["rsi"], e["chg6m"], val_component)

    for t in trades:
        v = valuations.get(t["sym"])
        if v:
            t["valuationLabel"] = v["valuationLabel"]
            t["peVsSectorPct"] = v["peVsSectorPct"]
            t["sectorMedianPE"] = v["sectorMedianPE"]
            t["sectorPeerCount"] = v["sectorPeerCount"]
            t["sector"] = v["sector"]
        val_component = valuation_score_component(v, t["side"])
        if t["strat"] in ("Long Call", "Long Put"):
            t["score"] = composite_score_long_option(t["pot"], t["ivr"], t["breakevenMovePct"], val_component)
        elif t["strat"] == "Double Diagonal":
            pass  # always-neutral side, see composite_score_double_diagonal's docstring — score unchanged
        else:
            t["score"] = composite_score(t["ap"], t["pot"], t["ivr"], val_component)

    print(f"Fetching hedge candidate ({HEDGE_TICKER})...")
    hedge = build_hedge_candidate()
    if not hedge:
        # Not fatal — the frontend just hides the hedge section when this key
        # is absent, same as any other optional field in this file.
        print(f"  hedge candidate unavailable this run — omitting from {OUTPUT_PATH}")

    print("Fetching market snapshot (QQQ, SPY, IWM, VIX)...")
    market = build_market_snapshot()
    if not market:
        print("  market snapshot unavailable this run — the page hides the tiles")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance (unofficial, free, EOD)",
        "trades": trades,
        "equities": equities,
        "hedge": hedge,
        "market": market,
    }
    if UNIVERSE_META:
        output["universe"] = UNIVERSE_META

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(trades)} trades, {len(equities)} equity snapshots, "
          f"and {'a' if hedge else 'no'} hedge candidate to {OUTPUT_PATH}")

    # Best-effort — a bug or an unfetchable contract here must never take
    # down the core recommendation run above, which has already succeeded
    # and been written to disk by this point.
    try:
        update_tracked_positions()
    except Exception as e:
        print(f"  tracked positions: update failed, leaving existing {TRACKED_PNL_PATH} untouched: {e}")


if __name__ == "__main__":
    # `--ticker SYMBOL` runs the on-demand single-symbol lookup path instead
    # of the normal full-watchlist batch — see run_lookup() above. No flag
    # (the normal nightly/hourly invocation) behaves exactly as before.
    if "--ticker" in sys.argv:
        idx = sys.argv.index("--ticker")
        symbol_arg = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        run_lookup(symbol_arg)
    else:
        main()
