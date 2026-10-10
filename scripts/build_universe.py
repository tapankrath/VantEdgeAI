"""
VantEdgeAI universe builder.

Screens Yahoo Finance (via yfinance's built-in screener) for large US-listed
stocks and writes the top slice of them to universe.json at the repo root.
scripts/fetch_data.py reads that file on every run and screens those names in
addition to the hand-pinned list in tickers.json.

Why this is a separate, slower-cadence job: market cap moves slowly, so the
universe only needs refreshing once a day — the hourly data run just reads the
cached file and never pays for the screen itself.

Selection rule (edit the constants below):
  1. US-listed (NASDAQ / NYSE), common stock, market cap above MIN_MARKET_CAP.
  2. Ranked by average daily dollar volume (price x 3-month average volume) —
     a proxy for how tradeable the options are — with market cap as tiebreak.
     Market cap alone would just give you the same 50 mega-caps forever; it's
     the size filter, not the ranking.
  3. The top UNIVERSE_SIZE survive. Two share classes of one company (GOOG /
     GOOGL, BRK-A / BRK-B) only take one slot.

If the screen fails or looks incomplete, the existing universe.json is left
untouched and this script exits non-zero, so a bad day never shrinks the list.

Added 2026-10-10:
  4. Options-liquidity check. Stock dollar volume is only a proxy, so every
     name is also checked on its options: the expiration nearest ~30 days out
     needs MIN_EXPIRY_OI total open interest (calls + puts). Names that fail
     are dropped and the next one in the ranking takes the slot.
  5. Momentum picks. Up to MOMENTUM_PICKS extra names from a wider pool
     (market cap above MOMENTUM_MIN_MARKET_CAP) whose price, 50-day and
     200-day averages are stacked in one direction, strongest trend first —
     so this week's momentum leaders (up OR down) get screened without
     anyone hand-editing tickers.json. They must pass the same options check.
  6. Pinned-list health. Every name in tickers.json gets the same options
     check, written to universe.json as "pinnedReport" and printed in the
     workflow log, so a pinned name whose options have dried up is easy to
     spot and remove.
  If Yahoo rate-limits the options checks (most of them erroring rather than
  failing), the check is skipped for that run and the plain ranking is used,
  so a bad day never empties the list.

Like the rest of this repo, this rides on Yahoo's unofficial endpoints: free,
no API key, and liable to change or rate-limit without warning.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone, date

try:
    import yfinance as yf
    from yfinance import EquityQuery
except ImportError:
    print("yfinance (>=1.0) not installed — run: pip install -r scripts/requirements.txt", file=sys.stderr)
    raise

# --- Configuration -----------------------------------------------------------
MIN_MARKET_CAP = 100_000_000_000   # $100B
UNIVERSE_SIZE = 50                # how many dynamic names to keep
EXCHANGES = ("NMS", "NYQ")        # Yahoo codes for NASDAQ and NYSE
PAGE_SIZE = 250                   # Yahoo's screener page cap
MAX_PAGES = 4                     # safety stop; >$100B is only a few hundred names
MIN_ACCEPTABLE_RESULTS = 20       # fewer than this = treat the screen as broken
OUTPUT_PATH = "universe.json"
PINNED_PATH = "tickers.json"

# Options-liquidity check (2026-10-10)
MIN_EXPIRY_OI = 20_000            # calls + puts OI on the expiration nearest ~30 days out
TARGET_DTE, MIN_DTE, MAX_DTE = 30, 14, 60
MAX_OPTION_CHECKS = 120           # cap on chain fetches per run (rate-limit safety)
CHECK_PAUSE_SEC = 0.35

# Momentum picks (2026-10-10)
MOMENTUM_MIN_MARKET_CAP = 20_000_000_000   # $20B
MOMENTUM_PICKS = 10
MOMENTUM_MIN_STRENGTH = 0.10      # price at least 10% away from its 200-day average

# Same company, multiple listed share classes — keep only the more liquid one.
SHARE_CLASS_GROUPS = [
    {"GOOG", "GOOGL"},
    {"BRK-A", "BRK-B"},
    {"NWS", "NWSA"},
    {"FOX", "FOXA"},
]

SYMBOL_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")


def fetch_quotes(min_cap=MIN_MARKET_CAP):
    """Pages through the screener, largest market cap first."""
    query = EquityQuery("and", [
        EquityQuery("gt", ["intradaymarketcap", min_cap]),
        EquityQuery("is-in", ["exchange", *EXCHANGES]),
    ])
    quotes = []
    for page in range(MAX_PAGES):
        result = yf.screen(
            query,
            offset=page * PAGE_SIZE,
            size=PAGE_SIZE,
            sortField="intradaymarketcap",
            sortAsc=False,
        )
        batch = (result or {}).get("quotes") or []
        quotes.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
    return quotes


def _num(v):
    try:
        f = float(v)
        return f if f == f and f > 0 else None  # drops NaN, zero and negatives
    except (TypeError, ValueError):
        return None


def clean(quotes, min_cap=MIN_MARKET_CAP):
    """Keeps plain US common stocks with a usable symbol and market cap."""
    rows, seen = [], set()
    for q in quotes:
        sym = (q.get("symbol") or "").strip().upper()
        if not SYMBOL_RE.match(sym) or sym in seen:
            continue
        qtype = (q.get("quoteType") or "EQUITY").upper()
        if qtype != "EQUITY":
            continue
        cap = _num(q.get("marketCap"))
        if cap is None or cap < min_cap:
            continue  # re-check client-side; the server filter uses intraday cap
        price = _num(q.get("regularMarketPrice"))
        avg_vol = _num(q.get("averageDailyVolume3Month"))
        dollar_vol = price * avg_vol if price and avg_vol else None
        seen.add(sym)
        rows.append({
            "symbol": sym,
            "name": q.get("shortName") or q.get("longName") or sym,
            "marketCap": cap,
            "avgDollarVolume": dollar_vol,
            "price": price,
            "ma50": _num(q.get("fiftyDayAverage")),
            "ma200": _num(q.get("twoHundredDayAverage")),
        })
    return rows


def rank(rows):
    """Dollar volume first (rows without it sort last), market cap as tiebreak."""
    return sorted(
        rows,
        key=lambda r: (r["avgDollarVolume"] is not None, r["avgDollarVolume"] or 0, r["marketCap"]),
        reverse=True,
    )


def drop_duplicate_share_classes(ranked):
    """`ranked` is best-first, so the first member of a group we meet is the keeper."""
    drop = set()
    for group in SHARE_CLASS_GROUPS:
        members = [r["symbol"] for r in ranked if r["symbol"] in group]
        drop.update(members[1:])
    return [r for r in ranked if r["symbol"] not in drop]


def options_check(sym):
    """('ok' | 'thin' | 'none' | 'error', total_oi, expiration) for the listed
    expiration nearest TARGET_DTE days out (within MIN_DTE..MAX_DTE)."""
    try:
        tk = yf.Ticker(sym)
        exps = tk.options or ()
        today = date.today()
        best = None
        for e in exps:
            try:
                d = datetime.strptime(e, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (d - today).days
            if MIN_DTE <= dte <= MAX_DTE and (best is None or abs(dte - TARGET_DTE) < abs(best[1] - TARGET_DTE)):
                best = (e, dte)
        if not best:
            return ("none", 0, None)
        chain = tk.option_chain(best[0])
        oi = int(chain.calls["openInterest"].fillna(0).sum() + chain.puts["openInterest"].fillna(0).sum())
        return ("ok" if oi >= MIN_EXPIRY_OI else "thin", oi, best[0])
    except Exception as e:
        return ("error", 0, f"{type(e).__name__}")
    finally:
        time.sleep(CHECK_PAUSE_SEC)


class OptionsGate:
    """Runs options_check with a per-run budget, and turns itself off if most
    checks error out (Yahoo rate-limiting), so the ranking still goes through."""
    def __init__(self):
        self.calls = self.errors = 0
        self.disabled = False
        self.cache = {}

    def check(self, sym):
        if sym in self.cache:
            return self.cache[sym]
        if self.disabled or self.calls >= MAX_OPTION_CHECKS:
            res = ("unchecked", 0, None)
        else:
            self.calls += 1
            res = options_check(sym)
            if res[0] == "error":
                self.errors += 1
                if self.calls >= 10 and self.errors / self.calls > 0.5:
                    print(f"Options checks failing ({self.errors}/{self.calls} errors) — "
                          f"skipping the options gate for this run.", file=sys.stderr)
                    self.disabled = True
        self.cache[sym] = res
        return res

    def passes(self, sym):
        status = self.check(sym)[0]
        # 'error' / 'unchecked' = couldn't tell -> keep (never empty the list on a bad day)
        return status in ("ok", "error", "unchecked")


def momentum_candidates(rows, exclude):
    """Names whose price, 50-day and 200-day averages are stacked one way,
    strongest (furthest from the 200-day) first."""
    picks = []
    for r in rows:
        if r["symbol"] in exclude:
            continue
        p, m50, m200 = r.get("price"), r.get("ma50"), r.get("ma200")
        if not (p and m50 and m200):
            continue
        strength = p / m200 - 1
        up = p > m50 > m200
        down = p < m50 < m200
        if (up or down) and abs(strength) >= MOMENTUM_MIN_STRENGTH:
            picks.append(dict(r, momentum="up" if up else "down", strength=round(strength * 100, 1)))
    picks.sort(key=lambda r: abs(r["strength"]), reverse=True)
    return picks


def load_pinned():
    try:
        with open(PINNED_PATH) as f:
            return [t.strip().upper() for t in (json.load(f).get("tickers") or []) if t.strip()]
    except Exception:
        return []


def main():
    try:
        quotes = fetch_quotes()
    except Exception as e:
        print(f"Screener call failed ({type(e).__name__}: {e}) — leaving existing {OUTPUT_PATH} untouched.", file=sys.stderr)
        sys.exit(1)

    rows = clean(quotes)
    print(f"Screener returned {len(quotes)} quotes; {len(rows)} usable above ${MIN_MARKET_CAP/1e9:.0f}B.")
    if len(rows) < MIN_ACCEPTABLE_RESULTS:
        print(f"Only {len(rows)} usable results (< {MIN_ACCEPTABLE_RESULTS}) — screen looks broken or partial; "
              f"leaving existing {OUTPUT_PATH} untouched.", file=sys.stderr)
        sys.exit(1)

    gate = OptionsGate()
    ranked = drop_duplicate_share_classes(rank(rows))
    top, dropped = [], []
    for r in ranked:
        if len(top) >= UNIVERSE_SIZE:
            break
        if gate.passes(r["symbol"]):
            top.append(r)
        else:
            dropped.append(r["symbol"])
    if dropped:
        print(f"Dropped for thin options: {', '.join(dropped)}")

    # Momentum picks from a wider pool (best effort — never fails the run)
    momentum = []
    try:
        wide = clean(fetch_quotes(MOMENTUM_MIN_MARKET_CAP), MOMENTUM_MIN_MARKET_CAP)
        for r in momentum_candidates(drop_duplicate_share_classes(wide), {x["symbol"] for x in top}):
            if len(momentum) >= MOMENTUM_PICKS:
                break
            if gate.check(r["symbol"])[0] == "ok":
                momentum.append(r)
        labels = ["%s (%+.0f%% vs 200d)" % (r["symbol"], r["strength"]) for r in momentum]
        print("Momentum picks: " + (", ".join(labels) or "none"))
    except Exception as e:
        print(f"Momentum picks skipped ({type(e).__name__}: {e})", file=sys.stderr)

    # Pinned-list health report
    pinned_report = []
    for sym in load_pinned():
        status, oi, exp = gate.check(sym)
        pinned_report.append({"symbol": sym, "options": status, "expiryOI": oi, "expiration": exp})
    weak = [p["symbol"] for p in pinned_report if p["options"] in ("thin", "none")]
    if weak:
        print(f"Pinned names with thin/no options (consider removing from tickers.json): {', '.join(weak)}")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance screener (unofficial, free)",
        "criteria": {
            "min_market_cap": MIN_MARKET_CAP,
            "size": UNIVERSE_SIZE,
            "exchanges": list(EXCHANGES),
            "ranked_by": "avg daily dollar volume (3-month)",
            "candidates_above_min_cap": len(rows),
            "min_expiry_open_interest": MIN_EXPIRY_OI,
            "options_gate": "skipped (rate-limited)" if gate.disabled else "applied",
            "momentum_pool_min_market_cap": MOMENTUM_MIN_MARKET_CAP,
        },
        "tickers": [r["symbol"] for r in top] + [r["symbol"] for r in momentum],
        "details": top,
        "momentum": momentum,
        "droppedThinOptions": dropped,
        "pinnedReport": pinned_report,
    }

    tmp_path = OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp_path, OUTPUT_PATH)  # atomic — a crash mid-write can't leave a half-file

    print(f"Wrote {len(output['tickers'])} tickers ({len(top)} core + {len(momentum)} momentum) to {OUTPUT_PATH}:")
    print("  " + ", ".join(output["tickers"]))


if __name__ == "__main__":
    main()
