# ============================================================
# GATEAU PORTFOLIO SYSTEM v1.0
# Trading system inspired by quallamaggie, JLaw & Martin Luk
# Combines: Tightness Screener + Anticipation Setup + Area of Value
# ============================================================
# 
# Version: 1.0
# Date: May 24, 2026
# Status: Production Ready
#
# Install: pip install yfinance pandas numpy matplotlib curl_cffi tqdm

import io, re, time, random, warnings, hashlib, pickle, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import requests

warnings.filterwarnings("ignore")

# Pandas display optimization
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", "{:.4f}".format)

# =================== SYSTEM CONFIG ===================
CONFIG = {
    # Universe
    "min_price": 15.0,
    "min_avg_vol": 1_000_000,
    "min_atr_pct": 0.04,
    "max_atr_pct": 0.15,
    # Tightness (quallamaggie VCP)
    "tight_ratio": 0.8,                   # 3-day range <= 0.8 * ATR20 (compressed)
    "tight_lookback": "60d",
    # Anticipation setup (Stockbee/JLaw)
    "atr_period": 14,
    "tight_pct_threshold": 1.5,           # |Close-Open|/Open% <= 1.5% for tight day
    "min_tight_days": 2,
    "max_tight_days": 3,
    "vol_ratio_threshold": 0.8,
    "vol_avg_period": 50,
    "momentum_burst_pct": 8.0,
    "momentum_burst_window": 5,
    "momentum_lookback_max": 60,
    "momentum_lookback_min": 10,
    "sma_period": 50,
    "max_dist_from_52w_high_pct": 25,
    # Area of Value (Martin Luk)
    "lookback_days": 252,
    "swing_order": 5,
    "zone_cluster_pct": 0.01,
    "min_touches": 2,
    "min_tol_pct": 0.01,
    "max_tol_pct": 0.05,
    "require_above_200ma": True,
    "min_market_cap": 300_000_000,
    "min_beta": 1.5,
    # System
    "batch_size": 100,
    "max_workers": 4,
    "cache_dir": "/tmp/gateau_cache",
    "cache_ttl_minutes": 60,
}


# =================== UTILITY FUNCTIONS ===================
def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    Path(CONFIG["cache_dir"]).mkdir(parents=True, exist_ok=True)


def get_cache_path(ticker: str) -> str:
    """Generate cache file path for a ticker."""
    return f"{CONFIG['cache_dir']}/{ticker}.pkl"


def cache_is_fresh(ticker: str) -> bool:
    """Check if cached data is still fresh."""
    cache_path = get_cache_path(ticker)
    if not Path(cache_path).exists():
        return False
    age_minutes = (time.time() - Path(cache_path).stat().st_mtime) / 60
    return age_minutes < CONFIG["cache_ttl_minutes"]


def load_cached(ticker: str) -> pd.DataFrame:
    """Load cached OHLCV data."""
    try:
        with open(get_cache_path(ticker), "rb") as f:
            return pickle.load(f)
    except:
        return pd.DataFrame()


def save_cached(ticker: str, df: pd.DataFrame):
    """Save OHLCV data to cache."""
    ensure_cache_dir()
    try:
        with open(get_cache_path(ticker), "wb") as f:
            pickle.dump(df, f)
    except:
        pass


# =================== DATA FETCHING ===================
def fetch_ohlcv(ticker: str, period: str = "1y", force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV data with intelligent caching.
    Returns DataFrame with columns: Open, High, Low, Close, Volume
    """
    if not force_refresh and cache_is_fresh(ticker):
        return load_cached(ticker)
    
    try:
        data = yf.download(ticker, period=period, progress=False, threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
        data.columns = ["open", "high", "low", "close", "volume"]
        save_cached(ticker, data)
        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


# =================== TECHNICAL ANALYSIS ===================
def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    if len(data) < period:
        return pd.Series([0] * len(data), index=data.index)
    
    high = data["high"]
    low = data["low"]
    close = data["close"]
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    
    return tr.rolling(period).mean()


def compute_sma(data: pd.Series, period: int) -> pd.Series:
    """Compute Simple Moving Average."""
    return data.rolling(period).mean()


def compute_ema(data: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()


def find_swing_highs(data: pd.DataFrame, order: int = 5) -> list:
    """
    Identify swing highs (local maxima) in price data.
    order: number of bars on either side to check.
    """
    highs = []
    close = data["close"].values
    
    for i in range(order, len(close) - order):
        if close[i] == max(close[i - order : i + order + 1]):
            highs.append((i, close[i]))
    
    return highs


def find_swing_lows(data: pd.DataFrame, order: int = 5) -> list:
    """
    Identify swing lows (local minima) in price data.
    order: number of bars on either side to check.
    """
    lows = []
    close = data["close"].values
    
    for i in range(order, len(close) - order):
        if close[i] == min(close[i - order : i + order + 1]):
            lows.append((i, close[i]))
    
    return lows


# =================== SCREENING LOGIC ===================
def check_tightness(data: pd.DataFrame) -> bool:
    """
    quallamaggie VCP: Check if stock is in consolidation phase.
    Tight ratio = min_high / max_low over lookback period.
    """
    if len(data) < 20:
        return False
    
    recent = data.tail(20)
    min_high = recent["high"].min()
    max_low = recent["low"].max()
    
    if max_low == 0:
        return False
    
    tight_ratio = min_high / max_low
    return tight_ratio >= CONFIG["tight_ratio"]


def check_anticipation_setup(data: pd.DataFrame) -> dict:
    """
    Stockbee/JLaw Anticipation Setup:
    1. Stock in tight consolidation
    2. Rising volume before breakout
    3. Price close to breakout level
    4. Momentum setup ready
    """
    results = {
        "is_setup": False,
        "tightness_pct": 0.0,
        "vol_ratio": 0.0,
        "dist_from_52w_high": 0.0,
        "above_sma": False,
    }
    
    if len(data) < 60:
        return results
    
    recent = data.tail(20)
    atr = compute_atr(data, CONFIG["atr_period"]).iloc[-1]
    close = data["close"].iloc[-1]
    
    # Check tightness
    range_pct = (recent["high"].max() - recent["low"].min()) / close * 100
    if range_pct > CONFIG["tight_pct_threshold"]:
        return results
    results["tightness_pct"] = range_pct
    
    # Check volume ratio
    recent_vol = recent["volume"].tail(5).mean()
    avg_vol = data["volume"].tail(50).mean()
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 0
    results["vol_ratio"] = vol_ratio
    
    if vol_ratio < CONFIG["vol_ratio_threshold"]:
        return results
    
    # Check distance from 52-week high
    high_52w = data["high"].tail(252).max()
    dist_from_high = (1 - close / high_52w) * 100 if high_52w > 0 else 100
    results["dist_from_52w_high"] = dist_from_high
    
    if dist_from_high > CONFIG["max_dist_from_52w_high_pct"]:
        return results
    
    # Check above SMA
    sma = compute_sma(data["close"], CONFIG["sma_period"]).iloc[-1]
    results["above_sma"] = close > sma
    
    # All checks passed
    results["is_setup"] = True
    return results


def find_area_of_value(data: pd.DataFrame) -> dict:
    """
    Martin Luk Area of Value: Find support zones based on swing analysis.
    Returns zones with multiple price touches (support/resistance clusters).
    """
    results = {
        "has_value_area": False,
        "support_levels": [],
        "resistance_levels": [],
        "zone_strength": 0,
    }
    
    if len(data) < CONFIG["lookback_days"]:
        return results
    
    lookback = data.tail(CONFIG["lookback_days"])
    
    # Find swings
    lows = find_swing_lows(lookback, CONFIG["swing_order"])
    highs = find_swing_highs(lookback, CONFIG["swing_order"])
    
    if not lows and not highs:
        return results
    
    # Cluster lows into support zones
    support_prices = [low[1] for low in lows]
    support_zones = cluster_prices(support_prices, CONFIG["zone_cluster_pct"])
    
    # Cluster highs into resistance zones
    resistance_prices = [high[1] for high in highs]
    resistance_zones = cluster_prices(resistance_prices, CONFIG["zone_cluster_pct"])
    
    # Filter by touch count
    strong_support = [z for z in support_zones if z["touches"] >= CONFIG["min_touches"]]
    strong_resistance = [z for z in resistance_zones if z["touches"] >= CONFIG["min_touches"]]
    
    if strong_support or strong_resistance:
        results["has_value_area"] = True
        results["support_levels"] = [z["level"] for z in strong_support]
        results["resistance_levels"] = [z["level"] for z in strong_resistance]
        results["zone_strength"] = len(strong_support) + len(strong_resistance)
    
    return results


def cluster_prices(prices: list, tolerance_pct: float) -> list:
    """
    Cluster similar prices into zones.
    tolerance_pct: percentage tolerance for grouping (e.g., 0.01 = 1%).
    """
    if not prices:
        return []
    
    prices_sorted = sorted(prices)
    clusters = []
    current_cluster = [prices_sorted[0]]
    
    for price in prices_sorted[1:]:
        if abs(price - current_cluster[0]) / current_cluster[0] <= tolerance_pct:
            current_cluster.append(price)
        else:
            avg_price = np.mean(current_cluster)
            clusters.append({
                "level": avg_price,
                "touches": len(current_cluster),
                "prices": current_cluster,
            })
            current_cluster = [price]
    
    if current_cluster:
        avg_price = np.mean(current_cluster)
        clusters.append({
            "level": avg_price,
            "touches": len(current_cluster),
            "prices": current_cluster,
        })
    
    return clusters


# =================== UNIVERSE SCREENING ===================
def screen_ticker(ticker: str) -> dict:
    """
    Run full screening logic on a single ticker.
    Returns dict with setup scores and classifications.
    """
    result = {
        "ticker": ticker,
        "price": 0.0,
        "is_screened": False,
        "reason": "Unknown",
        "score": 0.0,
        "tightness_score": 0.0,
        "anticipation_score": 0.0,
        "value_area_score": 0.0,
    }
    
    # Fetch data
    data = fetch_ohlcv(ticker)
    if data.empty or len(data) < 60:
        result["reason"] = "Insufficient data"
        return result
    
    current_price = data["close"].iloc[-1]
    result["price"] = current_price
    
    # Filter: Minimum price
    if current_price < CONFIG["min_price"]:
        result["reason"] = "Price too low"
        return result
    
    # Filter: Average volume
    avg_vol = data["volume"].tail(20).mean()
    if avg_vol < CONFIG["min_avg_vol"]:
        result["reason"] = "Volume too low"
        return result
    
    # Filter: ATR range
    atr = compute_atr(data, CONFIG["atr_period"]).iloc[-1]
    atr_pct = atr / current_price * 100
    if atr_pct < CONFIG["min_atr_pct"] or atr_pct > CONFIG["max_atr_pct"]:
        result["reason"] = f"ATR out of range: {atr_pct:.2f}%"
        return result
    
    # Screening: Tightness
    is_tight = check_tightness(data)
    result["tightness_score"] = 25.0 if is_tight else 0.0
    
    # Screening: Anticipation Setup
    anticipation = check_anticipation_setup(data)
    result["anticipation_score"] = 35.0 if anticipation["is_setup"] else 0.0
    
    # Screening: Area of Value
    value_area = find_area_of_value(data)
    result["value_area_score"] = 40.0 if value_area["has_value_area"] else 0.0
    
    # Combined score (0-100)
    result["score"] = (
        result["tightness_score"] +
        result["anticipation_score"] +
        result["value_area_score"]
    )
    
    # Mark as screened if score >= threshold
    if result["score"] >= 60.0:
        result["is_screened"] = True
        result["reason"] = "GATEAU SETUP DETECTED"
    else:
        result["reason"] = f"Score too low: {result['score']:.1f}/100"
    
    return result


def screen_universe(tickers: list) -> pd.DataFrame:
    """
    Screen a list of tickers in parallel.
    Returns DataFrame with screening results sorted by score descending.
    """
    results = []
    
    print(f"Screening {len(tickers)} tickers...")
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = {executor.submit(screen_ticker, t): t for t in tickers}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                print(f"  [{completed}/{len(tickers)}] tickers processed...")
            results.append(future.result())
    
    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    
    return df


# =================== REPORTING ===================
def print_screened_results(results: pd.DataFrame):
    """Print formatted screening results."""
    screened = results[results["is_screened"]].head(20)
    
    if screened.empty:
        print("\n❌ No GATEAU setups found.")
        return
    
    print(f"\n✅ GATEAU SETUPS ({len(screened)} found):")
    print("-" * 120)
    print(
        f"{'Ticker':<10} {'Price':<10} {'Score':<8} "
        f"{'Tightness':<12} {'Anticipation':<14} {'Value Area':<12} {'Status':<30}"
    )
    print("-" * 120)
    
    for _, row in screened.iterrows():
        print(
            f"{row['ticker']:<10} ${row['price']:<9.2f} {row['score']:<7.1f} "
            f"{row['tightness_score']:<11.1f} {row['anticipation_score']:<13.1f} "
            f"{row['value_area_score']:<11.1f} {row['reason']:<30}"
        )
    
    print("-" * 120)


# =================== MAIN ===================
def main():
    """Run the GATEAU portfolio screener."""
    print("=" * 80)
    print("  GATEAU PORTFOLIO SYSTEM")
    print("  Trading system inspired by quallamaggie, JLaw & Martin Luk")
    print("  Tightness Screener + Anticipation Setup + Area of Value")
    print("=" * 80)
    
    # Sample universe (S&P 500 top 100 by market cap)
    universe = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "JNJ", "V",
        "UNH", "XOM", "WMT", "MA", "HD", "CVX", "LLY", "ABBV", "AVGO", "PG",
        "MRK", "COST", "KO", "PEP", "TMO", "MCD", "ACN", "NKE", "ADBE", "CRM",
        "BAC", "NFLX", "ORCL", "LIN", "PM", "TXN", "DHR", "AMD", "QCOM", "HON",
        "UNP", "INTU", "IBM", "RTX", "AMGN", "LOW", "CAT", "GS", "SBUX", "BLK",
        # Add more tickers as needed
    ]
    
    # Run screening
    results = screen_universe(universe)
    
    # Print results
    print_screened_results(results)
    
    # Save to CSV
    results.to_csv("gateau_screening_results.csv", index=False)
    print(f"\n📊 Full results saved to: gateau_screening_results.csv")
    
    return results


if __name__ == "__main__":
    results = main()
