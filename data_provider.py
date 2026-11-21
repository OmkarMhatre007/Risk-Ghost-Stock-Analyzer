"""
Online Market Data Provider for Risk Ghost.

Retrieves historical prices, volumes, and instrument metadata using yfinance online.
No runtime dependence on offline CSV fallbacks.
Handles connection failures, rate limits, currency identification, and quote subunits.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class MarketDataError(Exception):
    """Base exception for market data errors."""
    pass


class TickerNotFoundError(MarketDataError):
    """Raised when a ticker cannot be found or resolved."""
    pass


class UnsupportedInstrumentError(MarketDataError):
    """Raised when instrument is unsupported (e.g., indices, crypto, options)."""
    pass


class InsufficientDataError(MarketDataError):
    """Raised when historical observations are insufficient for covariance/risk analysis."""
    pass


class RateLimitError(MarketDataError):
    """Raised when provider rate limits requests."""
    pass


class FXConversionError(MarketDataError):
    """Raised when foreign exchange rates cannot be retrieved or aligned."""
    pass


@dataclass
class InstrumentMetadata:
    symbol: str
    name: str
    exchange: str
    quote_type: str
    currency: str
    is_subunit: bool = False
    subunit_factor: float = 1.0
    fetch_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def standard_currency(self) -> str:
        """Returns the standard currency code (e.g., GBP instead of GBp)."""
        if self.currency.upper() == "GBP" and self.is_subunit:
            return "GBP"
        if self.currency.upper() == "GBPEUR" or self.currency.upper() == "GBP":
            return "GBP"
        if self.currency.upper() == "GBX" or self.currency.upper() == "GBP":
            return "GBP"
        return self.currency.upper()


def search_company(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    """
    Search for companies or tickers using yfinance.Search.
    Returns a list of candidate dictionaries with symbol, name, exchange, type, and currency.
    Does not automatically select the first match.
    Filters out unsupported instruments like cryptos or indices for portfolio equity holding.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    try:
        search_obj = yf.Search(clean_query, max_results=max_results, enable_fuzzy_query=True)
        raw_quotes = getattr(search_obj, "quotes", [])
    except Exception as e:
        err_msg = str(e).lower()
        if "rate" in err_msg or "too many" in err_msg:
            raise RateLimitError(f"Yahoo Finance rate limit reached while searching for '{clean_query}'.") from e
        raise MarketDataError(f"Search request failed for '{clean_query}': {e}") from e

    results: List[Dict[str, str]] = []
    supported_types = {"EQUITY", "ETF", "MUTUALFUND"}

    for q in raw_quotes:
        symbol = q.get("symbol", "").strip()
        if not symbol:
            continue
        quote_type = q.get("quoteType", "").upper()
        # Allow user to see results, but note the quote type
        name = q.get("shortname") or q.get("longname") or symbol
        exchange = q.get("exchange", "")
        # Search API may or may not return currency directly; record if present
        currency = q.get("currency", "")

        results.append({
            "symbol": symbol,
            "name": name,
            "exchange": exchange,
            "quote_type": quote_type,
            "currency": currency,
            "is_supported": quote_type in supported_types,
        })

    return results


def get_ticker_metadata(symbol: str, retries: int = 2) -> InstrumentMetadata:
    """
    Fetch instrument metadata online for a specific ticker symbol.
    Identifies trading currency and handles pence/GBP subunit conventions.
    Rejects unsupported instruments such as indices as purchasable holdings.
    """
    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("Symbol cannot be empty.")

    # Reject obvious index symbols upfront if attempted as holdings
    if clean_symbol.startswith("^"):
        raise UnsupportedInstrumentError(
            f"Symbol '{clean_symbol}' appears to be a market index. Indices cannot be purchased as direct stock holdings."
        )

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            ticker_obj = yf.Ticker(clean_symbol)
            fast_info = getattr(ticker_obj, "fast_info", None)
            info = {}
            # Try fast_info first (faster, less overhead)
            currency = ""
            exchange = ""
            quote_type = ""
            name = clean_symbol

            if fast_info:
                try:
                    currency = getattr(fast_info, "currency", "") or ""
                    exchange = getattr(fast_info, "exchange", "") or ""
                    quote_type = getattr(fast_info, "quote_type", "") or ""
                except Exception:
                    pass

            if not currency or not quote_type:
                try:
                    info = ticker_obj.info or {}
                    currency = currency or info.get("currency", "")
                    exchange = exchange or info.get("exchange", "")
                    quote_type = quote_type or info.get("quoteType", "")
                    name = info.get("shortName") or info.get("longName") or clean_symbol
                except Exception:
                    pass

            if not currency:
                # If still empty, attempt history probe to verify symbol exists
                hist = ticker_obj.history(period="5d")
                if hist.empty:
                    raise TickerNotFoundError(f"Symbol '{clean_symbol}' not found or lacks available market data.")
                currency = "USD"  # fallback if provider omits currency

            quote_type_upper = quote_type.upper()
            if quote_type_upper == "INDEX":
                raise UnsupportedInstrumentError(
                    f"'{clean_symbol}' is a market index and cannot be treated as a stock holding."
                )

            # Detect London Stock Exchange pence quote conventions
            is_subunit = False
            subunit_factor = 1.0
            currency_upper = currency.upper()

            if currency_upper in ("GBX", "GBP") or (exchange in ("LSE", "LON") and currency_upper == "GBP"):
                # Frequently LSE equities are quoted in pence (GBp / GBX) where 100 pence = 1 GBP
                if currency_upper in ("GBX", "GBP") or currency == "GBp":
                    is_subunit = True
                    subunit_factor = 100.0
                    standard_curr = "GBP"
                else:
                    standard_curr = currency_upper
            else:
                standard_curr = currency_upper

            return InstrumentMetadata(
                symbol=clean_symbol,
                name=name,
                exchange=exchange,
                quote_type=quote_type_upper or "EQUITY",
                currency=standard_curr,
                is_subunit=is_subunit,
                subunit_factor=subunit_factor,
            )
        except (TickerNotFoundError, UnsupportedInstrumentError):
            raise
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "rate" in err_str or "too many" in err_str:
                if attempt < retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise RateLimitError(f"Rate limit exceeded while querying '{clean_symbol}'.") from e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue

    raise MarketDataError(f"Failed to retrieve metadata for '{clean_symbol}': {last_err}")


def fetch_historical_prices_and_volumes(
    tickers: List[str],
    period: str = "60d",
    interval: str = "1d",
    min_observations: int = 15,
    retries: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, InstrumentMetadata], str]:
    """
    Downloads historical close prices and trading volumes online using yfinance.
    Performs bounded retries on transient connection failures.
    Returns:
        close_df: DataFrame of closing prices (in instrument native trading units, corrected for subunits)
        volume_df: DataFrame of share trading volumes
        metadata_map: Dict mapping ticker to InstrumentMetadata
        fetch_timestamp: UTC ISO timestamp of retrieval
    Raises:
        MarketDataError, TickerNotFoundError, InsufficientDataError, RateLimitError
    """
    if not tickers:
        raise ValueError("Tickers list cannot be empty.")

    clean_tickers = [t.strip().upper() for t in tickers]
    metadata_map: Dict[str, InstrumentMetadata] = {}

    for t in clean_tickers:
        meta = get_ticker_metadata(t)
        metadata_map[t] = meta

    fetch_time = datetime.now(timezone.utc).isoformat()
    raw_df: Optional[pd.DataFrame] = None
    last_err: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            # yfinance download
            raw_df = yf.download(
                tickers=clean_tickers,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                timeout=10,
            )
            if raw_df is not None and not raw_df.empty:
                break
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "rate" in err_str or "too many" in err_str:
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RateLimitError(f"Rate limited while downloading historical data: {e}") from e
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue

    if raw_df is None or raw_df.empty:
        raise MarketDataError(
            f"Failed to fetch market data online for {clean_tickers}. Error: {last_err or 'Empty response from provider'}"
        )

    # Extract Close and Volume series
    close_data: Dict[str, pd.Series] = {}
    volume_data: Dict[str, pd.Series] = {}

    # Multi-ticker downloads have MultiIndex columns (Price, Ticker) or (Ticker, Price)
    if isinstance(raw_df.columns, pd.MultiIndex):
        level_0 = raw_df.columns.get_level_values(0)
        if "Close" in level_0:
            closes_sub = raw_df["Close"]
            vols_sub = raw_df["Volume"] if "Volume" in level_0 else pd.DataFrame()
            for t in clean_tickers:
                if t in closes_sub.columns:
                    close_data[t] = closes_sub[t]
                    volume_data[t] = vols_sub[t] if t in vols_sub.columns else pd.Series(dtype=float)
        else:
            # Level 1 might be Close
            for t in clean_tickers:
                try:
                    c = raw_df.xs(key=(t, "Close"), axis=1, drop_level=True)
                    v = raw_df.xs(key=(t, "Volume"), axis=1, drop_level=True)
                    close_data[t] = c
                    volume_data[t] = v
                except Exception:
                    pass
    else:
        # Single ticker case
        t = clean_tickers[0]
        if "Close" in raw_df.columns:
            close_data[t] = raw_df["Close"]
        if "Volume" in raw_df.columns:
            volume_data[t] = raw_df["Volume"]

    # Verify every requested ticker has returned data
    missing_tickers = [t for t in clean_tickers if t not in close_data or close_data[t].dropna().empty]
    if missing_tickers:
        raise TickerNotFoundError(
            f"No usable historical data returned for: {', '.join(missing_tickers)}. Check ticker symbols."
        )

    close_df = pd.DataFrame(close_data)
    volume_df = pd.DataFrame(volume_data)

    # Forward fill bounded to 2 trading days to account for holiday differences, then drop unaligned edges
    close_df = close_df.ffill(limit=2).dropna()
    volume_df = volume_df.reindex(close_df.index).ffill(limit=2).fillna(0.0)

    if len(close_df) < min_observations:
        raise InsufficientDataError(
            f"Insufficient aligned trading observations ({len(close_df)} found, minimum {min_observations} required). "
            f"Check if tickers trade on conflicting market schedules or have insufficient history."
        )

    # Correct subunit pricing (e.g. GBp pence -> GBP pounds)
    for t in clean_tickers:
        meta = metadata_map[t]
        if meta.is_subunit and meta.subunit_factor > 1.0:
            close_df[t] = close_df[t] / meta.subunit_factor

    return close_df, volume_df, metadata_map, fetch_time


def fetch_fx_history(
    from_currency: str,
    to_currency: str,
    period: str = "60d",
    target_index: Optional[pd.DatetimeIndex] = None,
    retries: int = 2,
) -> pd.Series:
    """
    Retrieves historical FX exchange rates online to convert from_currency -> to_currency.
    Quote Direction:
      1 unit of from_currency = rate units of to_currency.
      Converted_Price = Original_Price * rate.
    Handles reciprocal lookup if the inverse pair is standard on Yahoo Finance (e.g. USD/INR).
    Aligns dates against target_index using bounded forward-fill.
    """
    from_curr = from_currency.strip().upper()
    to_curr = to_currency.strip().upper()

    if from_curr == to_curr:
        if target_index is not None:
            return pd.Series(1.0, index=target_index)
        return pd.Series([1.0])

    # Direct pair symbol, e.g., EURUSD=X (1 EUR = X USD)
    direct_symbol = f"{from_curr}{to_curr}=X"
    inverse_symbol = f"{to_curr}{from_curr}=X"

    fx_series: Optional[pd.Series] = None
    last_err: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            # Try direct pair first
            df = yf.download(direct_symbol, period=period, progress=False, auto_adjust=True, timeout=10)
            if df is not None and not df.empty and "Close" in df:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna()
                if len(close) >= 5:
                    fx_series = close
                    break

            # Try inverse pair
            df_inv = yf.download(inverse_symbol, period=period, progress=False, auto_adjust=True, timeout=10)
            if df_inv is not None and not df_inv.empty and "Close" in df_inv:
                close_inv = df_inv["Close"]
                if isinstance(close_inv, pd.DataFrame):
                    close_inv = close_inv.iloc[:, 0]
                close_inv = close_inv.dropna()
                if len(close_inv) >= 5:
                    # Invert rate: 1 from_currency = 1 / rate to_currency
                    fx_series = 1.0 / close_inv
                    break
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue

    if fx_series is None or fx_series.empty:
        raise FXConversionError(
            f"Unable to retrieve online FX rates for {from_curr} -> {to_curr} (tried {direct_symbol} & {inverse_symbol}). "
            f"Details: {last_err or 'No data returned.'}"
        )

    # Normalize timezone if needed
    if target_index is not None:
        if hasattr(fx_series.index, "tz") and fx_series.index.tz is not None:
            fx_series.index = fx_series.index.tz_localize(None)
        if hasattr(target_index, "tz") and target_index.tz is not None:
            norm_target = target_index.tz_localize(None)
        else:
            norm_target = target_index

        fx_series = fx_series.reindex(norm_target).ffill(limit=5).bfill(limit=5)
        if fx_series.isna().any():
            raise FXConversionError(
                f"Missing FX rate observations when aligning {from_curr}/{to_curr} to trading calendar."
            )

    return fx_series
