"""
ДЗ 7: Финальный ансамбль с RL
Загружает данные, добавляет технические индикаторы и разбивает на train/test
"""

import itertools
import os
import warnings
from config import (
    OUTPUT_DIR,
    TRAIN_START,
    TRAIN_END,
    TRADE_START,
    TRADE_END,
    DOW_30_TICKERS,
    INDICATORS,
)

warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import yfinance as yf
import talib
from finrl.meta.preprocessor.preprocessors import data_split


def download_stock_data(tickers, start, end):
    """Download stock data using yfinance"""
    all_data = []
    for ticker in tickers:
        logger.info(f"Downloading {ticker}...")
        try:
            df = yf.download(
                ticker, start=start, end=end, progress=False, auto_adjust=True
            )
            if len(df) > 0:
                df = df.reset_index()
                df.columns = [
                    col[0].lower() if isinstance(col, tuple) else col.lower()
                    for col in df.columns
                ]
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df["tic"] = ticker
                df = df[["date", "tic", "open", "high", "low", "close", "volume"]]
                all_data.append(df)
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")

    if not all_data:
        raise ValueError("No data downloaded")

    return pd.concat(all_data, ignore_index=True)


def add_talib_indicators(df):
    """Add technical indicators using TA-Lib"""
    logger.info("Adding TA-Lib indicators...")
    tickers = df["tic"].unique()
    all_results = []

    for tic in tickers:
        tic_data = df[df["tic"] == tic].copy()
        tic_data = tic_data.sort_values("date").reset_index(drop=True)

        open_p = tic_data["open"].values
        high = tic_data["high"].values
        low = tic_data["low"].values
        close = tic_data["close"].values
        volume = tic_data["volume"].values.astype(float)

        rsi = talib.RSI(close, timeperiod=14)
        tic_data["rsi_14"] = rsi

        macd, macds, macdh = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        tic_data["macd"] = macd
        tic_data["macds"] = macds
        tic_data["macdh"] = macdh

        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        tic_data["bb_upper"] = bb_upper
        tic_data["bb_middle"] = bb_middle
        tic_data["bb_lower"] = bb_lower

        adx = talib.ADX(high, low, close, timeperiod=14)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
        tic_data["adx"] = adx
        tic_data["plus_di"] = plus_di
        tic_data["minus_di"] = minus_di

        cci = talib.CCI(high, low, close, timeperiod=20)
        tic_data["cci"] = cci

        mom = talib.MOM(close, timeperiod=10)
        tic_data["mom_10"] = mom

        roc = talib.ROC(close, timeperiod=10)
        tic_data["roc_10"] = roc

        tic_data["obv"] = talib.OBV(close, volume)

        for window in [5, 10, 20, 30, 60]:
            tic_data[f"sma_{window}"] = talib.SMA(close, timeperiod=window)

        tic_data["volume"] = volume
        all_results.append(tic_data)

    return pd.concat(all_results, ignore_index=True)


def add_vix(df):
    """Add VIX data"""
    logger.info("Downloading VIX...")
    try:
        vix = yf.download(
            "^VIX",
            start="2010-01-01",
            end="2026-01-01",
            progress=False,
            auto_adjust=True,
        )
        if len(vix) > 0:
            vix = vix.reset_index()
            vix.columns = [
                col[0].lower() if isinstance(col, tuple) else col.lower()
                for col in vix.columns
            ]
            vix["date"] = pd.to_datetime(vix["date"]).dt.strftime("%Y-%m-%d")
            vix = vix.rename(columns={"close": "vix"})[["date", "vix"]]
            df = df.merge(vix, on="date", how="left")
            df["vix"] = df["vix"].ffill().fillna(0)
        else:
            df["vix"] = 0
    except Exception as e:
        logger.warning(f"VIX error: {e}")
        df["vix"] = 0
    return df


def calculate_turbulence(df, lookback=20):
    """Calculate turbulence index"""
    logger.info(f"Calculating turbulence (lookback={lookback})...")
    df = df.copy()
    df["turbulence"] = 0.0
    for tic in df["tic"].unique():
        mask = df["tic"] == tic
        returns = df.loc[mask, "close"].pct_change()
        rolling_std = returns.rolling(window=lookback).std()
        df.loc[mask, "turbulence"] = rolling_std.fillna(0) * 100
    return df


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ДЗ 7: Финальный ансамбль - Загрузка и обработка данных")
    logger.info(f"Загрузка DOW 30 ({len(DOW_30_TICKERS)} тикеров)")
    logger.info(f"Период: {TRAIN_START} - {TRADE_END}")

    df_raw = download_stock_data(DOW_30_TICKERS, TRAIN_START, TRADE_END)
    logger.info(f"Загружено: {len(df_raw)} строк")

    logger.info("Добавление индикаторов (TA-Lib)")
    processed = add_talib_indicators(df_raw)

    logger.info("Добавление VIX")
    processed = add_vix(processed)

    logger.info("Расчет турбулентности")
    processed = calculate_turbulence(processed)

    logger.info("Финальная обработка")
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"]).ffill().fillna(0)

    logger.info(f"После fillna: {len(processed_full)} строк")

    logger.info("Разбиение на train/trade")
    train = data_split(processed_full, TRAIN_START, TRAIN_END)
    trade = data_split(processed_full, TRADE_START, TRADE_END)

    logger.info(f"Train: {len(train)} строк ({TRAIN_START} - {TRAIN_END})")
    logger.info(f"Trade: {len(trade)} строк ({TRADE_START} - {TRADE_END})")

    train.to_csv(os.path.join(OUTPUT_DIR, "train_data.csv"), index=False)
    trade.to_csv(os.path.join(OUTPUT_DIR, "trade_data.csv"), index=False)

    logger.info("Данные сохранены: train_data.csv, trade_data.csv")
    logger.info("Этап 1 завершен!")
