import numpy as np
import pandas as pd
import ta

from src.logger import logger
from src.config import PROCESSED_DATA_DIR


class FeatureEngineer:

    def __init__(self):

        self.input_file = PROCESSED_DATA_DIR / "nifty50_clean.csv"

        self.output_file = PROCESSED_DATA_DIR / "feature_data.csv"

    def process(self):

        logger.info("Loading Clean Dataset...")

        df = pd.read_csv(self.input_file)

        df["Date"] = pd.to_datetime(df["Date"])

        logger.info("Creating Price Features...")

        df["Return"] = df["Close"].pct_change()

        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

        df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

        logger.info("Creating Trend Features...")

        df["SMA20"] = df["Close"].rolling(20).mean()

        df["SMA50"] = df["Close"].rolling(50).mean()

        df["EMA20"] = df["Close"].ewm(span=20).mean()

        df["EMA50"] = df["Close"].ewm(span=50).mean()

        df["Price_SMA20"] = df["Close"] / df["SMA20"]

        logger.info("Creating Volatility Features...")

        df["Volatility"] = df["Return"].rolling(20).std()

        bb = ta.volatility.BollingerBands(df["Close"])

        df["BB_UPPER"] = bb.bollinger_hband()

        df["BB_LOWER"] = bb.bollinger_lband()

        atr = ta.volatility.AverageTrueRange(

            high=df["High"],
            low=df["Low"],
            close=df["Close"]

        )

        df["ATR"] = atr.average_true_range()

        logger.info("Creating Market State Features...")

        rsi = ta.momentum.RSIIndicator(df["Close"])

        df["RSI"] = rsi.rsi()

        macd = ta.trend.MACD(df["Close"])

        df["MACD"] = macd.macd()

        df["MACD_SIGNAL"] = macd.macd_signal()

        df["MACD_HIST"] = macd.macd_diff()

        df.dropna(inplace=True)

        df.to_csv(self.output_file, index=False)

        logger.success(f"Saved : {self.output_file}")

        logger.info(f"Final Shape : {df.shape}")

        return df