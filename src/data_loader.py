import yfinance as yf
import pandas as pd

from src.logger import logger
from src.config import RAW_DATA_DIR


class DataLoader:

    def __init__(self):

        self.symbol = "^NSEI"

        self.start = "2015-01-01"

        self.end = "2026-01-01"

    def download(self):

        logger.info("Downloading NIFTY 50 Data...")

        df = yf.download(

            self.symbol,

            start=self.start,

            end=self.end,

            progress=True,

            auto_adjust=False,

            group_by="column"

        )

        df = df.reset_index()

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = [

                c[0] if c[0] else c[1]

                for c in df.columns

            ]

        output = RAW_DATA_DIR / "nifty50.csv"

        df.to_csv(output, index=False)

        logger.success(f"Saved : {output}")

        return df