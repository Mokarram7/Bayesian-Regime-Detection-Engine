import pandas as pd

from src.logger import logger
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class Preprocessor:

    def __init__(self):

        self.input_file = RAW_DATA_DIR / "nifty50.csv"

        self.output_file = PROCESSED_DATA_DIR / "nifty50_clean.csv"

    def process(self):

        logger.info("Loading Raw Dataset...")

        df = pd.read_csv(self.input_file)

        logger.info(f"Original Shape : {df.shape}")

        df.drop_duplicates(inplace=True)

        df.dropna(inplace=True)

        df["Date"] = pd.to_datetime(df["Date"])

        df.sort_values("Date", inplace=True)

        df.reset_index(drop=True, inplace=True)

        df.to_csv(self.output_file, index=False)

        logger.success(f"Saved : {self.output_file}")

        logger.info(f"Processed Shape : {df.shape}")

        return df