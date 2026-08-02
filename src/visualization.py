import matplotlib.pyplot as plt
import pandas as pd

from src.logger import logger
from src.config import (
    PROCESSED_DATA_DIR,
    FIGURES_DIR
)


class Visualizer:

    """
    Visualization Module
    """

    def __init__(self):

        self.input_file = (
            PROCESSED_DATA_DIR /
            "conformal_predictions.csv"
        )

    ############################################################
    # Load Dataset
    ############################################################

    def load_dataset(self):

        logger.info(
            "Loading Dataset..."
        )

        df = pd.read_csv(
            self.input_file
        )

        logger.success(
            f"Dataset Loaded : {df.shape}"
        )

        return df

    ############################################################
    # Regime Plot
    ############################################################

    def regime_plot(
        self,
        df
    ):

        logger.info(
            "Creating Regime Plot..."
        )

        plt.figure(figsize=(15,6))

        plt.plot(
            df["Close"],
            linewidth=1
        )

        plt.scatter(

            range(len(df)),

            df["Close"],

            c=df["Regime"],

            cmap="viridis",

            s=10

        )

        plt.title(
            "Hidden Market Regimes"
        )

        plt.xlabel("Time")

        plt.ylabel("Close Price")

        plt.tight_layout()

        plt.savefig(

            FIGURES_DIR /
            "regime_plot.png"

        )

        plt.close()

        logger.success(
            "Regime Plot Saved."
        )
        ############################################################
    # Volatility Plot
    ############################################################

    def volatility_plot(
        self,
        df
    ):

        logger.info(
            "Creating Volatility Plot..."
        )

        plt.figure(figsize=(15,6))

        plt.plot(
            df["Volatility"]
        )

        plt.title(
            "Market Volatility"
        )

        plt.xlabel("Time")

        plt.ylabel("Volatility")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR /
            "volatility_plot.png"
        )

        plt.close()

        logger.success(
            "Volatility Plot Saved."
        )

    ############################################################
    # Confidence Plot
    ############################################################

    def confidence_plot(
        self,
        df
    ):

        logger.info(
            "Creating Confidence Plot..."
        )

        plt.figure(figsize=(15,6))

        plt.plot(
            df["Confidence"]
        )

        plt.title(
            "Prediction Confidence"
        )

        plt.xlabel("Time")

        plt.ylabel("Confidence")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR /
            "confidence_plot.png"
        )

        plt.close()

        logger.success(
            "Confidence Plot Saved."
        )

    ############################################################
    # Return Distribution
    ############################################################

    def return_distribution(
        self,
        df
    ):

        logger.info(
            "Creating Return Distribution..."
        )

        plt.figure(figsize=(10,6))

        plt.hist(
            df["Return"],
            bins=50
        )

        plt.title(
            "Return Distribution"
        )

        plt.xlabel("Return")

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR /
            "return_distribution.png"
        )

        plt.close()

        logger.success(
            "Return Distribution Saved."
        )

    ############################################################
    # Run Pipeline
    ############################################################

    def run(self):

        logger.info("=" * 70)
        logger.info("Starting Visualization...")
        logger.info("=" * 70)

        df = self.load_dataset()

        self.regime_plot(df)

        self.volatility_plot(df)

        self.confidence_plot(df)

        self.return_distribution(df)

        logger.success("=" * 70)
        logger.success(
            "Visualization Completed Successfully."
        )
        logger.success("=" * 70)