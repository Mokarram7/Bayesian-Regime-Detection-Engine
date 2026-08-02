import pandas as pd
import numpy as np

from src.logger import logger
from src.config import PROCESSED_DATA_DIR


class EnsembleModel:

    """
    Ensemble Model

    Combines:
    1. Hidden Markov Model
    2. Bayesian Probabilities
    """

    def __init__(self):

        self.input_file = (
            PROCESSED_DATA_DIR /
            "bayesian_regime_data.csv"
        )

        self.output_file = (
            PROCESSED_DATA_DIR /
            "ensemble_predictions.csv"
        )


    # Load Dataset
    def load_dataset(self):

        logger.info(
            "Loading Bayesian Dataset..."
        )

        df = pd.read_csv(
            self.input_file
        )

        logger.success(
            f"Dataset Loaded : {df.shape}"
        )

        return df


    # Ensemble Score
    def calculate_ensemble_score(
        self,
        df
    ):

        logger.info(
            "Calculating Ensemble Score..."
        )

        df = df.copy()

        df["Ensemble_Score"] = (

            0.6 * df["Posterior_Prob"]

            +

            0.4 * (
                df["Confidence"] / 100
            )

        )

        logger.success(
            "Ensemble Score Generated."
        )

        return df

    # Final Regime
    def assign_final_regime(
        self,
        df
    ):

        logger.info(
            "Assigning Final Regime..."
        )

        regime_map = {

            0: "Bull",

            1: "Bear",

            2: "Correction",

            3: "Crash",

            4: "Recovery"

        }

        df["Final_Regime"] = (

            df["Regime"]

            .map(regime_map)

        )

        logger.success(
            "Final Regime Assigned."
        )

        return df


    # Risk Score
    def calculate_risk(
        self,
        df
    ):

        logger.info(
            "Calculating Risk Score..."
        )

        df["Risk_Score"] = (

            df["Volatility"]

            * 100

        )

        return df


    # Confidence Level
    def confidence_level(
        self,
        df
    ):

        logger.info(
            "Calculating Confidence Level..."
        )

        conditions = [

            df["Confidence"] >= 80,

            df["Confidence"] >= 60,

            df["Confidence"] >= 40,

        ]

        values = [

            "High",

            "Medium",

            "Low"

        ]

        df["Confidence_Level"] = np.select(

            conditions,

            values,

            default="Very Low"

        )

        return df


    # Save Dataset
    def save_dataset(
        self,
        df
    ):

        logger.info(
            "Saving Ensemble Dataset..."
        )

        df.to_csv(

            self.output_file,

            index=False

        )

        logger.success(

            f"Saved : {self.output_file}"

        )

    # Run Pipeline
    def run(self):

        logger.info("=" * 70)
        logger.info("Starting Ensemble Model...")
        logger.info("=" * 70)

        df = self.load_dataset()

        df = self.calculate_ensemble_score(df)

        df = self.assign_final_regime(df)

        df = self.calculate_risk(df)

        df = self.confidence_level(df)

        self.save_dataset(df)

        logger.success("=" * 70)
        logger.success(
            "Ensemble Pipeline Completed Successfully."
        )
        logger.success("=" * 70)

        return df