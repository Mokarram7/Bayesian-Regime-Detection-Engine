import numpy as np
import pandas as pd
from src.logger import logger
from src.config import PROCESSED_DATA_DIR


class ConformalPrediction:

    """
    Conformal Prediction
    """

    def __init__(self):

        self.input_file = (
            PROCESSED_DATA_DIR /
            "ensemble_predictions.csv"
        )

        self.output_file = (
            PROCESSED_DATA_DIR /
            "conformal_predictions.csv"
        )

    
    # Load Dataset
    def load_dataset(self):

        logger.info(
            "Loading Ensemble Dataset..."
        )

        df = pd.read_csv(
            self.input_file
        )

        logger.success(
            f"Dataset Loaded : {df.shape}"
        )

        return df

    # Nonconformity Score
    def calculate_nonconformity(
        self,
        df
    ):

        logger.info(
            "Calculating Nonconformity Score..."
        )

        df = df.copy()

        df["Nonconformity"] = (

            1 -

            df["Ensemble_Score"]

        )

        logger.success(
            "Nonconformity Calculated."
        )

        return df

    # Quantile
    def calculate_quantile(
        self,
        df,
        alpha=0.05
    ):

        logger.info(
            "Calculating Quantile..."
        )

        q = np.quantile(

            df["Nonconformity"],

            1 - alpha

        )

        logger.success(
            f"Quantile : {q:.4f}"
        )

        return q

    # Prediction Interval
    def prediction_interval(
        self,
        df,
        q
    ):

        logger.info(
            "Generating Prediction Interval..."
        )

        df = df.copy()

        df["Prediction_Lower"] = (
            df["Ensemble_Score"] - q
        )

        df["Prediction_Upper"] = (
            df["Ensemble_Score"] + q
        )

        df["Prediction_Width"] = (
            df["Prediction_Upper"] -
            df["Prediction_Lower"]
        )

        return df


    # Coverage
    def calculate_coverage(
        self,
        df
    ):

        logger.info(
            "Calculating Coverage..."
        )

        df["Coverage"] = np.where(

            (
                df["Ensemble_Score"] >=
                df["Prediction_Lower"]
            )

            &

            (
                df["Ensemble_Score"] <=
                df["Prediction_Upper"]
            ),

            1,

            0

        )

        return df


    # Prediction Confidence
    def prediction_confidence(
        self,
        df
    ):

        logger.info(
            "Assigning Prediction Confidence..."
        )

        conditions = [

            df["Coverage"] == 1,

            df["Coverage"] == 0

        ]

        values = [

            "Reliable",

            "Uncertain"

        ]

        df["Prediction_Confidence"] = np.select(

            conditions,

            values,

            default="Unknown"

        )

        return df


    # Save Dataset
    def save_dataset(
        self,
        df
    ):

        logger.info(
            "Saving Conformal Dataset..."
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
        logger.info("Starting Conformal Prediction...")
        logger.info("=" * 70)

        df = self.load_dataset()

        df = self.calculate_nonconformity(df)

        q = self.calculate_quantile(df)

        df = self.prediction_interval(df, q)

        df = self.calculate_coverage(df)

        df = self.prediction_confidence(df)

        self.save_dataset(df)

        logger.success("=" * 70)
        logger.success(
            "Conformal Prediction Completed Successfully."
        )
        logger.success("=" * 70)

        return df