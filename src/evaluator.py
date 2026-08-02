import pandas as pd

from src.logger import logger
from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR
)


class Evaluator:

    """
    Evaluate Final Regime Detection Results
    """

    def __init__(self):

        self.input_file = (
            PROCESSED_DATA_DIR /
            "conformal_predictions.csv"
        )

        self.report_file = (
            REPORTS_DIR /
            "evaluation_report.txt"
        )

    ############################################################
    # Load Dataset
    ############################################################

    def load_dataset(self):

        logger.info(
            "Loading Final Dataset..."
        )

        df = pd.read_csv(
            self.input_file
        )

        logger.success(
            f"Dataset Loaded : {df.shape}"
        )

        return df

    ############################################################
    # Regime Distribution
    ############################################################

    def regime_distribution(
        self,
        df
    ):

        logger.info(
            "Calculating Regime Distribution..."
        )

        regime = (

            df["Final_Regime"]

            .value_counts()

            .sort_index()

        )

        return regime

    ############################################################
    # Confidence Summary
    ############################################################

    def confidence_summary(
        self,
        df
    ):

        logger.info(
            "Calculating Confidence Summary..."
        )

        confidence = (

            df["Confidence"]

            .describe()

        )

        return confidence
        ############################################################
    # Risk Summary
    ############################################################

    def risk_summary(
        self,
        df
    ):

        logger.info(
            "Calculating Risk Summary..."
        )

        risk = (

            df["Risk_Score"]

            .describe()

        )

        return risk

    ############################################################
    # Coverage Summary
    ############################################################

    def coverage_summary(
        self,
        df
    ):

        logger.info(
            "Calculating Coverage Summary..."
        )

        coverage = (

            df["Coverage"]

            .value_counts()

        )

        return coverage

    ############################################################
    # Generate Report
    ############################################################

    def generate_report(
        self,
        regime,
        confidence,
        risk,
        coverage
    ):

        report = []

        report.append("=" * 70)
        report.append("Bayesian Regime Detection Evaluation Report")
        report.append("=" * 70)

        report.append("\nRegime Distribution\n")
        report.append(regime.to_string())

        report.append("\n\nConfidence Summary\n")
        report.append(confidence.to_string())

        report.append("\n\nRisk Summary\n")
        report.append(risk.to_string())

        report.append("\n\nCoverage Summary\n")
        report.append(coverage.to_string())

        return "\n".join(report)

    ############################################################
    # Save Report
    ############################################################

    def save_report(
        self,
        report
    ):

        logger.info(
            "Saving Evaluation Report..."
        )

        with open(
            self.report_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(report)

        logger.success(
            f"Saved : {self.report_file}"
        )

    ############################################################
    # Run Pipeline
    ############################################################

    def run(self):

        logger.info("=" * 70)
        logger.info("Starting Evaluation...")
        logger.info("=" * 70)

        df = self.load_dataset()

        regime = self.regime_distribution(df)

        confidence = self.confidence_summary(df)

        risk = self.risk_summary(df)

        coverage = self.coverage_summary(df)

        report = self.generate_report(
            regime,
            confidence,
            risk,
            coverage
        )

        self.save_report(report)

        logger.success("=" * 70)
        logger.success(
            "Evaluation Completed Successfully."
        )
        logger.success("=" * 70)

        return report
    