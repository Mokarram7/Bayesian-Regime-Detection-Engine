import numpy as np
import pandas as pd

from src.logger import logger
from src.config import PROCESSED_DATA_DIR


class BayesianModel:

    """
    Bayesian Regime Probability Model
    """

    def __init__(self):

        self.input_file = (
            PROCESSED_DATA_DIR /
            "regime_data.csv"
        )

        self.output_file = (
            PROCESSED_DATA_DIR /
            "bayesian_regime_data.csv"
        )

    
    # Load Dataset
    def load_dataset(self):

        logger.info(
            "Loading Regime Dataset..."
        )

        df = pd.read_csv(
            self.input_file
        )

        logger.success(
            f"Dataset Loaded : {df.shape}"
        )

        return df

    
    # Prior Probability
    def calculate_prior(
        self,
        df
    ):

        logger.info(
            "Calculating Prior Probability..."
        )

        prior = (

            df["Regime"]

            .value_counts(normalize=True)

            .sort_index()

        )

        logger.success(
            "Prior Calculated."
        )

        return prior

    
    # Likelihood
    def calculate_likelihood(
        self,
        df
    ):

        logger.info(
            "Calculating Likelihood..."
        )

        likelihood = (

            df.groupby("Regime")["Return"]

            .mean()

            .abs()

        )

        logger.success(
            "Likelihood Generated."
        )

        return likelihood
        
    # Posterior Probability
    def calculate_posterior(
        self,
        prior,
        likelihood
    ):

        logger.info(
            "Calculating Posterior Probability..."
        )

        posterior = prior * likelihood

        posterior = posterior / posterior.sum()

        logger.success(
            "Posterior Probability Generated."
        )

        return posterior

    
    # Confidence Score
    def calculate_confidence(
        self,
        posterior
    ):

        logger.info(
            "Calculating Confidence Score..."
        )

        confidence = posterior * 100

        return confidence

    
    # Uncertainty Score
    def calculate_uncertainty(   
        self,
        confidence
    ):

        logger.info(
            "Calculating Uncertainty..."
        )

        uncertainty = 100 - confidence

        return uncertainty

    
    # Attach Bayesian Results
    def attach_results(
        self,
        df,
        posterior,
        confidence,
        uncertainty
    ):

        logger.info(
            "Attaching Bayesian Results..."
        )

        posterior_map = posterior.to_dict()

        confidence_map = confidence.to_dict()

        uncertainty_map = uncertainty.to_dict()

        df["Posterior_Prob"] = (
            df["Regime"]
            .map(posterior_map)
        )

        df["Confidence"] = (
            df["Regime"]
            .map(confidence_map)
        )

        df["Uncertainty"] = (
            df["Regime"]
            .map(uncertainty_map)
        )

        logger.success(
            "Bayesian Results Attached."
        )

        return df
        
    # Save Dataset
    def save_dataset(
        self,
        df
    ):

        logger.info(
            "Saving Bayesian Dataset..."
        )

        df.to_csv(
            self.output_file,
            index=False
        )

        logger.success(
            f"Saved : {self.output_file}"
        )

    
    # Summary
    def summary(
        self,
        posterior
    ):

        logger.info("=" * 60)
        logger.info("Bayesian Model Summary")
        logger.info("=" * 60)

        logger.info("\nPosterior Probability")

        logger.info(posterior)

        logger.info("=" * 60)


    # Run Pipeline
    def run(self):

        logger.info("=" * 70)
        logger.info("Starting Bayesian Model...")
        logger.info("=" * 70)

        df = self.load_dataset()

        prior = self.calculate_prior(df)

        likelihood = self.calculate_likelihood(df)

        posterior = self.calculate_posterior(
            prior,
            likelihood
        )

        confidence = self.calculate_confidence(
            posterior
        )

        uncertainty = self.calculate_uncertainty(
            confidence
        )

        df = self.attach_results(
            df,
            posterior,
            confidence,
            uncertainty
        )

        self.save_dataset(df)

        self.summary(posterior)

        logger.success("=" * 70)
        logger.success(
            "Bayesian Pipeline Completed Successfully."
        )
        logger.success("=" * 70)

        return df