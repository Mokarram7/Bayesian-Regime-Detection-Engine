import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.logger import logger
from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR
)


class HMMTrainer:
    """
    Hidden Markov Model Trainer

    Responsibilities
    ----------------
    1. Load Feature Dataset
    2. Select Features
    3. Scale Features
    4. Train Gaussian HMM
    5. Save Model
    6. Predict Regimes
    """

    def __init__(self):

        self.input_file = (
            PROCESSED_DATA_DIR /
            "feature_data.csv"
        )

        self.model_path = (
            MODEL_DIR /
            "trained" /
            "gaussian_hmm.pkl"
        )

        self.scaler_path = (
            MODEL_DIR /
            "scalers" /
            "standard_scaler.pkl"
        )

        # Create folders automatically
        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.scaler_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.scaler = StandardScaler()

        self.model = GaussianHMM(

            n_components=5,

            covariance_type="full",

            n_iter=1000,

            random_state=42,

            verbose=False,
            tol=1e-3

        )

        self.features = [

            "Return",

            "Log_Return",

            "Momentum_10",

            "SMA20",

            "SMA50",

            "EMA20",

            "EMA50",

            "Price_SMA20",

            "Volatility",

            "ATR",

            "RSI",

            "MACD",

            "MACD_SIGNAL",

            "MACD_HIST"

        ]

    ####################################################################
    # Dataset
    ####################################################################

    def load_dataset(self):

        logger.info(
            "Loading Feature Dataset..."
        )

        df = pd.read_csv(
            self.input_file
        )

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        logger.success(
            f"Dataset Loaded : {df.shape}"
        )

        return df

    ####################################################################
    # Feature Selection
    ####################################################################

    def prepare_features(
        self,
        df
    ):

        logger.info(
            "Preparing Feature Matrix..."
        )

        X = (
            df[self.features]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        df = df.loc[X.index].reset_index(drop=True)
        X = X.reset_index(drop=True)

        self.df = df

        logger.info(
            f"Selected Features : {len(self.features)}"
        )

        logger.info(
            f"Matrix Shape : {X.shape}"
        )

        return X

    ####################################################################
    # Scaling
    ####################################################################

    def scale_features(
        self,
        X
    ):

        logger.info(
            "Scaling Features..."
        )

        X_scaled = self.scaler.fit_transform(
            X
        )

        joblib.dump(

            self.scaler,

            self.scaler_path

        )

        logger.success(
            "Scaler Saved Successfully."
        )

        return X_scaled
        ####################################################################
    # Train Model
    ####################################################################

    def train(
        self,
        X_scaled
    ):

        logger.info(
            "Training Gaussian Hidden Markov Model..."
        )

        self.model.fit(
            X_scaled
        )

        logger.success(
            "Training Completed Successfully."
        )

        return self.model

    ####################################################################
    # Save Model
    ####################################################################

    def save_model(self):

        joblib.dump(

            self.model,

            self.model_path

        )

        logger.success(

            f"Model Saved : {self.model_path}"

        )

    ####################################################################
    # Load Model
    ####################################################################

    def load_model(self):

        logger.info(
            "Loading Trained Model..."
        )

        self.model = joblib.load(
            self.model_path
        )

        logger.success(
            "Model Loaded Successfully."
        )

        return self.model

    ####################################################################
    # Predict Hidden States
    ####################################################################

    def predict_hidden_states(
        self,
        X_scaled
    ):

        logger.info(
            "Predicting Hidden Regimes..."
        )

        hidden_states = self.model.predict(
            X_scaled
        )

        logger.success(
            "Prediction Completed."
        )

        return hidden_states

    ####################################################################
    # Posterior Probability
    ####################################################################

    def predict_probability(
        self,
        X_scaled
    ):

        logger.info(
            "Calculating Posterior Probabilities..."
        )

        try:
            probabilities = self.model.predict_proba(X_scaled)
        except AttributeError:
            probabilities = self.model.predict_proba(X_scaled)
        

        logger.success(
            "Posterior Probabilities Generated."
        )

        return probabilities

    ####################################################################
    # Transition Matrix
    ####################################################################

    def transition_matrix(self):

        logger.info(
            "Generating Transition Matrix..."
        )

        transition = pd.DataFrame(

            self.model.transmat_,

            columns=[
                f"State_{i}"
                for i in range(
                    self.model.n_components
                )
            ],

            index=[
                f"State_{i}"
                for i in range(
                    self.model.n_components
                )
            ]

        )

        logger.success(
            "Transition Matrix Created."
        )

        return transition

    ####################################################################
    # Initial State Probability
    ####################################################################

    def initial_probability(self):

        probability = pd.DataFrame(

            {

                "State":

                [

                    f"State_{i}"

                    for i in range(

                        self.model.n_components

                    )

                ],

                "Probability":

                self.model.startprob_

            }

        )

        return probability
        ####################################################################
    # Regime Statistics
    ####################################################################

    def regime_statistics(
        self,
        df
    ):

        logger.info(
            "Calculating Regime Statistics..."
        )

        stats = (

            df
            .groupby("Regime")
            .agg(

                Mean_Return=("Return", "mean"),

                Volatility=("Volatility", "mean"),

                Avg_RSI=("RSI", "mean"),

                Avg_MACD=("MACD", "mean"),

                Samples=("Regime", "count")

            )

        )

        logger.success(
            "Statistics Generated."
        )

        return stats

    ####################################################################
    # Attach Hidden States
    ####################################################################

    def attach_regimes(
        self,
        df,
        hidden_states,
        probabilities
    ):

        logger.info(
            "Attaching Regimes..."
        )

        df = self.df.copy()

        df["Regime"] = hidden_states

        for i in range(self.model.n_components):

            df[f"Prob_State_{i}"] = probabilities[:, i]

        logger.success(
            "Regimes Attached Successfully."
        )

        return df

    ####################################################################
    # Save Dataset
    ####################################################################

    def save_dataset(
        self,
        df
    ):

        output = (
            PROCESSED_DATA_DIR /
            "regime_data.csv"
        )

        df.to_csv(
            output,
            index=False
        )

        logger.success(
            f"Dataset Saved : {output}"
        )

    ####################################################################
    # Evaluate
    ####################################################################

    def evaluate(
        self,
        df
    ):

        logger.info(
            "Model Summary"
        )

        logger.info(
            f"Log Likelihood : {self.model.score(self.scaler.transform(df[self.features])):.2f}"
        )

        logger.info(
            f"Hidden States : {self.model.n_components}"
        )

        logger.info(
            f"Observations : {len(df)}"
        )

    ####################################################################
    # Complete Pipeline
    ####################################################################

    def run(self):

        logger.info(
            "=" * 70
        )

        logger.info(
            "Starting HMM Pipeline..."
        )

        logger.info(
            "=" * 70
        )

        df = self.load_dataset()

        X = self.prepare_features(df)

        df = self.df

        X_scaled = self.scale_features(X)

        self.train(X_scaled)

        self.save_model()

        hidden_states = self.predict_hidden_states(
            X_scaled
        )

        probabilities = self.predict_probability(
            X_scaled
        )

        transition = self.transition_matrix()

        regime_df = self.attach_regimes(

            df,

            hidden_states,

            probabilities

        )

        self.save_dataset(
            regime_df
        )

        stats = self.regime_statistics(
            regime_df
        )

        self.evaluate(
            regime_df
        )

        logger.success(
            "=" * 70
        )

        logger.success(
            "HMM Pipeline Finished Successfully."
        )

        logger.success(
            "=" * 70
        )

        return {

            "data": regime_df,

            "transition_matrix": transition,

            "statistics": stats

        }