from src.logger import logger

from src.data_loader import DataLoader
from src.preprocess import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.hmm_model import HMMTrainer
from src.bayesian_model import BayesianModel
from src.ensemble import EnsembleModel
from src.conformal import ConformalPrediction
from src.evaluator import Evaluator
from src.visualization import Visualizer
from src.dashboard import Dashboard


class BayesianPipeline:

    """
    Complete Bayesian Regime Detection Pipeline
    """

    def __init__(self):

        self.loader = DataLoader()

        self.preprocessor = Preprocessor()

        self.feature_engineer = FeatureEngineer()

        self.hmm = HMMTrainer()

        self.bayesian = BayesianModel()

        self.ensemble = EnsembleModel()

        self.conformal = ConformalPrediction()

        self.evaluator = Evaluator()

        self.visualizer = Visualizer()

        self.dashboard = Dashboard()

    ############################################################
    # Run Complete Pipeline
    ############################################################

    def run(self):

        logger.info("=" * 70)
        logger.info("Starting Complete Bayesian Pipeline")
        logger.info("=" * 70)

        self.loader.download()

        self.preprocessor.process()

        self.feature_engineer.process()

        self.hmm.run()

        self.bayesian.run()

        self.ensemble.run()

        self.conformal.run()

        self.evaluator.run()

        self.visualizer.run()

        self.dashboard.run()

        logger.success("=" * 70)
        logger.success("Complete Pipeline Finished Successfully")
        logger.success("=" * 70)