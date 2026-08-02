from src.logger import logger
from src.pipeline import BayesianPipeline
from src.dashboard import Dashboard


def main():

    logger.info("=" * 70)
    logger.info("Bayesian Regime Detection Engine Started")
    logger.info("=" * 70)

    pipeline = BayesianPipeline()

    pipeline.run()


if __name__ == "__main__":
    main()