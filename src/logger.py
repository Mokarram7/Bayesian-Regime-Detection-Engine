from loguru import logger

from .config import LOG_DIR

logger.remove()

logger.add(

    LOG_DIR / "project.log",

    rotation="5 MB",

    retention="10 days",

    level="INFO"

)

logger.add(

    lambda msg: print(msg, end=""),

    level="INFO"

)