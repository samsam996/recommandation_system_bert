from src.log.logger import logger


def sentiment_classifier_pipeline(category: str, nb_rows: int) -> None:
    """
    Description:
        ...
    Args:
        - category: ...
        - nb_rows: ...
    """
    logger.info(f"Running sentiment classifiers pipeline | category={category}, nb_rows={nb_rows}")

    # core pipeline code here