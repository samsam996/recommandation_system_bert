from src.log.logger import logger
from src.get_first_rows import get_first_rows
from src.xgb_classifier import xgb_classifier
from src.llm_classifier import llm_classifier

def sentiment_xgb_classifier_pipeline(category: str, nb_rows: int) -> None:
    """
    Description:
        ...
    Args:
        - category: ...
        - nb_rows: ...
    """


    logger.info(f"Running sentiment classifiers pipeline | category={category}, nb_rows={nb_rows}")

    path1 = "data/All_Beauty.jsonl"
    path2 = "data/meta_All_Beauty.jsonl"

    # data processing 
    first_rows = 1000
    df1, df2 = get_first_rows(path1, path2, first_rows)
    df1["label"] = (df1["rating"] >= 4).astype(int)

    xgb_classifier(df1)

    # core pipeline code here



def sentiment_llm_classifier_pipeline(category: str, nb_rows: int) -> None:

    logger.info(f"Running sentiment classifiers pipeline | category={category}, nb_rows={nb_rows}")


    
    path1 = "data/All_Beauty.jsonl"
    path2 = "data/meta_All_Beauty.jsonl"

    first_rows = 1000
    df1, df2 = get_first_rows(path1, path2, first_rows)
    df1["label"] = (df1["rating"] >= 4).astype(int)

    llm_classifier(df1)

