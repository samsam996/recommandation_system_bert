import typer

from src.nlp import sentiment_analysis
from src.get_first_rows import get_first_rows
from src.xgb_classifier import xgb_classifier


app = typer.Typer()

app.add_typer(sentiment_analysis.app)

if __name__ == "__main__":
    app()


    path1 = "data/All_Beauty.jsonl"
    path2 = "data/meta_All_Beauty.jsonl"

    # data processing 
    first_rows = 1000
    df1, df2 = get_first_rows(path1, path2, first_rows)
    print(df1.shape)
    print(df2.shape)
    df1["label"] = (df1["rating"] >= 4).astype(int)

    xgb_classifier(df1)

