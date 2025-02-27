import typer
from typing_extensions import Annotated

app = typer.Typer(name="sentiment")


@app.command(name="xgboost-classifier")
def run_sentiment_classifier_pipeline(
    category: Annotated[str, typer.Option(help="Amazon data category")] = "All Beauty",
    nb_rows: Annotated[int, typer.Option(help="Amazon data category")] = 10_000,
) -> None:
    
    from src.nlp.sentiment_analysis.core import sentiment_xgb_classifier_pipeline

    sentiment_xgb_classifier_pipeline(
        category=category,
        nb_rows=nb_rows,
    )

@app.command(name="llm")
def run_llm_sentiment_classifier_pipeline(
    category: Annotated[str, typer.Option(help="Amazon data category")] = "All Beauty",
    nb_rows: Annotated[int, typer.Option(help="Amazon data category")] = 10_000,
) -> None:
    
    from src.nlp.sentiment_analysis.core import sentiment_llm_classifier_pipeline

    sentiment_llm_classifier_pipeline(
            category=category,
            nb_rows=nb_rows,
    )
    