import typer

from src.nlp import sentiment_analysis


app = typer.Typer()

app.add_typer(sentiment_analysis.app)

if __name__ == "__main__":
    app()
