import typer
from commands import benchmark, generate, train, visualize

app = typer.Typer(
    name="mini-nav",
    help="Mini-Nav: A vision-language navigation system",
    add_completion=False,
)

app.command(name="train")(train)
app.command(name="benchmark")(benchmark)
app.command(name="visualize")(visualize)
app.command(name="generate")(generate)

if __name__ == "__main__":
    app()
