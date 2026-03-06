import typer
from commands import app


@app.command()
def train(
    ctx: typer.Context,
    epoch_size: int = typer.Option(10, "--epoch", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(64, "--batch", "-b", help="Batch size"),
    lr: float = typer.Option(1e-4, "--lr", "-l", help="Learning rate"),
    checkpoint_path: str = typer.Option(
        "hash_checkpoint.pt", "--checkpoint", "-c", help="Checkpoint path"
    ),
):
    from compressors import train as train_module

    train_module(
        epoch_size=epoch_size,
        batch_size=batch_size,
        lr=lr,
        checkpoint_path=checkpoint_path,
    )
