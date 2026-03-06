import typer


def visualize(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8050, "--port", "-p", help="Server port"),
    debug: bool = typer.Option(True, "--debug/--no-debug", help="Enable debug mode"),
):
    from visualizer import app as dash_app

    dash_app.run(host=host, port=port, debug=debug)
