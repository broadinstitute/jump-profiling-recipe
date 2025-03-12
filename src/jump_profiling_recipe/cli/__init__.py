import click

from .converter import convert_command


@click.group()
def cli():
    """JUMP Profiling Recipe CLI tools."""
    pass


# Register commands
cli.add_command(convert_command)

if __name__ == "__main__":
    cli()
