from pathlib import Path

import click
import numpy as np

from jacobi import RunConfig, Solver
from jacobi.config import load_toml
from jacobi.writers import write_csv


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="TOML file with run settings. Command-line options override it.",
)
@click.option(
    "--grid-size", type=click.IntRange(min=3), help="Number of cells along each side."
)
@click.option(
    "--iterations",
    type=click.IntRange(min=0),
    help="Number of Jacobi steps. [default: 500]",
)
@click.option(
    "--hot-value", type=float, help="Start value of the centre cell. [default: 100.0]"
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="CSV file to write the grid to. [default: output.csv]",
)
def main(
    config_path: Path | None,
    grid_size: int | None,
    iterations: int | None,
    hot_value: float | None,
    output: Path | None,
) -> None:
    """Diffuse heat from one hot cell in the middle of a square grid."""
    values = load_toml(config_path) if config_path else {}
    overrides = {
        "grid_size": grid_size,
        "iterations": iterations,
        "hot_value": hot_value,
        "output": output,
    }
    values.update({k: v for k, v in overrides.items() if v is not None})
    if "grid_size" not in values:
        raise click.UsageError(
            "Give --grid-size or set grid_size in the --config file."
        )
    try:
        config = RunConfig(**values)
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    n = config.grid_size
    initial = np.zeros((n, n), dtype=np.float64)
    initial[n // 2, n // 2] = config.hot_value

    solver = Solver(initial)
    solver.step(config.iterations)
    grid = solver.to_numpy()

    write_csv(config.output, grid)
    click.echo(f"Wrote {config.output} after {solver.steps} steps")
    click.echo(f"Grid sum: {grid.sum()}  max: {grid.max()}")


if __name__ == "__main__":
    main()
