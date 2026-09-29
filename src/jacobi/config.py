import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    grid_size: int
    iterations: int = 500
    hot_value: float = 100.0
    output: Path = Path("output.csv")

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if self.iterations < 0:
            raise ValueError("iterations must not be negative")
        object.__setattr__(self, "output", Path(self.output))


def load_toml(path: Path) -> dict[str, Any]:
    """Read RunConfig fields from a TOML file. Keys use the field names."""
    with path.open("rb") as f:
        values = tomllib.load(f)
    unknown = set(values) - {f.name for f in fields(RunConfig)}
    if unknown:
        raise ValueError(f"unknown keys in {path}: {', '.join(sorted(unknown))}")
    return values
