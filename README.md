# jacobi-omp

This project is a small example of a C++ solver that Python calls. The solver runs Jacobi
iterations of 2D heat diffusion on a square grid. OpenMP runs the solver in parallel, on the CPU
threads or on a GPU. nanobind makes the solver available to Python, and a click command drives it.

The command puts one hot cell in the middle of the grid. It then runs the diffusion and writes
the grid to a CSV file. The outer ring of cells stays at 0.

## Run it

You need a C++ compiler with OpenMP, CMake and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run jacobi-solve --grid-size 128
```

The `uv sync` command builds the extension with scikit-build-core, which runs `CMakeLists.txt`.
The command writes `output.csv` and prints the sum and the maximum of the grid.

The command has these options:

- `--grid-size`: the number of cells along each side. The minimum is 3.
- `--iterations`: the number of Jacobi steps. The default is 500.
- `--hot-value`: the start value of the center cell. The default is 100.
- `--output`: the CSV file to write. The default is `output.csv`.
- `--config`: a TOML file with the same values. Options on the command line override it.

The TOML keys are the field names of `RunConfig` in `src/jacobi/config.py`:

```toml
grid_size = 256
iterations = 2000
output = "run.csv"
```

`OMP_NUM_THREADS` sets the number of CPU threads.

## Design

The physics is in C++. Python does the configuration, the orchestration and the file output.
Python never calls C++ for each cell or for each row. Each call does work on the whole grid.

The C++ code gives Python two ways to run the solver:

- `Solver` is the path for real runs. Its constructor copies the grid to the device one time.
  `step(n)` runs `n` steps on the device with no copies to the host. Python calls
  `to_numpy()` or `sync_to_host(out)` to copy the grid back, and only for output or tests.
- `jacobi_step(grid)` is a stateless function for tests. It copies the grid to the device, runs
  one step, and returns a new array. The tests in `tests/` compare it with a numpy version.

## Make targets

The Makefile is a wrapper around uv and Docker:

- `make build` rebuilds the extension in `.venv`.
- `make test` runs the tests with pytest.
- `make run` runs the command on a 64 x 64 grid.
- `make docker` builds the Docker image.
- `make clean` removes the CMake build directory.

When a file in `src/cpp/` or `CMakeLists.txt` changes, `uv sync` also rebuilds the extension.

## GPU offload

By default, the OpenMP target regions run on the CPU threads. To run them on a GPU, set `OFFLOAD`:

CMake keeps the first compiler that it finds in `build/`. If you change `CXX`, run `make clean` first.

```sh
make clean
CXX=clang++ make build OFFLOAD=nvidia
CXX=clang++ make build OFFLOAD=amd GPU_ARCH=gfx90a
```

GPU offload needs a clang that has the OpenMP offload runtime (`libomptarget`) for your GPU.

## Docker

The Docker image builds the wheel with clang and libomp, and installs it in a slim Python image.
The image runs `jacobi-solve`. Mount a directory to keep the CSV file:

```sh
docker build -t jacobi-omp .
docker run --rm -v "$PWD:/work" jacobi-omp --grid-size 128
```
