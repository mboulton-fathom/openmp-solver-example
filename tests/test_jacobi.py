import numpy as np
import pytest
from jacobi import Solver, jacobi_step


def reference_step(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[1:-1, 1:-1] = 0.25 * (
        grid[:-2, 1:-1] + grid[2:, 1:-1] + grid[1:-1, :-2] + grid[1:-1, 2:]
    )
    return out


@pytest.fixture
def grid() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((17, 17))


def test_step_matches_reference(grid):
    np.testing.assert_allclose(
        jacobi_step(grid), reference_step(grid), rtol=0, atol=1e-15
    )


def test_step_does_not_change_input(grid):
    before = grid.copy()
    jacobi_step(grid)
    np.testing.assert_array_equal(grid, before)


def test_step_keeps_boundary(grid):
    out = jacobi_step(grid)
    np.testing.assert_array_equal(out[0], grid[0])
    np.testing.assert_array_equal(out[-1], grid[-1])
    np.testing.assert_array_equal(out[:, 0], grid[:, 0])
    np.testing.assert_array_equal(out[:, -1], grid[:, -1])


@pytest.mark.parametrize("iterations", [0, 1, 2, 7])
def test_solver_matches_repeated_steps(grid, iterations):
    expected = grid
    for _ in range(iterations):
        expected = jacobi_step(expected)
    solver = Solver(grid)
    solver.step(iterations)
    assert solver.steps == iterations
    np.testing.assert_array_equal(solver.to_numpy(), expected)


def test_solver_steps_accumulate(grid):
    solver = Solver(grid)
    solver.step(3)
    solver.step(4)
    other = Solver(grid)
    other.step(7)
    assert solver.steps == 7
    np.testing.assert_array_equal(solver.to_numpy(), other.to_numpy())


def test_solver_does_not_keep_reference_to_input(grid):
    solver = Solver(grid)
    grid[5, 5] = 1e9
    assert solver.to_numpy()[5, 5] != 1e9


def test_sync_to_host_fills_array(grid):
    solver = Solver(grid)
    solver.step(5)
    out = np.empty_like(grid)
    solver.sync_to_host(out)
    np.testing.assert_array_equal(out, solver.to_numpy())


def test_sync_to_host_rejects_wrong_dtype(grid):
    solver = Solver(grid)
    with pytest.raises(TypeError):
        solver.sync_to_host(np.empty(grid.shape, dtype=np.float32))


def test_rejects_non_square():
    with pytest.raises(ValueError):
        jacobi_step(np.zeros((4, 5)))
    with pytest.raises(ValueError):
        Solver(np.zeros((4, 5)))


def test_rejects_negative_iterations(grid):
    with pytest.raises(ValueError):
        Solver(grid).step(-1)
