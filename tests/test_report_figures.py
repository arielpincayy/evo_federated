import numpy as np

from make_report_figures import attainment, boot_ci, common_grid, pareto_front


def test_attainment_is_nondecreasing_and_zero_below_the_smallest_model():
    acc, size = [0.3, 0.5, 0.4, 0.6], [0.1, 0.2, 0.3, 0.4]
    f = attainment(acc, size, [0.05, 0.1, 0.25, 0.35, 0.5])
    assert f.tolist() == [0.0, 0.3, 0.5, 0.5, 0.6] and np.all(np.diff(f) >= 0)


def test_pareto_front_excludes_dominated_points():
    acc, size = np.array([0.3, 0.5, 0.4, 0.6, 0.55]), np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert sorted(pareto_front(acc, size).tolist()) == [0, 1, 3]  # el 2 (0.4 con 0.3) lo domina el 1 (0.5 con 0.2); el 4 (0.55 con 0.5) lo domina el 3


def test_common_grid_interpolates_inside_the_shared_range_only():
    x, M = common_grid([(np.array([0.0, 10.0]), np.array([0.0, 10.0])), (np.array([2.0, 8.0]), np.array([4.0, 4.0]))], n=7)
    assert x[0] == 2.0 and x[-1] == 8.0 and M.shape == (2, 7) and np.allclose(M[0], x) and np.allclose(M[1], 4.0)


def test_boot_ci_contains_the_mean_and_is_deterministic():
    lo, hi = boot_ci([0.0, 1.0, 2.0, 3.0, 4.0])
    assert lo < 2.0 < hi and boot_ci([0.0, 1.0, 2.0, 3.0, 4.0]) == (lo, hi)
