import numpy as np
import pandas as pd

from analyze_confirm import archive_hv, best_at_size, holm


def test_best_at_size_is_zero_when_no_model_is_small_enough():
    assert best_at_size([0.3, 0.5], [0.01, 0.2], 0.05) == 0.3
    assert best_at_size([0.3, 0.5], [0.1, 0.2], 0.05) == 0.0


def test_archive_hv_prefix_grows_and_matches_full_at_one():
    pop = pd.DataFrame({"acc_rep": [0.2, 0.4, 0.5, 0.45], "n_params_norm": [0.01, 0.05, 0.3, 0.02]})
    assert archive_hv(pop, 0.5) <= archive_hv(pop, 1.0) and archive_hv(pop, 1.0) == archive_hv(pop)
    assert archive_hv(pop, 0.25) == archive_hv(pop.iloc[:1])


def test_holm_adjusts_in_input_order_and_is_monotone():
    adj = holm([0.04, 0.001, 0.03])
    assert np.allclose(adj, [0.06, 0.003, 0.06])
