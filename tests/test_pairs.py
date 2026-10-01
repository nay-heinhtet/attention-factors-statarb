import numpy as np
import pytest

from statarb.baselines.pairs import fit_pair, pair_states


def ou(n: int, phi: float, sd: float, rng) -> np.ndarray:
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sd)
    return x


def test_cointegrated_pair_is_detected():
    rng = np.random.default_rng(0)
    log_x = np.log(50) + np.cumsum(rng.normal(0, 0.015, 252))
    log_y = 0.2 + 1.5 * log_x + ou(252, 0.8, 0.01, rng)
    fit = fit_pair(log_y, log_x)
    assert fit.pvalue < 0.01
    assert fit.beta == pytest.approx(1.5, abs=0.1)
    assert 1 <= fit.halflife <= 10


def test_independent_random_walks_are_mostly_rejected():
    # The test has nominal 5% size; 400 draws keep the check tight without being flaky.
    rng = np.random.default_rng(1)
    false_positives = sum(fit_pair(*(np.cumsum(rng.normal(0, 0.015, 252)) for _ in range(2))).pvalue < 0.05
                          for _ in range(400))
    assert false_positives / 400 <= 0.08


def test_state_machine_entries_exits_and_stop():
    z = np.array([0.0, -2.1, -1.0, -0.4, 0.0, 2.2, 1.0, 0.3, -2.5, -4.2, -2.5])
    states = pair_states(z, np.ones(len(z), bool), entry=2.0, exit_=0.5, stop=4.0)
    assert states.tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 1, 0, 0]  # stop-loss disables the pair


def test_untradable_leg_closes_and_disables():
    z = np.array([-2.5, -2.5, -2.5, -2.5])
    states = pair_states(z, np.array([True, False, True, True]), 2.0, 0.5, 4.0)
    assert states.tolist() == [1, 0, 0, 0]


def test_reverting_pair_makes_money_before_costs():
    rng = np.random.default_rng(2)
    spread = ou(2000, 0.9, 0.02, rng)
    z = spread / spread.std()
    states = pair_states(z[:-1], np.ones(1999, bool), 2.0, 0.5, 4.0)
    pnl = states * np.diff(spread)  # long spread profits when the spread rises
    assert pnl.sum() > 0
