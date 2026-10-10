import numpy as np
import pandas as pd
import pytest

from reorg.survival import bootstrap_cif, cumulative_incidence, kaplan_meier


def test_km_hand_computed():
    t = np.array([1, 2, 2, 3, 4])
    e = np.array([1, 1, 0, 1, 0], bool)
    S = kaplan_meier(t, e, np.array([0.5, 1, 2, 3, 4]))
    # 5 at risk at t=1 -> 4/5; at t=2: 4 at risk, 1 event -> 0.8*0.75=0.6; t=3: 2 at risk -> 0.3
    assert np.allclose(S, [1, 0.8, 0.6, 0.3, 0.3])


def test_cif_no_censoring_equals_empirical_fractions():
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 10, 500)
    c = rng.choice([1, 2], 500, p=[0.3, 0.7])
    grid = np.array([2.0, 5.0, 10.0])
    out = cumulative_incidence(t, c, grid)
    for g, row in zip(grid, out.itertuples(), strict=True):
        assert row.cif_1 == pytest.approx(np.mean((t <= g) & (c == 1)))
        assert row.cif_2 == pytest.approx(np.mean((t <= g) & (c == 2)))
    assert out["cif_1"].iloc[-1] + out["cif_2"].iloc[-1] == pytest.approx(1.0)


def test_cif_plus_surv_is_one_with_censoring():
    rng = np.random.default_rng(2)
    t = rng.exponential(5, 300)
    c = rng.choice([0, 1, 2], 300)
    out = cumulative_incidence(t, c, np.linspace(0, 20, 11))
    assert np.allclose(out["cif_1"] + out["cif_2"] + out["surv"], 1.0)


def test_bootstrap_brackets_estimate():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "match_id": rng.integers(0, 10, 400),
            "time_s": rng.uniform(0, 20, 400),
            "code": rng.choice([0, 1, 2], 400),
        }
    )
    out = bootstrap_cif(df, np.array([5.0, 10.0]), n_boot=50, seed=0)
    assert (out["cif_1_lo"] <= out["cif_1"]).all() and (out["cif_1"] <= out["cif_1_hi"]).all()


def _cif_loop(time, code, grid, causes=(1, 2)):
    """Reference textbook loop (the original implementation)."""
    from reorg.survival import _step_eval

    time = np.asarray(time, float)
    code = np.asarray(code, int)
    ut = np.unique(time[code > 0])
    S, cif, vals, svals = 1.0, {c: 0.0 for c in causes}, {c: [] for c in causes}, []
    for t in ut:
        n = np.sum(time >= t)
        at = time == t
        d_all = np.sum(at & (code > 0))
        for c in causes:
            cif[c] += S * np.sum(at & (code == c)) / n
            vals[c].append(cif[c])
        S *= 1 - d_all / n
        svals.append(S)
    out = {"t": grid}
    for c in causes:
        out[f"cif_{c}"] = _step_eval(ut, np.asarray(vals[c]), grid, 0.0)
    out["surv"] = _step_eval(ut, np.asarray(svals), grid, 1.0)
    return pd.DataFrame(out)


def test_vectorised_cif_is_bit_identical_to_loop():
    rng = np.random.default_rng(5)
    grid = np.round(np.arange(0, 20.01, 0.1), 1)
    for _ in range(20):
        n = rng.integers(5, 400)
        t = np.round(rng.uniform(0, 20, n), 1)  # tenth-of-second ties, as in real data
        c = rng.choice([0, 1, 2], n)
        a = cumulative_incidence(t, c, grid)
        b = _cif_loop(t, c, grid)
        assert all(np.array_equal(a[k].to_numpy(), b[k].to_numpy()) for k in a.columns)


def test_cif_with_no_events():
    out = cumulative_incidence(np.array([1.0, 2.0]), np.array([0, 0]), np.array([0.0, 5.0]))
    assert (out["cif_1"] == 0).all() and (out["surv"] == 1).all()
