import numpy as np

from bsc import saxs


def sphere_curve(q, R):
    x = q * R
    return (3 * (np.sin(x) - x * np.cos(x)) / x**3) ** 2


def test_scale_and_background_recovers_factors():
    q = np.linspace(0.01, 0.3, 200)
    model = sphere_curve(q, 20.0)
    exp = 3.5 * model + 0.02
    sigma = np.full_like(q, 0.01)
    c, b = saxs.scale_and_background(model, exp, sigma)
    assert abs(c - 3.5) < 1e-6 and abs(b - 0.02) < 1e-6
    assert saxs.chi2(model, exp, sigma) < 1e-10


def test_chi2_near_one_for_noise_at_sigma():
    rng = np.random.default_rng(0)
    q = np.linspace(0.01, 0.3, 400)
    model = sphere_curve(q, 20.0)
    sigma = 0.02 * model + 1e-4
    exp = model + rng.normal(0, sigma)
    c2 = saxs.chi2(model, exp, sigma)
    assert 0.8 < c2 < 1.25


def test_guinier_recovers_rg_of_sphere():
    R = 25.0
    q = np.linspace(0.005, 0.25, 500)
    I = sphere_curve(q, R)
    g = saxs.guinier(q, I, 0.001 * I + 1e-6)
    assert g["valid"]
    assert abs(g["rg"] - R * np.sqrt(3 / 5)) / (R * np.sqrt(3 / 5)) < 0.03
    assert abs(g["upturn"]) < 0.02


def test_guinier_flags_aggregation_upturn():
    R = 25.0
    q = np.linspace(0.005, 0.25, 500)
    I = sphere_curve(q, R) * (1 + 0.6 * np.exp(-((q / 0.012) ** 2)))
    g = saxs.guinier(q, I, 0.001 * I + 1e-6)
    assert g["upturn"] > 0.1


def test_reweighting_moves_towards_the_data():
    rng = np.random.default_rng(1)
    q = np.linspace(0.01, 0.3, 150)
    radii = np.linspace(15, 35, 40)
    curves = np.array([sphere_curve(q, r) for r in radii])
    truth = 0.8 * sphere_curve(q, 32.0) + 0.2 * sphere_curve(q, 16.0)
    sigma = 0.01 * truth + 1e-5
    exp = truth + rng.normal(0, sigma)
    raw = saxs.chi2(saxs.ensemble_curve(curves), exp, sigma)
    path = saxs.reweighting_curve(curves, exp, sigma, 10.0 ** np.linspace(-2, 6, 17))
    best = min(p["chi2"] for p in path)
    assert raw > 10 and best < 1.5
    phis = [p["phi"] for p in path]
    assert phis[0] > 0.9 and phis[-1] < phis[0]
    w, _ = saxs.reweight(curves, exp, sigma, 1e-3)
    assert abs(w.sum() - 1) < 1e-9
    assert radii[np.argmax(w)] > 28


def test_phi_at_chi2_and_chi2_at_phi():
    path = [{"theta": 10, "chi2": 9.0, "phi": 1.0}, {"theta": 1, "chi2": 2.0, "phi": 0.5},
            {"theta": 0.1, "chi2": 0.9, "phi": 0.1}]
    assert saxs.phi_at_chi2(path, 1.0) == 0.1
    assert saxs.phi_at_chi2(path, 2.5) == 0.5
    assert np.isnan(saxs.phi_at_chi2(path, 0.5))
    assert saxs.chi2_at_phi(path, 0.4) == 2.0
