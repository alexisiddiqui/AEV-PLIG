"""3D Zernike moments: rotation invariance, origin point, translation, ZMPY3D cross-check, timing."""
import os
import pickle
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.special import sph_harm

import zernike3d as zk


def _cloud(n=300, channels=3, seed=0, scale=3.0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n, 3)) * scale
    w = rng.normal(size=(n, channels))
    return xyz, w


def test_invariant_counts_and_shapes():
    assert zk.n_invariants(12) == 49
    assert zk.n_invariants(8) == 25
    assert zk.n_invariants(15) == 72
    assert zk.n_invariants(20) == 121
    xyz, w = _cloud(channels=7)
    om = zk.moments(xyz, w, order=12, radius=12.0, center=np.zeros(3))
    assert om.shape == (7, 13, 13, 25)
    assert zk.invariants(om).shape == (7, 49)
    assert zk.cross_invariants(om).shape == (21, 49)
    assert zk.channel_pairs(7)[:3] == ((0, 1), (0, 2), (0, 3))
    om1 = zk.moments(xyz, w[:, 0], order=12, radius=12.0)
    assert om1.shape == (1, 13, 13, 25)
    np.testing.assert_allclose(om1[0], om[0], atol=1e-12)


def test_matches_direct_definition():
    """Every (n, l, m), negative m included, equals the defining sum evaluated term by term."""
    xyz, w = _cloud(n=80, channels=1, seed=3)
    order, radius = 6, 10.0
    om, n_out = zk.moments(xyz, w, order=order, radius=radius, return_outside=True)
    x = xyz / radius
    r = np.linalg.norm(x, axis=1)
    keep = r <= 1
    assert n_out == int((~keep).sum())
    x, r, ww = x[keep], r[keep], w[keep, 0]
    polar, azim = np.arccos(x[:, 2] / r), np.arctan2(x[:, 1], x[:, 0])
    rad = zk.radial(r, order)
    worst = 0.0
    for k, (n, l) in enumerate(zk.nl_index(order)):
        for m in range(-l, l + 1):
            ref = 3 / (4 * np.pi) * np.sum(ww * rad[:, k] * np.conj(sph_harm(m, l, azim, polar)))
            worst = max(worst, abs(ref - om[0, n, l, m + order]))
    assert worst < 1e-12


def test_rotation_invariance():
    """(a) 3 channels, 20 random rotations about the centre: F and G identical to 1e-8."""
    rng = np.random.default_rng(1)
    direction = rng.normal(size=(400, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    center = np.array([1.0, -2.0, 0.5])
    xyz = direction * 11.5 * rng.uniform(size=(400, 1)) ** (1 / 3) + center  # inside the 12 A ball
    w = rng.normal(size=(400, 3))
    om = zk.moments(xyz, w, order=12, radius=12.0, center=center)
    f0, g0 = zk.invariants(om), zk.cross_invariants(om)
    worst = 0.0
    for seed in range(20):
        rot = Rotation.random(random_state=seed).as_matrix()
        moved = (xyz - center) @ rot.T + center
        om_r, n_out = zk.moments(moved, w, order=12, radius=12.0, center=center, return_outside=True)
        assert n_out == 0
        worst = max(worst, np.abs(zk.invariants(om_r) - f0).max(), np.abs(zk.cross_invariants(om_r) - g0).max())
    print(f"rotation invariance max abs deviation {worst:.2e} (max |F| {np.abs(f0).max():.2f})")
    assert worst < 1e-8


def test_single_point_at_origin_is_l0_only():
    """(b) a point at the origin has non-zero moments only for l = 0."""
    om = zk.moments(np.zeros((1, 3)), np.ones(1), order=12, radius=12.0)
    for n in range(13):
        for l in range(13):
            if l > 0:
                assert np.all(om[0, n, l] == 0)
    f = zk.invariants(om)[0]
    pairs = zk.nl_index(12)
    for k, (n, l) in enumerate(pairs):
        if l == 0:
            expected = 3 / (4 * np.pi) * abs(zk.radial(np.zeros(1), 12)[0, k]) / np.sqrt(4 * np.pi)
            assert f[k] > 0
            assert f[k] == pytest.approx(expected, rel=1e-12)
        else:
            assert f[k] == 0


def test_translation_sensitivity_and_center():
    """(c) a shifted cloud changes F; passing the shift as `center` restores it."""
    xyz, w = _cloud(n=300, channels=2, seed=2, scale=2.5)
    shift = np.array([2.0, -1.0, 3.0])
    f0 = zk.invariants(zk.moments(xyz, w, order=12, radius=12.0))
    f_shift = zk.invariants(zk.moments(xyz + shift, w, order=12, radius=12.0))
    f_back = zk.invariants(zk.moments(xyz + shift, w, order=12, radius=12.0, center=shift))
    assert np.abs(f_shift - f0).max() > 1e-3 * np.abs(f0).max()
    np.testing.assert_allclose(f_back, f0, atol=1e-10)


def test_points_outside_dropped_and_counted():
    xyz = np.array([[0.0, 0.0, 0.0], [11.9, 0, 0], [12.1, 0, 0], [0, -30.0, 0]])
    om_all, n_out = zk.moments(xyz, np.ones(4), order=4, radius=12.0, return_outside=True)
    om_in = zk.moments(xyz[:2], np.ones(2), order=4, radius=12.0)
    assert n_out == 2
    np.testing.assert_allclose(om_all, om_in, atol=1e-14)


# ---- (d) ZMPY3D cross-check (GPL package, used only here; skipped when absent)

def _zmpy3d_invariants(voxels, order):
    """ZMPY3D's exact per-voxel moments of a voxel grid spanning [-1, 1]^3, reduced to F_nl (n <= order)."""
    zm = pytest.importorskip("ZMPY3D")
    cache = os.path.join(os.path.dirname(zm.__file__), "cache_data", "LogG_CLMCache_MaxOrder20.pkl")
    with open(cache, "rb") as fh:  # the installed package's own cache file
        c = pickle.load(fh)
    n_grid = voxels.shape[0]
    samples = zm.get_bbox_moment_xyz_sample([n_grid / 2] * 3, n_grid / 2, voxels.shape)
    _, _, bbox = zm.calculate_bbox_moment(voxels, 20, samples)
    scaled, _ = zm.calculate_bbox_moment_2_zm(20, c["GCache_complex"], c["GCache_pqr_linear"],
                                              c["GCache_complex_index"], c["CLMCache3D"], bbox)
    scaled = np.nan_to_num(scaled)  # ZMPY3D stores m >= 0 only; Omega_{-m} = (-1)^m conj(Omega_m)
    return np.array([np.sqrt(abs(scaled[n, l, 0]) ** 2 + 2 * np.sum(abs(scaled[n, l, 1:l + 1]) ** 2))
                     for n, l in zk.nl_index(order)])


def _shapes(n_grid):
    c = (np.arange(n_grid) + 0.5) / n_grid * 2 - 1
    gx, gy, gz = np.meshgrid(c, c, c, indexing="ij")
    centres = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return centres, {"sphere": np.linalg.norm(centres, axis=1) <= 0.9,
                     "cube": np.abs(centres).max(axis=1) <= 0.5}


def _compare(f_ours, f_ref):
    big = f_ref > 1e-3 * f_ref.max()
    rel = np.abs(f_ours[big] - f_ref[big]) / f_ref[big]
    return rel.max(), np.abs(f_ours - f_ref).max() / f_ref.max()


def test_zmpy3d_exact_quadrature():
    """Same voxel field, each voxel integrated exactly (5-point Gauss-Legendre per axis is exact for the
    degree-8 Zernike polynomials): our F x sqrt(4 pi) must equal ZMPY3D's to rounding."""
    pytest.importorskip("ZMPY3D")
    n_grid, order = 16, 8
    h = 2.0 / n_grid
    nodes, weights = np.polynomial.legendre.leggauss(5)
    sub = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), -1).reshape(-1, 3) * (h / 2)
    sub_w = np.einsum("i,j,k->ijk", weights, weights, weights).ravel() * (h / 2) ** 3
    centres, shapes = _shapes(n_grid)
    for name, mask in shapes.items():
        f_ref = _zmpy3d_invariants(mask.reshape((n_grid,) * 3).astype(float), order)
        pts = (centres[mask][:, None, :] + sub[None]).reshape(-1, 3)
        w = np.tile(sub_w, int(mask.sum()))
        f = zk.NK_CODE_SCALE * zk.invariants(zk.moments(pts, w, order=order, radius=1.0))[0]
        rel, absmax = _compare(f, f_ref)
        print(f"ZMPY3D exact quadrature {name}: max rel {rel:.2e}, max abs / max F {absmax:.2e}")
        assert rel < 1e-8


def test_zmpy3d_voxel_centres():
    """Voxel centres as points (midpoint rule) vs ZMPY3D's exact voxel integrals: the O(h^2) quadrature gap."""
    pytest.importorskip("ZMPY3D")
    n_grid, order = 64, 8
    vol = (2.0 / n_grid) ** 3
    centres, shapes = _shapes(n_grid)
    for name, mask in shapes.items():
        f_ref = _zmpy3d_invariants(mask.reshape((n_grid,) * 3).astype(float), order)
        f = zk.NK_CODE_SCALE * zk.invariants(
            zk.moments(centres[mask], np.full(int(mask.sum()), vol), order=order, radius=1.0))[0]
        rel, absmax = _compare(f, f_ref)
        print(f"ZMPY3D voxel centres (64^3) {name}: max rel {rel:.2e}, max abs / max F {absmax:.2e}")
        assert rel < 2.5e-2
        assert absmax < 5e-3


def test_timing_1500_points_7_channels():
    """(e) 1,500 points x 7 channels at order 12 in < 50 ms (best of 7 after a warm-up)."""
    xyz, w = _cloud(n=1500, channels=7, seed=4, scale=4.0)
    zk.moments(xyz, w, order=12, radius=12.0, center=np.zeros(3))
    best = np.inf
    for _ in range(7):
        t0 = time.perf_counter()
        om = zk.moments(xyz, w, order=12, radius=12.0, center=np.zeros(3))
        zk.invariants(om)
        zk.cross_invariants(om)
        best = min(best, time.perf_counter() - t0)
    print(f"1500 points x 7 channels, order 12: {best * 1e3:.1f} ms")
    assert best < 0.050
