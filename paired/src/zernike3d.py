"""3D Zernike moments and rotation invariants of weighted point clouds (Novotni & Klein 2003).

Definition (Novotni & Klein 2003, ACM Solid Modeling 216-225, doi:10.1145/781606.781639):
  Z_nlm(r, theta, phi) = R_nl(r) Y_lm(theta, phi),  n <= order, (n - l) even, |m| <= l,
  R_nl(r) = sum_{v=0}^{k} q_kl^v r^(2v + l),  k = (n - l) / 2,
  q_kl^v = (-1)^k / 2^(2k) * sqrt((2l + 4k + 3) / 3) * C(2k, k) * (-1)^v
           * C(k, v) * C(2(k + l + v) + 1, 2k) / C(k + l + v, k),
  Omega_nlm = (3 / 4 pi) * sum_i w_i conj(Z_nlm(x_i)),   x_i = (r_i - center) / radius.
Y_lm is scipy.special.sph_harm (orthonormal on the sphere, Condon-Shortley phase). Points with
|x_i| > 1 are dropped and counted. The field is a sum of delta functions at the points (no grid, no
smoothing); a voxelised field is the special case w_i = voxel value * voxel volume (in unit-ball units).

Invariants:
  F_nl = sqrt(sum_m |Omega_nlm|^2)                       (49 per channel at order 12)
  G^{ab}_nl = Re sum_m Omega^a_nlm conj(Omega^b_nlm)     (a < b; SOAP-style cross-channel)
Both are invariant to a common rotation of all channels because Omega_nl. transforms with the
unitary Wigner matrix D^l, identical for every n and every channel.

Scale: Omega here is 1/sqrt(4 pi) times the Omega of Novotni's code and ZMPY3D (NK_CODE_SCALE).
Weights are real, so only m >= 0 is evaluated and Omega_{n,l,-m} = (-1)^m conj(Omega_nlm) fills the rest.
Pure numpy/scipy.
"""
import os
import sys

sys.path.insert(0, os.environ["AEVPLIG_REPO"])

import itertools
from functools import lru_cache
from math import comb, sqrt

import numpy as np
from scipy.special import sph_harm

NORM = 3.0 / (4.0 * np.pi)
# With orthonormal Y_lm, integral_0^1 R_nl^2 r^2 dr = 1/3, so (3/4pi) * integral Z conj(Z) = 1/(4pi).
# Novotni's C++ code and ZMPY3D normalise Y_lm to 4pi on the sphere, which makes their Omega (and every
# invariant) exactly sqrt(4pi) times ours (tests/test_zernike3d.py, ZMPY3D comparison).
NK_CODE_SCALE = float(np.sqrt(4.0 * np.pi))


@lru_cache(maxsize=None)
def nl_index(order):
    """[(n, l)] with n <= order, l <= n, (n - l) even; n ascending, then l ascending."""
    return tuple((n, l) for n in range(order + 1) for l in range(n % 2, n + 1, 2))


def n_invariants(order):
    return len(nl_index(order))


@lru_cache(maxsize=None)
def channel_pairs(n_channels):
    """[(a, b)] with a < b in itertools.combinations order (the row order of cross_invariants)."""
    return tuple(itertools.combinations(range(n_channels), 2))


@lru_cache(maxsize=None)
def radial_coefficients(order):
    """Q[K, order + 1]: R_nl(r) = sum_p Q[k, p] r^p for the k-th (n, l) of nl_index(order)."""
    pairs = nl_index(order)
    q = np.zeros((len(pairs), order + 1))
    for i, (n, l) in enumerate(pairs):
        k = (n - l) // 2
        pre = (-1) ** k / 2.0 ** (2 * k) * sqrt((2 * l + 4 * k + 3) / 3.0) * comb(2 * k, k)
        for v in range(k + 1):
            q[i, 2 * v + l] = (pre * (-1) ** v * comb(k, v) * comb(2 * (k + l + v) + 1, 2 * k)
                               / comb(k + l + v, k))
    q.setflags(write=False)
    return q


def radial(r, order):
    """R[n_points, K] = R_nl(r) for every (n, l) of nl_index(order)."""
    r = np.asarray(r, dtype=float)
    powers = np.empty((r.shape[0], order + 1))
    powers[:, 0] = 1.0
    for p in range(1, order + 1):
        powers[:, p] = powers[:, p - 1] * r
    return powers @ radial_coefficients(order).T


@lru_cache(maxsize=None)
def _lm_nonneg(order):
    ls = np.array([l for l in range(order + 1) for m in range(l + 1)])
    ms = np.array([m for l in range(order + 1) for m in range(l + 1)])
    return ls, ms


@lru_cache(maxsize=None)
def _fill_index(order):
    """Flat index arrays mapping (k = (n, l) row, lm column with m >= 0) to Omega[n, l, m]."""
    kk, nn, ll, mm, col = [], [], [], [], []
    for k, (n, l) in enumerate(nl_index(order)):
        for m in range(l + 1):
            kk.append(k)
            nn.append(n)
            ll.append(l)
            mm.append(m)
            col.append(l * (l + 1) // 2 + m)
    out = tuple(np.array(a) for a in (kk, nn, ll, mm, col))
    neg = out[3] > 0
    return out, neg


def moments(xyz, w, order=12, radius=12.0, center=None, return_outside=False):
    """Complex Omega[C, order + 1, order + 1, 2 * order + 1] indexed [c, n, l, m + order].

    xyz: (n, 3) coordinates; w: (n,) or (n, C) real weights; the cloud is shifted by `center`
    (default: no shift) and divided by `radius`; points with |x| > 1 are dropped. Entries that are not
    valid (n, l, m) triples are 0. With return_outside=True returns (Omega, n_outside).
    """
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        w = w[:, None]
    if w.ndim != 2 or w.shape[0] != xyz.shape[0]:
        raise ValueError(f"weights shape {w.shape} does not match {xyz.shape[0]} points")
    if not np.isfinite(xyz).all() or not np.isfinite(w).all():
        raise ValueError("non-finite coordinates or weights")
    if radius <= 0:
        raise ValueError("radius must be positive")
    n_channels = w.shape[1]
    x = xyz if center is None else xyz - np.asarray(center, dtype=float).reshape(1, 3)
    x = x / float(radius)
    r = np.sqrt((x * x).sum(axis=1))
    inside = r <= 1.0
    n_outside = int((~inside).sum())
    x, r, w = x[inside], r[inside], w[inside]

    omega = np.zeros((n_channels, order + 1, order + 1, 2 * order + 1), dtype=complex)
    if len(r):
        safe = np.where(r > 0, r, 1.0)
        polar = np.arccos(np.clip(x[:, 2] / safe, -1.0, 1.0))
        polar = np.where(r > 0, polar, 0.0)
        azimuth = np.arctan2(x[:, 1], x[:, 0])
        ls, ms = _lm_nonneg(order)
        # Y[n_points, n_lm] for m >= 0 (one vectorised scipy call); conj(Y) = yr - i * yi
        ylm = sph_harm(ms[None, :], ls[None, :], azimuth[:, None], polar[:, None])
        yr = np.ascontiguousarray(ylm.real)   # contiguous operands keep the products in BLAS
        yi = np.ascontiguousarray(ylm.imag)
        rad = radial(r, order)                                          # [n_points, K]
        n_k = rad.shape[1]
        a = (w[:, :, None] * rad[:, None, :]).reshape(len(r), n_channels * n_k)
        # real x complex product as two real GEMMs: B[C * K, n_lm]
        at = np.ascontiguousarray(a.T)
        b = (at @ yr - 1j * (at @ yi)) * NORM
        b = b.reshape(n_channels, n_k, -1)
        (kk, nn, ll, mm, col), neg = _fill_index(order)
        vals = b[:, kk, col]                                            # [C, n_valid_m>=0]
        omega[:, nn, ll, order + mm] = vals
        sign = np.where(mm[neg] % 2 == 0, 1.0, -1.0)
        omega[:, nn[neg], ll[neg], order - mm[neg]] = sign * np.conj(vals[:, neg])
    if return_outside:
        return omega, n_outside
    return omega


def _order_of(omega):
    order = omega.shape[1] - 1
    if omega.ndim != 4 or omega.shape[2] != order + 1 or omega.shape[3] != 2 * order + 1:
        raise ValueError(f"unexpected Omega shape {omega.shape}")
    return order


def invariants(omega):
    """F[C, K] = sqrt(sum_m |Omega_nlm|^2) over the (n, l) of nl_index(order)."""
    order = _order_of(omega)
    pairs = nl_index(order)
    n_idx = np.array([n for n, _ in pairs])
    l_idx = np.array([l for _, l in pairs])
    power = (np.abs(omega) ** 2).sum(axis=3)  # invalid m entries are 0
    return np.sqrt(power[:, n_idx, l_idx])


def cross_invariants(omega):
    """G[P, K] = Re sum_m Omega^a_nlm conj(Omega^b_nlm) for the (a, b) of channel_pairs(C)."""
    order = _order_of(omega)
    pairs = nl_index(order)
    n_idx = np.array([n for n, _ in pairs])
    l_idx = np.array([l for _, l in pairs])
    sel = omega[:, n_idx, l_idx, :]  # [C, K, 2 * order + 1]
    cp = channel_pairs(omega.shape[0])
    if not cp:
        return np.zeros((0, len(pairs)))
    a = np.array([p[0] for p in cp])
    b = np.array([p[1] for p in cp])
    return np.real((sel[a] * np.conj(sel[b])).sum(axis=2))


def descriptor(xyz, w, order=12, radius=12.0, center=None):
    """(F[C, K], G[P, K], n_inside, n_outside) for one weighted cloud."""
    omega, n_out = moments(xyz, w, order=order, radius=radius, center=center, return_outside=True)
    n_in = int(np.asarray(xyz).reshape(-1, 3).shape[0]) - n_out
    return invariants(omega), cross_invariants(omega), n_in, n_out
