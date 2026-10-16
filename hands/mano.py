"""MANO hand model without chumpy: loader, joints-only forward model and fits.

MANO (non-commercial licence) is optional. The official pickles are Python-2
pickles holding chumpy objects and an old scipy sparse class; they are read
with a custom Unpickler that turns those into stubs and then into dense
float64 arrays. The model files are read in place and never copied.

Only the 16 kinematic joints and the 5 fingertip vertices are computed, so a
forward pass is a few small matrix products (no mesh).

Fit residuals: joint error in metres weighted by sqrt(conf) (NaN / conf 0
joints contribute 0), plus a pose prior 1e-3 * (pose[3:] - hands_mean) per
element; the shape fit adds 0.1 * betas. The pose fit uses the analytic
Jacobian of the kinematic chain and tip vertices (forward_joints_jac); the
shape fit (10 betas, few calls) keeps finite differences.
"""
from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix
from scipy.spatial.transform import Rotation

DEFAULT_MANO_DIR = r"D:\Claude Projects\Player One Face Removal\cloud\mano"

# Fingertip vertices: thumb, index, middle, ring, pinky (same for both sides).
TIP_VERTICES = [745, 317, 444, 556, 673]
# MANO joints 0..15 (+ tips 16..20) -> MediaPipe order: J21[MP_FROM_MANO].
MP_FROM_MANO = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

POSE_PRIOR = 1e-3
BETA_PRIOR = 0.1
# trf with looser-than-default tolerances: same sub-0.1 mm accuracy, about half the evaluations.
_LSQ = dict(method="trf", ftol=1e-6, xtol=1e-6, gtol=1e-6)


@dataclass
class ManoModel:
    side: str
    v_template: np.ndarray  # (778, 3)
    shapedirs: np.ndarray  # (778, 3, 10)
    posedirs: np.ndarray  # (778, 3, 135)
    J_regressor: np.ndarray  # (16, 778)
    weights: np.ndarray  # (778, 16)
    kintree_table: np.ndarray  # (2, 16)
    hands_mean: np.ndarray  # (45,)
    hands_components: np.ndarray  # (45, 45)
    f: np.ndarray  # (1538, 3)

    def __post_init__(self):
        # Precomputed pieces used by forward_joints.
        self._parents = self.kintree_table[0].astype(np.int64)
        depth = np.zeros(16, np.int64)
        for k in range(1, 16):
            depth[k] = depth[self._parents[k]] + 1
        # Non-root joints grouped by tree depth, so the chain runs one level at a time.
        self._levels = [np.flatnonzero(depth == d) for d in range(1, depth.max() + 1)]
        self._jtemp = self.J_regressor @ self.v_template  # (16, 3)
        self._jshape = np.einsum("jv,vcb->jcb", self.J_regressor, self.shapedirs)  # (16, 3, 10)
        self._tip_t = self.v_template[TIP_VERTICES]  # (5, 3)
        self._tip_s = self.shapedirs[TIP_VERTICES]  # (5, 3, 10)
        self._tip_p = self.posedirs[TIP_VERTICES]  # (5, 3, 135)
        self._tip_w = self.weights[TIP_VERTICES]  # (5, 16)


class _Stub:
    """Stands in for chumpy and old scipy objects; keeps the pickled state."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state


_NUMPY_MODULES = {"numpy", "numpy.core.multiarray", "numpy.core.numeric", "numpy._core.multiarray", "numpy._core.numeric"}
_NUMPY_NAMES = {"ndarray", "dtype", "_reconstruct", "scalar"}


class _Unpickler(pickle.Unpickler):
    """Constructs only stubs, builtins.set and numpy array reconstruction; anything else is refused."""

    def find_class(self, module, name):
        if module.startswith("chumpy") or module.startswith("scipy.sparse"):
            return _Stub
        if module in ("__builtin__", "builtins") and name == "set":
            return set
        if module in _NUMPY_MODULES and name in _NUMPY_NAMES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"MANO pickle references {module}.{name}, which is not allowed")


def _dense(obj) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    st = obj.state
    if "x" in st:  # chumpy.ch.Ch
        return np.asarray(st["x"])
    if "a" in st and "idxs" in st:  # chumpy.reordering.Select
        return _dense(st["a"]).ravel()[st["idxs"]].reshape(st["preferred_shape"])
    if "indptr" in st:  # scipy.sparse.csc.csc_matrix
        return csc_matrix((st["data"], st["indices"], st["indptr"]), shape=st["_shape"]).toarray()
    raise TypeError(f"unknown pickled object with keys {sorted(st)}")


def load_mano(side: str, mano_dir=None) -> ManoModel:
    d = Path(mano_dir if mano_dir is not None else os.environ.get("MANO_DIR", DEFAULT_MANO_DIR))
    path = d / f"MANO_{side.upper()}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"MANO files not found at {d}")
    with open(path, "rb") as fh:
        raw = _Unpickler(fh, encoding="latin1").load()
    a = {k: np.asarray(_dense(raw[k]), dtype=np.float64) for k in
         ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "hands_mean", "hands_components")}
    return ManoModel(side=side.lower(), kintree_table=np.asarray(raw["kintree_table"]).astype(np.int64),
                     f=np.asarray(raw["f"]), **a)


def rodrigues(r: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle -> (N, 3, 3) rotation matrices."""
    theta = np.linalg.norm(r, axis=1)[:, None]
    k = r / np.maximum(theta, 1e-12)
    kx, ky, kz = k.T
    z = np.zeros_like(kx)
    K = np.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], axis=1).reshape(-1, 3, 3)
    s, c = np.sin(theta)[:, :, None], np.cos(theta)[:, :, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def forward_joints(model: ManoModel, betas, pose, trans) -> np.ndarray:
    """(21, 3) joints in MediaPipe order, metres."""
    betas = np.asarray(betas, dtype=np.float64)
    R = rodrigues(np.asarray(pose, dtype=np.float64).reshape(16, 3))
    J = model._jtemp + model._jshape @ betas  # rest joints (16, 3)
    par = model._parents
    Rg = np.empty((16, 3, 3))
    tg = np.empty((16, 3))
    Rg[0], tg[0] = R[0], J[0]
    for k in model._levels:
        p = par[k]
        Rg[k] = Rg[p] @ R[k]
        tg[k] = np.einsum("kij,kj->ki", Rg[p], J[k] - J[p]) + tg[p]
    # Fingertip vertices: LBS of only those 5 vertices.
    v = model._tip_t + model._tip_s @ betas + model._tip_p @ (R[1:] - np.eye(3)).ravel()  # (5, 3)
    A_R = np.einsum("vk,kij->vij", model._tip_w, Rg)
    A_t = model._tip_w @ (tg - np.einsum("kij,kj->ki", Rg, J))
    tips = np.einsum("vij,vj->vi", A_R, v) + A_t
    return np.concatenate([tg, tips])[MP_FROM_MANO] + np.asarray(trans, dtype=np.float64)


def _skew(v: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3, 3) cross-product matrices."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = np.zeros_like(x)
    return np.stack([o, -z, y, z, o, -x, -y, x, o], axis=-1).reshape(v.shape[:-1] + (3, 3))


def _rodrigues_jac(r: np.ndarray):
    """(N, 3) axis-angle -> R (N, 3, 3) and dR (N, 3, 3, 3) with dR[n, i] = dR[n] / dr[n, i].

    Gallego & Yezzi: dR/dr_i = (r_i [r]x + [r x (I - R) e_i]x) / |r|^2 R; [e_i]x at r = 0.
    """
    R = rodrigues(r)
    th2 = np.sum(r * r, axis=1)
    small = th2 < 1e-16
    cols = np.swapaxes(np.eye(3) - R, 1, 2)  # [n, i] = column i of (I - R)
    M = r[:, :, None, None] * _skew(r)[:, None] + _skew(np.cross(r[:, None, :], cols))
    M = M / np.where(small, 1.0, th2)[:, None, None, None]
    M[small] = _skew(np.eye(3))
    return R, M @ R[:, None]


def forward_joints_jac(model: ManoModel, betas, pose, trans):
    """forward_joints and its Jacobian: (21, 3) joints, (21, 3, 51) d joints / d [pose (48), trans (3)]."""
    betas = np.asarray(betas, dtype=np.float64)
    R, dR = _rodrigues_jac(np.asarray(pose, dtype=np.float64).reshape(16, 3))
    J = model._jtemp + model._jshape @ betas
    par = model._parents
    Rg = np.empty((16, 3, 3))
    tg = np.empty((16, 3))
    dRg = np.zeros((16, 48, 3, 3))  # d Rg[k] / d pose
    dtg = np.zeros((16, 48, 3))
    Rg[0], tg[0] = R[0], J[0]
    dRg[0, 0:3] = dR[0]
    for k in model._levels:
        p = par[k]
        bone = J[k] - J[p]
        Rg[k] = Rg[p] @ R[k]
        tg[k] = np.einsum("kij,kj->ki", Rg[p], bone) + tg[p]
        dRg[k] = dRg[p] @ R[k][:, None]
        dRg[k[:, None], 3 * k[:, None] + np.arange(3)] += Rg[p][:, None] @ dR[k]
        dtg[k] = np.einsum("kpij,kj->kpi", dRg[p], bone) + dtg[p]
    v = model._tip_t + model._tip_s @ betas + model._tip_p @ (R[1:] - np.eye(3)).ravel()  # (5, 3)
    dv = np.zeros((5, 3, 48))
    dv[:, :, 3:] = np.einsum("vckm,kim->vcki", model._tip_p.reshape(5, 3, 15, 9),
                             dR[1:].reshape(15, 3, 9)).reshape(5, 3, 45)
    w = model._tip_w
    A_R = np.einsum("vk,kij->vij", w, Rg)
    A_t = w @ (tg - np.einsum("kij,kj->ki", Rg, J))
    tips = np.einsum("vij,vj->vi", A_R, v) + A_t
    dA_R = np.einsum("vk,kpij->vpij", w, dRg)
    dA_t = np.einsum("vk,kpi->vpi", w, dtg - np.einsum("kpij,kj->kpi", dRg, J))
    dtips = np.einsum("vpij,vj->vpi", dA_R, v) + np.einsum("vij,vjp->vpi", A_R, dv) + dA_t
    jac = np.zeros((21, 3, 51))
    jac[:, :, :48] = np.swapaxes(np.concatenate([dtg, dtips]), 1, 2)[MP_FROM_MANO]
    jac[:, :, 48:] = np.eye(3)
    joints = np.concatenate([tg, tips])[MP_FROM_MANO] + np.asarray(trans, dtype=np.float64)
    return joints, jac


def _mask(joints_obs, conf):
    """Observed joints with unusable ones zeroed, and (21, 1) weights sqrt(conf) (0 if unusable)."""
    w = np.sqrt(np.clip(np.nan_to_num(conf), 0, None))
    ok = np.isfinite(joints_obs).all(axis=1) & (w > 0)
    return np.where(ok[:, None], joints_obs, 0.0), np.where(ok, w, 0.0)[:, None]


def _neutral_start(model, betas, obs, w):
    """hands_mean finger pose; global rot + trans from Procrustes onto the observed joints."""
    pose = np.concatenate([np.zeros(3), model.hands_mean])
    src = forward_joints(model, betas, pose, np.zeros(3))
    ok = w[:, 0] > 0
    if ok.sum() < 3:
        c = obs[ok].mean(0) if ok.any() else np.zeros(3)
        return pose, c - src[0]
    a, b = src[ok], obs[ok]
    ma, mb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    t = mb - Rm @ ma
    pose[:3] = Rotation.from_matrix(Rm).as_rotvec()
    # The root rotation acts about the wrist J0: R (x - J0) + J0 + trans == R x + t.
    return pose, t + Rm @ src[0] - src[0]


def _pose_residual(x, model, betas, obs, w):
    pred = forward_joints(model, betas, x[:48], x[48:])
    return np.concatenate([((pred - obs) * w).ravel(), POSE_PRIOR * (x[3:48] - model.hands_mean)])


_PRIOR_JAC = np.hstack([np.zeros((45, 3)), POSE_PRIOR * np.eye(45), np.zeros((45, 3))])


def _pose_jac(x, model, betas, obs, w):
    _, jac = forward_joints_jac(model, betas, x[:48], x[48:])
    return np.vstack([(jac * w[:, :, None]).reshape(63, 51), _PRIOR_JAC])


def fit_pose(model, betas, joints_obs, conf, init=None):
    """One frame -> (pose (48,), trans (3,), rmse_m over the used joints)."""
    betas = np.asarray(betas, dtype=np.float64)
    obs, w = _mask(np.asarray(joints_obs, dtype=np.float64), np.asarray(conf, dtype=np.float64))
    pose0, trans0 = _neutral_start(model, betas, obs, w) if init is None else init
    x0 = np.concatenate([pose0, trans0])
    x = least_squares(_pose_residual, x0, jac=_pose_jac, args=(model, betas, obs, w), **_LSQ).x
    pose, trans = x[:48], x[48:]
    ok = w[:, 0] > 0
    err = np.linalg.norm(forward_joints(model, betas, pose, trans)[ok] - obs[ok], axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2))) if ok.any() else float("nan")
    return pose, trans, rmse


def _fit_betas(model, poses, trans, obs, w, betas0):
    def resid(b):
        r = [((forward_joints(model, b, poses[i], trans[i]) - obs[i]) * w[i]).ravel() for i in range(len(obs))]
        return np.concatenate(r + [BETA_PRIOR * b])

    return least_squares(resid, betas0, **_LSQ).x


def fit_shape(model, joints_obs, conf, init_poses=None, rounds: int = 3):
    """K frames -> (betas (10,), poses (K, 48), trans (K, 3)) by alternating pose and shape fits."""
    K = len(joints_obs)
    ow = [_mask(np.asarray(joints_obs[i], dtype=np.float64), np.asarray(conf[i], dtype=np.float64)) for i in range(K)]
    obs = np.stack([o for o, _ in ow])
    w = np.stack([x for _, x in ow])
    betas = np.zeros(10)
    poses = np.zeros((K, 48))
    trans = np.zeros((K, 3))
    for r in range(rounds):
        for i in range(K):
            init = (poses[i], trans[i]) if r else None
            if r == 0 and init_poses is not None:
                init = (np.asarray(init_poses[i], dtype=np.float64), _neutral_start(model, betas, obs[i], w[i])[1])
            poses[i], trans[i], _ = fit_pose(model, betas, joints_obs[i], conf[i], init)
        betas = _fit_betas(model, poses, trans, obs, w, betas)
    return betas, poses, trans


def fit_sequence(model, joints, conf, valid, n_shape_frames: int = 30) -> dict:
    """Shape from the best spread-out valid frames, then warm-started per-frame pose fits."""
    t0 = time.perf_counter()
    T = len(joints)
    valid = np.asarray(valid, dtype=bool)
    idx = np.flatnonzero(valid)
    out = {"betas": np.zeros(10), "pose": np.full((T, 48), np.nan), "trans": np.full((T, 3), np.nan),
           "rmse_m": np.full(T, np.nan), "valid": valid.copy()}
    if len(idx):
        mc = np.nan_to_num(np.nanmean(np.where(np.isfinite(conf[idx]), conf[idx], np.nan), axis=1), nan=-1.0)
        chunks = np.array_split(np.arange(len(idx)), min(n_shape_frames, len(idx)))
        pick = [idx[c[np.argmax(mc[c])]] for c in chunks]
        out["betas"] = fit_shape(model, joints[pick], conf[pick])[0]
        prev, last = None, None
        for t in idx:
            if last is not None and t - last - 1 > 10:
                prev = None
            pose, trans, rmse = fit_pose(model, out["betas"], joints[t], conf[t], prev)
            out["pose"][t], out["trans"][t], out["rmse_m"][t] = pose, trans, rmse
            prev, last = (pose, trans), t
    out["seconds"] = time.perf_counter() - t0
    return out
