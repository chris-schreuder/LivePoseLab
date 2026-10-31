"""Multi-view DLT triangulation (same equations as DLT_3_cameras / DLT_2_cameras in the repo).

For every point, each camera c with projection matrix P_c and pixel (u, v) adds two rows

    v * P_c[2] - P_c[1]
    P_c[0] - u * P_c[2]

to A; the 3D point is the last right-singular vector of A^T A, de-homogenised. This
version solves all points of a frame in one batched SVD instead of a Python loop.
"""
import numpy as np


def triangulate(points_2d, P):
    """points_2d: (C, N, 2) pixels for C cameras; P: (C, 3, 4). Returns (N, 3)."""
    u = points_2d[..., 0][..., None]                       # (C, N, 1)
    v = points_2d[..., 1][..., None]
    rows1 = v * P[:, None, 2] - P[:, None, 1]              # (C, N, 4)
    rows2 = P[:, None, 0] - u * P[:, None, 2]
    A = np.concatenate([rows1, rows2], axis=0).transpose(1, 0, 2)   # (N, 2C, 4)
    B = A.transpose(0, 2, 1) @ A                           # (N, 4, 4), as in the repo
    _, _, Vh = np.linalg.svd(B)
    X = Vh[:, -1]
    return X[:, :3] / X[:, 3:4]


def project(points_3d, P):
    """points_3d: (N, 3); P: (C, 3, 4). Returns (C, N, 2) pixels."""
    Xh = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    x = np.einsum("cij,nj->cni", P, Xh)
    return x[..., :2] / x[..., 2:3]


def reprojection_error(points_2d, points_3d, P):
    """Mean pixel distance per camera between observations and reprojected 3D points, (C,)."""
    return np.linalg.norm(project(points_3d, P) - points_2d, axis=-1).mean(axis=1)
