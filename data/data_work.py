import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import diags, csr_matrix, eye, tril
from scipy.sparse.linalg import gmres, LinearOperator, spsolve_triangular
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable, Optional, Dict
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


def poisson_rhs_manufactured(nx: int, ny: int, scale_with_h: bool = False) -> np.ndarray:
    """Generate RHS for 2D Poisson equation using manufactured solution."""
    hx = 1.0 / (nx + 1)
    hy = 1.0 / (ny + 1)
    x = np.linspace(hx, 1.0 - hx, nx)
    y = np.linspace(hy, 1.0 - hy, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = 2.0 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    if scale_with_h:
        f = (hx * hy) * f
    return f.reshape(-1)

def build_matrix(coef: np.ndarray, P: np.ndarray) -> csr_matrix:
    """
    Build sparse matrix A from coefficients.

    Parameters:
    -----------
    coef : array (K, K) - coefficient matrix
    P : array (s, s) - potential matrix

    Returns:
    --------
    A : sparse matrix (s^2, s^2)
    """
    K = coef.shape[0]
    s = K - 2

    P = P.reshape(s*s, order='C')

    diag_list = []
    off_diag_list = []

    for j in range(1, K-1):
        diag_values = np.array([
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]), [0])),
            0.5 * (coef[0:K-2, j] + coef[1:K-1, j]) + 0.5 * (coef[2:K, j] + coef[1:K-1, j]) + \
            0.5 * (coef[1:K-1, j-1] + coef[1:K-1, j]) + 0.5 * (coef[1:K-1, j+1] + coef[1:K-1, j]),
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]), [0]))
        ])
        diag_list.append(diag_values)

        if j != K-2:
            off_diag = -0.5 * (coef[1:K-1, j] + coef[1:K-1, j+1])
            off_diag_list.append(off_diag)

    diag_output = np.concatenate(diag_list, axis=1)
    off_diag_output = np.concatenate(off_diag_list, axis=0)

    A = (diags(diag_output, [-1,0,1], (s**2, s**2)) +
         diags((off_diag_output, off_diag_output), [-(K-2), (K-2)], (s**2, s**2))) * (K-1)**2 + \
         diags(P, 0, (s**2, s**2))

    return csr_matrix(A)

def extract_params_from_data(params: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract w and p from params tensor.

    Parameters:
    -----------
    params : torch.Tensor, shape (s, s, 2) where last dim is [w, p]

    Returns:
    --------
    w : np.ndarray (s, s) - coefficient matrix
    p : np.ndarray (s-2, s-2) - potential matrix
    """
    if torch.is_tensor(params):
        params = params.cpu().numpy()

    w = params[:, :, 0]  # First channel
    p = params[1:-1, 1:-1, 1]  # Second channel, interior points

    return w, p