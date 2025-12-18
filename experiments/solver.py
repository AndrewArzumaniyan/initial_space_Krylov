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
from preconditioners import BasePreconditioner
from linear_operators import PreconditionedLinearOperator, DeflatedPreconditionedLinearOperator
warnings.filterwarnings('ignore')


def standard_gmres(A: csr_matrix,
                   b: np.ndarray,
                   tol: float = 1e-7,
                   maxiter: int = 5000,
                   callback: Callable = None) -> Tuple[np.ndarray, dict]:
    """Standard GMRES without preconditioner."""
    info = {'method': 'Standard GMRES'}
    t_start = time.time()
    x, exit_code = gmres(A, b, rtol=tol, maxiter=maxiter, callback=callback, atol=0)
    info['time'] = time.time() - t_start
    info['exit_code'] = exit_code
    return x, info


def standard_gmres_with_precond(A: csr_matrix,
                                  b: np.ndarray,
                                  precond: BasePreconditioner,
                                  tol: float = 1e-7,
                                  maxiter: int = 5000,
                                  callback: Callable = None) -> Tuple[np.ndarray, dict]:
    """Standard GMRES with preconditioner: solve M^{-1}Ax = M^{-1}b."""
    info = {'method': f'GMRES + {precond.__class__.__name__}'}
    t_start = time.time()

    # Apply preconditioner
    A_precond = PreconditionedLinearOperator(A, precond)
    b_precond = precond.apply(b)

    x, exit_code = gmres(A_precond, b_precond, rtol=tol, maxiter=maxiter,
                         callback=callback, atol=0)
    info['time'] = time.time() - t_start
    info['exit_code'] = exit_code
    return x, info


def deflated_gmres_no_precond(A: csr_matrix,
                                b: np.ndarray,
                                Y_n: np.ndarray,
                                tol: float = 1e-7,
                                maxiter: int = 5000,
                                callback: Callable = None) -> Tuple[np.ndarray, dict]:
    """Deflated GMRES without preconditioner (baseline deflated method)."""
    info = {'method': 'Deflated GMRES (no precond)'}
    t_start = time.time()

    # QR decomposition
    AY_n = A @ Y_n
    C_n, R = np.linalg.qr(AY_n)

    # Project and deflate
    r_0 = b.copy()
    z_1 = Y_n @ np.linalg.solve(R, C_n.T @ r_0)
    r_1 = r_0 - C_n @ (C_n.T @ r_0)

    # Deflated operator
    class DeflatedOp(LinearOperator):
        def __init__(self, A, C_n):
            self.A = A
            self.C_n = C_n
            self.shape = A.shape
            self.dtype = A.dtype

        def _matvec(self, x):
            Ax = self.A @ x
            return Ax - self.C_n @ (self.C_n.T @ Ax)

    A_deflated = DeflatedOp(A, C_n)
    z_2, exit_code = gmres(A_deflated, r_1, rtol=tol, maxiter=maxiter,
                           callback=callback, atol=0)

    x = z_1 + z_2
    info['time'] = time.time() - t_start
    info['exit_code'] = exit_code
    return x, info


def deflated_gmres_precond_before(A: csr_matrix,
                                    b: np.ndarray,
                                    Y_n: np.ndarray,
                                    precond: BasePreconditioner,
                                    tol: float = 1e-7,
                                    maxiter: int = 5000,
                                    callback: Callable = None) -> Tuple[np.ndarray, dict]:
    """
    Preconditioner BEFORE model: Jacobi/BJacobi/SOR → Model → Deflated GMRES.

    Algorithm:
    1. Apply preconditioner to get M^{-1}A and M^{-1}b
    2. Use Y_n predicted for M^{-1}A
    3. QR: M^{-1}A Y_n = C_n R
    4. Deflated GMRES on M^{-1}A
    """
    info = {'method': f'{precond.__class__.__name__} → Model → Deflated'}
    t_start = time.time()

    # Apply preconditioner to b
    b_precond = precond.apply(b)

    # QR decomposition: M^{-1}A Y_n = C_n R
    MA_Y_n = precond.apply_to_matrix(A @ Y_n)
    C_n, R = np.linalg.qr(MA_Y_n)

    # Project and deflate
    r_0 = b_precond.copy()
    z_1 = Y_n @ np.linalg.solve(R, C_n.T @ r_0)
    r_1 = r_0 - C_n @ (C_n.T @ r_0)

    # Deflated operator: (I - C_n C_n^*) M^{-1} A
    class DeflatedPrecondOp(LinearOperator):
        def __init__(self, A, C_n, precond):
            self.A = A
            self.C_n = C_n
            self.precond = precond
            self.shape = A.shape
            self.dtype = A.dtype

        def _matvec(self, x):
            Ax = self.A @ x
            MAx = self.precond.apply(Ax)
            return MAx - self.C_n @ (self.C_n.T @ MAx)

    A_deflated = DeflatedPrecondOp(A, C_n, precond)
    z_2, exit_code = gmres(A_deflated, r_1, rtol=tol, maxiter=maxiter,
                           callback=callback, atol=0)

    x = z_1 + z_2
    info['time'] = time.time() - t_start
    info['exit_code'] = exit_code
    return x, info


def deflated_gmres_precond_after(A: csr_matrix,
                                   b: np.ndarray,
                                   Y_n: np.ndarray,
                                   precond: BasePreconditioner,
                                   tol: float = 1e-7,
                                   maxiter: int = 5000,
                                   callback: Callable = None) -> Tuple[np.ndarray, dict]:
    """
    Preconditioner AFTER model: Model → Jacobi/BJacobi/SOR → Deflated GMRES.

    Algorithm:
    1. Use Y_n predicted for original A
    2. QR: A Y_n = C_n R (no preconditioner yet)
    3. Project and deflate on original system
    4. Apply preconditioner: solve M^{-1}(I - C_n C_n^*)A z_2 = r_1
    """
    info = {'method': f'Model → {precond.__class__.__name__} → Deflated'}
    t_start = time.time()

    # QR decomposition (without preconditioner)
    AY_n = A @ Y_n
    C_n, R = np.linalg.qr(AY_n)

    # Project and deflate (on original system)
    r_0 = b.copy()
    z_1 = Y_n @ np.linalg.solve(R, C_n.T @ r_0)
    r_1 = r_0 - C_n @ (C_n.T @ r_0)

    # Deflated + preconditioned operator: M^{-1}(I - C_n C_n^*)A
    A_deflated_precond = DeflatedPreconditionedLinearOperator(A, C_n, precond)

    z_2, exit_code = gmres(A_deflated_precond, r_1, rtol=tol, maxiter=maxiter,
                           callback=callback, atol=0)

    x = z_1 + z_2
    info['time'] = time.time() - t_start
    info['exit_code'] = exit_code
    return x, info