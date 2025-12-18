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


class BasePreconditioner:
    """Base class for preconditioners."""

    def __init__(self, A: csr_matrix):
        self.A = A
        self.n = A.shape[0]
        self._setup()

    def _setup(self):
        """Setup preconditioner (compute M or M^{-1})."""
        raise NotImplementedError

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply M^{-1} to vector x."""
        raise NotImplementedError

    def apply_to_matrix(self, B: np.ndarray) -> np.ndarray:
        """Apply M^{-1} to matrix B (column-wise)."""
        return np.column_stack([self.apply(B[:, i]) for i in range(B.shape[1])])


class JacobiPreconditioner(BasePreconditioner):
    """Jacobi (diagonal) preconditioner: M = diag(A)."""

    def _setup(self):
        diag_vals = self.A.diagonal()
        # Avoid division by zero
        diag_vals = np.where(np.abs(diag_vals) < 1e-14, 1.0, diag_vals)
        self.M_inv_diag = 1.0 / diag_vals

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.M_inv_diag * x


class BlockJacobiPreconditioner(BasePreconditioner):
    """Block Jacobi preconditioner."""

    def __init__(self, A: csr_matrix, block_size: int = 300):
        self.block_size = block_size
        super().__init__(A)

    def _setup(self):
        """Extract diagonal blocks and compute their inverses."""
        n = self.n
        bs = self.block_size
        num_blocks = (n + bs - 1) // bs  # Ceiling division

        self.blocks_inv = []
        for i in range(num_blocks):
            start = i * bs
            end = min((i + 1) * bs, n)

            # Extract block
            block = self.A[start:end, start:end].toarray()

            # Compute inverse (or pseudo-inverse if singular)
            try:
                block_inv = np.linalg.inv(block)
            except np.linalg.LinAlgError:
                block_inv = np.linalg.pinv(block)

            self.blocks_inv.append((start, end, block_inv))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply block diagonal inverse."""
        result = np.zeros_like(x)
        for start, end, block_inv in self.blocks_inv:
            result[start:end] = block_inv @ x[start:end]
        return result


class SORPreconditioner(BasePreconditioner):
    """SOR preconditioner: M = (D + ωL) where ω=1 (Gauss-Seidel)."""

    def __init__(self, A: csr_matrix, omega: float = 1.0):
        self.omega = omega
        super().__init__(A)

    def _setup(self):
        """Setup M = D + ω*L."""
        D = diags(self.A.diagonal(), 0, format='csr')
        L = tril(self.A, k=-1, format='csr')
        self.M = D + self.omega * L

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Solve M z = x using forward substitution."""
        return spsolve_triangular(self.M, x, lower=True)
