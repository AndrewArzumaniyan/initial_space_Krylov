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
warnings.filterwarnings('ignore')

class PreconditionedLinearOperator(LinearOperator):
    """Linear operator for M^{-1}A."""

    def __init__(self, A: csr_matrix, precond: BasePreconditioner):
        self.A = A
        self.precond = precond
        self.shape = A.shape
        self.dtype = A.dtype

    def _matvec(self, x):
        return self.precond.apply(self.A @ x)

    def _rmatvec(self, x):
        return self.A.T @ self.precond.apply(x)


class DeflatedPreconditionedLinearOperator(LinearOperator):
    """Linear operator for M^{-1}(I - C_n C_n^*)A."""

    def __init__(self, A: csr_matrix, C_n: np.ndarray, precond: BasePreconditioner):
        self.A = A
        self.C_n = C_n
        self.precond = precond
        self.shape = A.shape
        self.dtype = A.dtype

    def _matvec(self, x):
        """Compute M^{-1}(I - C_n C_n^*)A x."""
        Ax = self.A @ x
        deflated_Ax = Ax - self.C_n @ (self.C_n.T @ Ax)
        return self.precond.apply(deflated_Ax)

    def _rmatvec(self, x):
        """Compute A^*(I - C_n C_n^*)M^{-*} x."""
        Minv_x = self.precond.apply(x)
        deflated = Minv_x - self.C_n @ (self.C_n.T @ Minv_x)
        return self.A.T @ deflated
