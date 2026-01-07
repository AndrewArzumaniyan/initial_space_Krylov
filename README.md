# initial_space_Krylov

Python=3.12.12

# Neural Krylov Iteration for Accelerating Linear System Solving

## About

This repository contains implementation and experiments based on the paper:  
**"Neural Krylov Iteration for Accelerating Linear System Solving"** (NeurIPS 2024)  
[Original paper link](https://proceedings.neurips.cc/paper_files/paper/2024/file/e88870ec82f2469b0ddf32c817920c68-Paper-Conference.pdf)

### Core Idea

The project investigates using Neural Fourier Operator (FNO) to accelerate convergence of the GMRES iterative method for solving large sparse linear systems arising from PDE discretization. The neural network learns to identify the "bad" invariant subspace of the matrix, enabling GMRES to work more efficiently on the remaining "good" subspace.

## Repository Structure
```
.
├── data/                  # Data generation code
├── experiments/           # Experiments and results in .ipynb files
├── train_model.ipynb      # Model training notebook
└── README.md
```

## Key Results

Based on our experiments, we obtained the following findings:

1. **FNO successfully approximates the invariant subspace** for Laplace PDE matrices

2. **Significant GMRES convergence acceleration**: Finding the "bad" invariant subspace using the neural network accelerates convergence both with and without preconditioning

3. **Critical importance of operation ordering**:
   - **Model→Precond→GMRES** configuration shows substantial speedup (up to 7.56× in iterations)
   - **Precond→Model→GMRES** configuration demonstrates lower efficiency

4. **Applying preconditioner after NeuroKitt** tends to worsen the convergence of subsequent GMRES

### Performance Comparison

| Configuration | None | Jacobi | Block-Jacobi | SOR |
|--------------|------|--------|--------------|-----|
| Precond→Model | 1.69/2.28 | 2.16/3.31 | 4.56/4.95 | 4.96/5.45 |
| Model→Precond | 1.69/2.28 | **0.72/0.97** | **0.61/0.64** | **0.87/0.95** |
| Speedup | 1.00× | **3.41×** | **7.84×** | **5.74×** |

*(Time/Iterations)*

## Implementation Details

- **Only Laplace matrices** were used for training and validation
- Code was rewritten from **OpenMP to sequential** version for local execution
- All experiments are reproducible and documented in notebooks
- Model training code is located in `train_model.ipynb` in the repository root
- Data generation code is in the `data/` folder
- Experiments and results are available in the `experiments/` folder

## Summary

1. FNO can successfully approximate the invariant subspace for Laplace PDE matrices
2. Finding "bad" invariant subspace accelerates the convergence of GMRES with and without preconditioning
3. Reordering of NeuroKitt and Preconditioning drastically influences the convergence
4. Preconditioning after NeuroKitt tends to worsen the convergence of following GMRES

---

**Skoltech, 2025**
