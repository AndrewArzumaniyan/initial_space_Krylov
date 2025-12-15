import scipy.io as sio
import urllib.request
import tarfile
import numpy as np
import os
from pathlib import Path

class SuiteSparseLoader:
    BASE_URL = 'https://sparse.tamu.edu/MM'

    def __init__(self, cache_dir='data/suitesparce'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_matrix(self, group, name):
        matrix_dir = self.cache_dir / f"{name}"
        
        if matrix_dir.exists():
            print(f"{name} already cached")
            return self.load_from_cache(matrix_dir)
        
        url = f"{self.BASE_URL}/{group}/{name}.tar.gz"
        tar_path = self.cache_dir / f"{name}.tar.gz"
        
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, tar_path)

        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(self.cache_dir)
        
        os.remove(tar_path)
        
        return self.load_from_cache(matrix_dir)
    
    def load_from_cache(self, matrix_dir):
        mtx_files = list(Path(matrix_dir).glob('*.mtx'))
        if mtx_files:
            A = sio.mmread(mtx_files[0])
            return A.tocsr()
        raise FileNotFoundError(f"No .mtx file in {matrix_dir}")
    
MATRICES = [
    # FEM problems
    ('HB', 'bcsstk14'),           # 1,806 × 1,806
    ('Pothen', 'bodyy4'),         # 17,546 × 17,546
    
    # Laplacians
    ('Newman', 'karate'),         # 34 × 34 (very small, for quick tests)
    
    # PDE problems
    ('GHS_psdef', 'ldoor'),       # 952 × 952
    ('HB', 'gr_30_30'),          # 900 × 900
]

def load_test_matrices():
    loader = SuiteSparseLoader()
    matrices = {}
    
    for group, name in MATRICES:
        try:
            A = loader.download_matrix(group, name)
            
            n = A.shape[0]
            b = np.ones(n)
            
            matrices[f"{group}_{name}"] = {
                'A': A,
                'b': b,
                'size': n,
                'nnz': A.nnz,
                'group': group,
                'name': name
            }
            
            print(f"Loaded {name}: {n}×{n}, nnz={A.nnz}")
            
        except Exception as e:
            print(f"!!! Failed to load {name}: {e}")
    
    return matrices