import numpy as np
from scipy import linalg

def rank(A:np.ndarray) -> int:
    return np.linalg.matrix_rank(A)

def ctrb(A:np.ndarray, B:np.ndarray) -> np.ndarray:
    assert A.shape[0] == B.shape[0], f'Dimension of A-matrix and B-matrix must be equal. A:{A.shape[0]}, B:{B.shape[0]}'
    n = A.shape[0]
    An = np.eye(n)
    ctrb = np.copy(B)
    for i in range(n-1):
        ctrb = np.hstack((ctrb,np.dot(An,B)))
        An = An * A
    return np.array(ctrb)

def obsv(A:np.ndarray, C:np.ndarray) -> np.ndarray:
    assert A.shape[0] == C.shape[1], f'Dimension of A-matrix and C-matrix must be equal. A:{A.shape[0]}, B:{C.shape[1]}'
    n = A.shape[0]
    An = np.eye(n)
    obsv = np.copy(C)
    for i in range(n-1):
        ctrb = np.vstack((ctrb,np.dot(C,An)))
        An = An * A
    return np.array(obsv)
