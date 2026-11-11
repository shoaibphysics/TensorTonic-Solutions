import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a float.
    """
    A,B=np.array(a),np.array(b)
    na,nb=np.linalg.norm(A),np.linalg.norm(B)
    if na==0 or nb==0:
        return float(0)
    return float(np.dot(A,B)/(na*nb))