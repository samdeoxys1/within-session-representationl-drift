import numpy as np

def svd_sign_flip(X):
    """
    SVD with corrected signs
    
    Bro, R., Acar, E., & Kolda, T. G. (2008). Resolving the sign ambiguity in the singular value decomposition.
    Journal of Chemometrics: A Journal of the Chemometrics Society, 22(2), 135-140.
    URL: https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2007/076422.pdf
    """
    # SDV dimensions:
    # U, S, V = np.linalg.svd(X, full_matrices=False)
    # X = U @ diag(S) @ V
    # (I,J) = (I,K) @ (K,K) @ (K,J)
    
    U, S, V = np.linalg.svd(X, full_matrices=False)
    
    I = U.shape[0]
    J = V.shape[1]
    K = S.shape[0]
    
    assert U.shape == (I,K)
    assert V.shape == (K,J)
    assert X.shape == (I,J)
    
    s = {'left': np.zeros(K), 'right': np.zeros(K)}

    for k in range(K):
        mask = np.ones(K).astype(bool)
        mask[k] = False
        # (I,J) = (I,K-1) @ (K-1,K-1) @ (K-1,J)
        Y = X - (U[:, mask] @ np.diag(S[mask]) @ V[mask, :])

        for j in range(J):
            d = np.dot(U[:, k], Y[:, j])
            s['left'][k] += np.sum(np.sign(d) * d**2)
        for i in range(I):
            d = np.dot(V[k, :], Y[i, :])
            s['right'][k] += np.sum(np.sign(d) * d**2)

    for k in range(K):
        if (s['left'][k] * s['right'][k]) < 0:
            if np.abs(s['left'][k]) < np.abs(s['right'][k]):
                s['left'][k] = -s['left'][k]
            else:
                s['right'][k] = -s['right'][k]
        U[:, k] = U[:, k] * np.sign(s['left'][k])
        V[k, :] = V[k, :] * np.sign(s['right'][k])
        
    return U, S, V