import numpy as np

def topsis_from_pareto(X, weights, criterion_types):
    """
    X: matrix of Pareto objective values, shape (n_solutions, n_criteria)
    weights: list or array of criterion weights
    criterion_types: list like ['cost', 'cost', 'benefit']
    """
    X = np.array(X, dtype=float)
    weights = np.array(weights, dtype=float)

    # Step 1: normalize
    denom = np.sqrt((X**2).sum(axis=0))
    R = X / denom

    # Step 2: weighted normalized matrix
    V = R * weights

    # Step 3: ideal best and worst
    A_plus = np.zeros(X.shape[1])
    A_minus = np.zeros(X.shape[1])

    for j in range(X.shape[1]):
        if criterion_types[j] == 'benefit':
            A_plus[j] = np.max(V[:, j])
            A_minus[j] = np.min(V[:, j])
        else:  # cost
            A_plus[j] = np.min(V[:, j])
            A_minus[j] = np.max(V[:, j])

    # Step 4: distances
    D_plus = np.sqrt(((V - A_plus)**2).sum(axis=1))
    D_minus = np.sqrt(((V - A_minus)**2).sum(axis=1))

    # Step 5: closeness coefficient
    CC = D_minus / (D_plus + D_minus)

    # Step 6: best solution
    best_index = np.argmax(CC)

    return best_index, CC, V, A_plus, A_minus