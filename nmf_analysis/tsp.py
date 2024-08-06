import numpy as np
def reverse_segment(path, n1, n2):
    """Reverse the nodes between n1 and n2.
    """
    q = path.copy()
    if n2 > n1:
        q[n1:(n2+1)] = path[n1:(n2+1)][::-1]
        return q
    else:
        seg = np.hstack((path[n1:], path[:(n2+1)]))[::-1]
        brk = len(q) - n1
        q[n1:] = seg[:brk]
        q[:(n2+1)] = seg[brk:]
        return q

def solve_tsp(dist):
    """Solve travelling salesperson problem (TSP) by two-opt swapping.
    
    Params
    ------
    dist (ndarray) : distance matrix
    
    Returns
    -------
    path (ndarray) : permutation of nodes in graph (rows of dist matrix)
    """

    # number of nodes
    N = dist.shape[0]

    # tsp path for quick calculation of cost
    ii = np.arange(N)
    jj = np.hstack((np.arange(1, N), 0))

    # for each node, a sorted list of closest nodes
    dsort = [np.argsort(d) for d in dist]
    dsort = [d[d != i] for i, d in enumerate(dsort)]

    # randomly initialize path through graph
    path = np.random.permutation(N)
    idx = np.argsort(path)
    cost = np.sum(dist[path[ii], path[jj]])
    
    # keep track of objective function over time
    cost_hist = [cost]

    # optimization loop
    node = 0

    while node < N:

        # we'll try breaking the connection i -> j
        i = path[node]
        j = path[(node+1) % N]
        
        # since we are breaking i -> j we can remove the cost of that connection
        c = cost - dist[i, j]

        # search over nodes k that are closer to j than i
        for k in dsort[j]:
            # can safely continue if dist[i,j] < dist[k,j] for the remaining k
            if k == i:
                node += 1
                break

            # break connection k -> p
            # add connection j -> p
            # add connection i -> k
            p = path[(idx[k]+1) % N]
            new_cost = c - dist[k,p] + dist[j,p] + dist[i,k]

            # if this swap improves the cost, implement it and move to next i
            if new_cost < cost:
                path = reverse_segment(path, idx[j], idx[k])
                idx = np.argsort(path)
                # make sure that we didn't screw up
                assert np.abs(np.sum(dist[path[ii], path[jj]]) - new_cost) < 1e-6
                cost = new_cost
                # restart from the begining of the graph
                cost_hist.append(cost)
                node = 0
                break

    return path, cost_hist
    