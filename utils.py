import numpy as np
from pulp import *

def valid_array(a):
    return np.all(np.isfinite(a))

def random_MRP(state_num, discrete_numbers=4):
    # Probability distribution
    if discrete_numbers:
        P = np.random.randint(0, discrete_numbers, size=(state_num, state_num)) * 1.
    else:
        P = np.random.uniform(0,1, (state_num, state_num))
    for i in range(state_num):
      if np.sum(P[i,:]) > 0:
        P[i,:] = P[i,:] / np.sum(P[i,:])
      else:
        P[i, 0] = 1.

    # Reward function
    if discrete_numbers:
        R = np.random.randint(0, discrete_numbers, size=(state_num)) * 1.
    else:
        R = np.random.uniform(0,1, (state_num))

    # Stationary distribution
    evals, evecs = np.linalg.eig(P.transpose())
    stotionary_distribution_count = np.sum(np.int64(np.isclose(evals, 1.)))
    max_eval = np.argmax(evals)
    assert np.allclose(evals[max_eval], 1), (evals[0], evals)
    p = evecs[:,max_eval].real
    p /= sum(p)
    assert np.allclose(p, np.matmul(p.transpose(), P))

    assert (p >= 0. ).all(), p
    return P, R, p, stotionary_distribution_count

def Wass(P, Q, cost):
    variable_num = len(P)
    P_vars = np.arange(variable_num)
    Q_vars = np.arange(variable_num)

    prob = LpProblem("Wasserstein",LpMinimize)

    var_pairs = [(x,y) for x in P_vars for y in Q_vars]
    coupling = LpVariable.dicts("Coupling",(P_vars,Q_vars),0,1,LpContinuous)

    prob += lpSum([coupling[x][y]*cost[x][y] for (x,y) in var_pairs]), "Sum of Transporting Costs"

    for i in range(variable_num):
        prob += lpSum([coupling[i][j] for j in range(variable_num)]) <= P[i], "Sum of Flow Out Of %s"%i

    for j in range(variable_num):
        prob += lpSum([coupling[i][j] for i in range(variable_num)]) >= Q[j], "Sum of Flow Into %s"%j

    prob.solve()

    # Get optimal solution
    if LpStatus[prob.status] != "Optimal":
        raise("Optimization failed.")

    sol = np.array([[coupling[i][j].varValue for j in range(variable_num)] for i in range(variable_num)])
    wass = np.sum(sol * cost)
    return wass

def random_distribution(variable_num, discrete_numbers=None):
    if discrete_numbers:
        v = np.random.random_integers(0, discrete_numbers, size=variable_num) * 1.
        if sum(v) == 0:
            v[0] = 1.
    else:
        v = np.random.uniform(0,1, variable_num)
    return v/np.sum(v)

def random_metric(variable_num, discrete_numbers=None):
    # Positivity
    if discrete_numbers:
        d = np.random.random_integers(0, discrete_numbers, size=(variable_num, variable_num)) * 1.
    else:
        d = np.random.uniform(0,10, (variable_num, variable_num))
    # d(x,x) = 0
    for i in range(variable_num):
        d[i,i] = 0
    # Symmetry
    for i in range(variable_num):
        for j in range(i):
            d[j,i] = d[i,j]
    # Triangle
    changing = True
    while changing:
        changing = False
        for i in range(variable_num):
            for j in range(variable_num):
                for k in range(variable_num):
                    if d[i,j] > d[i,k] + d[k, j]:
                        d[i,j] = d[i,k] + d[k, j]
                        changing = True
    return d

def weighted_l2(x, w):
    assert x.shape == w.shape
    assert valid_array(x)
    assert valid_array(w)
    assert (w >= 0 ).all(), w
    res =  np.sqrt(np.sum(w * np.square(x)))
    assert valid_array(res)
    return res

def weighted_l1(x, w):
    assert x.shape == w.shape
    assert valid_array(x)
    assert valid_array(w)
    res = np.sum(w * np.abs(x))
    assert valid_array(res)
    return res

def check_distribution(P):
    variable_num = len(P)
    mass = sum(P)
    assert np.allclose(mass, 1.), "sum(P) is %f, should be %f" %(mass, 1.)
    for i in range(variable_num):
        assert P[i] >= 0.


def check_metric(d):
    variable_num = d.shape[0]
    for i in range(variable_num):
        assert d[i,i] == 0

    for i in range(variable_num):
        for j in range(i):
            assert d[i,j] >= 0
            assert d[j,i] == d[i,j]

    for i in range(variable_num):
        for j in range(variable_num):
            for k in range(variable_num):
                assert d[i,j] <= d[i,k] + d[k,j], "%f < %f"%(d[i,j], d[i,k] + d[k,j])

def check_cost(c):
    variable_num = c.shape[0]
    for i in range(variable_num):
        assert c[i,i] == 0

    for i in range(variable_num):
        for j in range(i):
            assert c[i,j] >= 0
            assert c[j,i] == c[i,j]
