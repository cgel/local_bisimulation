import numpy as np
from utils import Wass, random_MRP, random_metric, weighted_l2, valid_array
from multiprocessing import Process

def find_counter():
    np.random.seed()
    state_num = 4
    discount = 0.999
    discrete_variables = None

    def F(d, P, R):
        Fd = np.zeros(d.shape)
        for i in range(state_num):
            for j in range(state_num):
                Fd[i,j] = np.abs(R[i] - R[j]) + discount * Wass(P[i], P[j], d)
        return Fd

    for __ in range(1000):
        P, R, p, stat_dist_count = random_MRP(state_num, discrete_variables)
        p =  p[None]
        p_cross = p.transpose().dot(p)

        for _ in range(10):
            d1 = random_metric(state_num, discrete_variables)
            d2 = random_metric(state_num, discrete_variables)

            Fd1 = F(d1, P, R)
            Fd2 = F(d2, P, R)
            l2_p_Fdiff = weighted_l2(Fd1 - Fd2, p_cross)
            l2_p_diff = weighted_l2(d1 - d2, p_cross)
            contracts = l2_p_Fdiff <= l2_p_diff
            if not contracts:
                print(l2_p_diff, l2_p_Fdiff)
                print("diff ", l2_p_diff - l2_p_Fdiff)
                print(P)
                print("p ", p)
                print("R ", R)
                print("d1 ", d1)
                print("d2 ", d2)
                print("Stationary distribution count ", stat_dist_count)


procs = []
for _ in range(16):
    proc = Process(target=find_counter)
    procs.append(proc)
    proc.start()

for proc in procs:
    proc.join()
