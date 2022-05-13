
import networkx as nx
import time
import scipy.io as sio
import numpy as np


def constrained_kmeans(data, demand, maxiter=None, fixedprec=1e9):
	data = np.array(data)
	
	min_ = np.min(data, axis = 0)
	max_ = np.max(data, axis = 0)
	
	C = min_ + np.random.random((len(demand), data.shape[1])) * (max_ - min_)
	M = np.array([-1] * len(data), dtype=np.int)
	
	itercnt = 0
	while True:
		itercnt += 1
		

		g = nx.DiGraph()
		g.add_nodes_from(range(0, data.shape[0]), demand=-1)
		for i in range(0, len(C)):
			g.add_node(len(data) + i, demand=demand[i])
		

		cost = np.array([np.linalg.norm(np.tile(data.T, len(C)).T - np.tile(C, len(data)).reshape(len(C) * len(data), C.shape[1]), axis=1)])
		

		data_to_C_edges = np.concatenate((np.tile([range(0, data.shape[0])], len(C)).T, np.tile(np.array([range(data.shape[0], data.shape[0] + C.shape[0])]).T, len(data)).reshape(len(C) * len(data), 1), cost.T * fixedprec), axis=1).astype(np.uint64)

		g.add_weighted_edges_from(data_to_C_edges)

		a = len(data) + len(C)
		g.add_node(a, demand=len(data)-np.sum(demand))
		C_to_a_edges = np.concatenate((np.array([range(len(data), len(data) + len(C))]).T, np.tile([[a]], len(C)).T), axis=1)
		g.add_edges_from(C_to_a_edges)
		
		

		f = nx.min_cost_flow(g)




		M_new = np.ones(len(data), dtype=np.int) * -1
		for i in range(len(data)):
			p = sorted(f[i].items(), key=lambda x: x[1])[-1][0]
			M_new[i] = p - len(data)
			

		if np.all(M_new == M):

			return (C, M, f)
			
		M = M_new
			

		for i in range(len(C)):
			C[i, :] = np.mean(data[M==i, :], axis=0)
			
		if maxiter is not None and itercnt >= maxiter:
			return (C,  M, f)

def main(k):
	k=int(k)
	dataFile = '.\plda_bl_score.mat'
	loaddata = sio.loadmat(dataFile)
	data = loaddata['data']
	ti = len(data)/k
	ti = int(ti)
	demand=[]
	for i in range(0,k):
		demand.append(ti)
	(C, M, f) = constrained_kmeans(data, demand)
	dataNew = '.\l4.mat'
	M=M+1
	sio.savemat(dataNew, {'l4': M})


if __name__ == '__main__':
	main()
