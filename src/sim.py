

import numpy as np
#import cvxpy as cp
from numpy.linalg import eigvalsh as eig
from numpy import inf
import scipy.optimize as opt
np.set_printoptions(precision=3, suppress=True, linewidth=999)


## helpers

def vecnorm(v, p):
	if p==2:
		return np.sqrt(np.sum(np.abs(v)**2))
	elif p==inf:
		return np.max(np.abs(v))
	else:
		return (np.sum(np.abs(v)**p))**(1./p)

def norm(X, p=2):
	if p==2:
		return np.sqrt(np.sum(np.abs(X)**2))
	else:
		return vecnorm(eig(X), p)

def ldot(LAM, M):
	return np.tensordot(LAM, M, axes=1)

def lamM(LAM,M):
	""" Return N=LAM.M.
	LAM is a stochastic (n,m) matrix.
	M is a (m,d,d) array. N is a (n,d,d) array.
	N[:,x,y] = LAM @ M[:,x,y]
	"""

def POVMcheck(M, eps=1e-9):
	vals  = np.stack([eig(M[i,:,:]) for i in range(len(M))])
	norm = np.sum(M, axis=0)
	goodvals = np.all(vals>=0)
	goodnorm = np.all(np.abs(norm-np.eye(len(M[0])))<=eps)
	print(vals)
	print(norm)
	if goodvals and goodnorm:
		print("OK")
	else:
		print("bad POVM")

def STOCHcheck(LAM, eps=1e-9):
	nonneg = np.all(LAM >= 0)
	norm = np.sum(LAM, axis=0)
	print(nonneg)
	print(norm)
	goodnorm = np.all(np.abs(norm-1)<=eps)
	if nonneg and goodnorm:
		print("OK")
	else:
		print("not STOCH")

def lam0(M,N):
	n, m = len(N), len(M)
	LAM = np.zeros((n,m))
	for i in range(m):
		if np.trace(M[i])>0:
			for j in range(n):
				LAM[j,i] = np.trace(N[j]@M[i])/np.trace(M[i])
	return LAM


## one way sim distance

def dup(M,N,LAM,p=inf):
	"""
	Calculate upper bound on one way simulation distance for particular LAM.

	d = sum_j 0.5 * ||LAM.M_j - N_j||_p

	Params:
		M: 		POVM with m outcomes acting in d dimensions.
				Array. Shape (m,d,d). Each M[i] is a POVM element.
		N: 		POVM with n outcomes acting in d dimensions.
				Array. Shape (n,d,d). Each N[j] is a POVM element.
		LAM:	Stochastic map from M to N outcomes.
				Array. Shape (n,m). Sum_i LAM[j,i] M[i] = N[j].
				Sum_j LAM[j,i] = 1 for all i.

	Returns: d
		d:	Upper bound on one way sim distance.
	"""
	LM = ldot(LAM,M)
	distances = np.array([0.5 * norm(LM[j]-N[j], p) for j in range(len(N))])
	d = np.sum(distances)
	return d

def dup_opt(M,N,p=inf):
	"""
	Calculate upper bound on one way simulation distance by optimizing LAM.

	d = sum_j 0.5 * ||LAM.M_j - N_j||_p

	Note that lag is a length m vector of lagrange multipliers.

	Params:
		M: 		POVM with m outcomes acting in d dimensions.
				Array. Shape (m,d,d). Each M[i] is a POVM element.
		N: 		POVM with n outcomes acting in d dimensions.
				Array. Shape (n,d,d). Each N[j] is a POVM element.

	Returns: d, LAM, M, N
		d:	Upper bound on one way sim distance.
		LAM:	Stochastic map from M to N outcomes.
				Array. Shape (n,m). Sum_i LAM[j,i] M[i] = N[j].
				Sum_j LAM[j,i] = 1 for all i.
		M,N: The input values.
	"""
	## dimensions
	m, n = len(M), len(N)

	## objective
	def objective(x):
		"""
		Minimize objective function dup(LAM).
		"""
		## 
		LAM = np.reshape(x, (n,m))
		##
		return dup(M, N, LAM, p)

	## constraint
	def constraint(x):
		##
		LAM = np.reshape(x, (n,m))
		##
		return np.sum((1. - np.sum(LAM, axis=0))**2)

	## opt guess
	l0 = lam0(M,N)
	l0 = np.zeros(n*m)
	x0 = np.reshape(l0, (n*m,))

	## opt bounds
	bounds = ((0,1),)*n*m
	
	## optimize
	result = opt.minimize(objective, x0, 
				bounds=bounds, 
				constraints={'type': 'eq', 'fun': constraint}, 
				method=None,
			)

	## results
	LAM = np.reshape(result.x, (n,m))
	d = result.fun

	## report
	if True:
		print()
		print("OPTIMIZE")
		print(result)
		print()

	## if bad
	if result.success==False:
		return None

	## return
	return d, LAM, M, N




##
if __name__=="__main__":
	from tests import *
	test2()

