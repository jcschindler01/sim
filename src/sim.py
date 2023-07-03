

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

	Returns: d, LAM
		d:	Upper bound on one way sim distance.
		LAM:	Stochastic map from M to N outcomes.
				Array. Shape (n,m). Sum_i LAM[j,i] M[i] = N[j].
				Sum_j LAM[j,i] = 1 for all i.
	"""
	## dimensions
	m, n = len(M), len(N)

	## objective function with lagrangian constraint enforcing stochastic
	def objective(x):
		"""
		Minimize objective function 
			dup(LAM) + lag*(1-constraint)
		where the constraint is stochastic sum condition.
		"""
		## x to LAM lag
		LAM = np.reshape(x[:n*m], (n,m))
		lag = np.reshape(x[n*m:], (m,))

		## constraint
		constraint = np.dot((1-np.sum(LAM, axis=0))**2, lag)
		
		##
		return dup(M, N, LAM, p) # + constraint

	## opt guess
	x = np.zeros((n+1)*m)
	x[:n*m] = np.reshape(lam0(M,N), (n*m,))

	## opt bounds
	bounds = ((0,1),)*n*m + ((-np.inf,np.inf),)*m
	
	## optimize
	result = opt.minimize(objective, x, bounds=bounds, method=None)

	## results
	x = result.x
	LAM = np.reshape(x[:n*m], (n,m))
	lag = np.reshape(x[n*m:], (m,))
	d = result.fun

	##
	print(d)
	print(LAM)
	print(lag)
	print(result.success)
	print(result.message)

	## return
	return d



## test
def test2():

	## target POVM
	N = []
	N += [np.array([[1,0],[0,0]])]
	N += [np.array([[0,0],[0,1]])/2]
	N += [np.array([[0,0],[0,1]])/2]
	N = np.stack(N)

	## available POVM
	M = []
	M += [np.array([[1,-1],[-1,1]])/2]
	M += [np.array([[1, 1],[ 1,1]])/2]
	M = np.stack(M)


	## upper bound on sim distance
	d = dup_opt(M,N,p=inf)
	print(d)





## test
def test1():

	## target POVM
	N = []
	N += [np.array([[1,0],[0,0]])]
	N += [np.array([[0,0],[0,1]])]
	N = np.stack(N)

	## available POVM
	M = []
	M += [np.array([[1,-1],[-1,1]])/2]
	M += [np.array([[1, 1],[ 1,1]])/2]
	M = np.stack(M)

	## is POVM?
	print()
	print("POVMchecks")
	POVMcheck(M)
	POVMcheck(N)
	print()

	## print POVMs
	print("available M")
	print(M)
	print()
	print("target N")
	print(N)
	print()

	## stochastic processing
	LAM = lam0(M,N)

	## is stochastic?
	print("STOCHcheck")
	STOCHcheck(LAM)
	print()

	## print STOCH
	print("stochastic LAM")
	print(LAM)
	print()

	## process M to LM
	LM = ldot(LAM, M)

	## is POVM?
	print("POVMcheck")
	POVMcheck(LM)
	print()

	## print simulated POVM
	print("simulated LM")
	print(LM)
	print()

	## upper bound on sim distance
	d = dup(M,N,LAM,p=inf)

	## print distance
	print("upper bound on sim distance dvec(M,N)")
	print(d)
	print()




##
if __name__=="__main__":
	test2()

