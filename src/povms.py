
"""
Generate POVMs.

    M:      POVM with m outcomes acting in d dimensions.
            Array. Shape (m,d,d). Each M[i] is a POVM element.

    d = dimension
    m = number of M outcomes
"""

## import
import numpy as np
from scipy.stats import unitary_group
from sim import *

## helpers

dag = lambda X: np.conjugate(np.transpose(X))

def convex(p, M, N, disjoint=True):
    ## non-disjoint fails
    if disjoint==False and len(M)!=len(N):
        print("Non-disjoint convex combo requires same outcome set.")
        return None
    ## non-disjoint succeeds
    if disjoint==False and len(M)==len(N):
        return p*M + (1-p)*N
    ## disjoint
    if disjoint==True:
        return np.vstack([p*M,(1-p)*N])

def BLOCHxyz(x,y,z):
    return (1/2)*np.array([[1+z,x-1j*y],[x+1j*y, 1-z]], dtype=complex)

def blochstate(param='z', mode=None):
    ## string mode = None
    if mode==None:
        ref = dict(x=(1,0,0), y=(0,1,0), z=(0,0,1))
        if param in ref.keys():
            x,y,z = ref[param]
            return BLOCHxyz(x,y,z)
    ## theta mode
    if mode=="theta":
        eps = 1e-12
        theta = param*np.pi/180
        x,y,z = (1-eps)*np.sin(theta), 0, (1-eps)*np.cos(theta)
        return BLOCHxyz(x,y,z)

## constructors

def computational(dim):
    M = np.zeros((dim,dim,dim), dtype=complex)
    for k in range(dim):
        M[k,k,k] = 1
    return M

def randombasis(dim):
    M = computational(dim)
    U = unitary_group.rvs(dim)
    for i in range(len(M)):
        M[i,:,:] = U @ M[i,:,:] @ dag(U)
    return M

def blochbasis(param='z', mode=None):
    ##
    M = np.zeros((2,2,2), dtype=complex)
    M[0,:,:] = blochstate(param, mode)
    M[1,:,:] = np.eye(2) - M[0,:,:]
    ##
    return M

def ignorant(dim,m=1):
    M = np.zeros((m,dim,dim), dtype=complex)
    for i in range(m):
        M[i,:,:] = np.eye(dim)/m
    return M





M = blochbasis(1, mode='theta')

print(M)
print()

POVMcheck(M)
