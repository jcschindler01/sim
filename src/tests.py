
from sim import *

## test
def test2():
    """TEST 2
    Input available M and target N povms.
    Find the optimal upper bound on one-way sim distance dvec using
        dup_opt(M,N,p=inf).
    Also compare to LAM0 the change of frame map.
    """

    ##
    print(test2.__doc__)

    ## available POVM
    M = []
    M += [np.array([[1,0],[0,1]])]
    M = np.stack(M)

    ## target POVM
    N = []
    N += [np.array([[1,0],[0,0]])]
    N += [np.array([[0,0],[0,1]])]
    N = np.stack(N)

    ## upper bound on sim distance
    d, LAM, M, N = dup_opt(M,N,p=inf)
    
    ## report
    print()
    print("dvec >= ")
    print(d)
    print()
    print("LAM = ")
    print(LAM)
    print("guessed LAM0 = ")
    print(lam0(M,N))
    print()
    print("LM = ")
    print(ldot(LAM,M))
    print()
    print("Target N = ")
    print(N)
    print() 
    print("dvec >= ")
    print("%.3f"%(d))
    print()









## test
def test1():

    ## available POVM
    M = []
    M += [np.array([[1,-1],[-1,1]])/2]
    M += [np.array([[1, 1],[ 1,1]])/2]
    M = np.stack(M)

    ## available POVM
    M = []
    M += [np.array([[1,0],[0,1]])]
    M = np.stack(M)

    ## target POVM
    N = []
    N += [np.array([[1,0],[0,0]])]
    N += [np.array([[0,0],[0,1]])]
    N = np.stack(N)

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



