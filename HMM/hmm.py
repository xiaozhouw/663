import numpy as np
from numpy import random
from collections import deque
def forward(A,B,pi,sequence,scale=False):
    '''
    Perform the forward step in Baum-Welch Algorithm.
    
    Returns likelihood, alpha
    
    (Details on alpha please see Baum-Welch Algorithm)
    
    Parameters
    ----------
    A: np.ndarray
        Stochastic transition matrix.
    B: np.ndarray
        Emission matrix.
    pi: np.1darray
        Initial state distribution.
    sequence: array-like
        The observed sequence.
        Need to be converted to integer coded.    
    scale: boolean, default False
        Whether alpha should be scale to have sum of 1 for each t. 
        When the chain is long, the likelihood would be very close to 0.
        To avoid overflow problem, scaling alpha can be applied.
        Typically used for chain longer than 1,000 or when warning of overflow shows up.
        When `scale`=Ture, the likelihood is going to be 1 for scaling reason! 
    '''

    N=A.shape[0]
    M=B.shape[1]
    T=len(sequence)
    alpha=np.zeros([T,N])
    alpha[0]=pi*B[:,sequence[0]]
    t=1
    while True:
        if t==T:
            break
        alpha[t]=alpha[t-1]@A*B[:,sequence[t]]
        if scale:
            alpha[t]/=np.sum(alpha[t])
        t+=1
    return(np.sum(alpha[T-1]),alpha)
def backward(A,B,pi,sequence,scale=False):
    '''
    Perform the backward step in Baum-Welch Algorithm.
    
    Returns likelihood, beta
    
    (Details on alpha please see Baum-Welch Algorithm)

    Parameters
    ----------
    A: np.ndarray
        Stochastic transition matrix.
    B: np.ndarray
        Emission matrix.
    pi: np.1darray
        Initial state distribution.
    sequence: array-like
        The observed sequence.
        Need to be converted to integer coded.   
    scale: Boolean, default False
        Whether beta should be scale to have sum of 1 for each t. 
        When the chain is long, the likelihood would be very close to 0.
        To avoid overflow problem, scaling beta can be applied.
        Typically used for chain longer than 1,000 or when warning of overflow shows up.
        When `scale`=Ture, the likelihood is going to be 1 for scaling reason! 
    '''

    N=A.shape[0]
    M=B.shape[1]
    T=len(sequence)
    beta=np.zeros([T,N])
    beta[T-1]=1
    t=T-2
    while True:
        if t<0:
            break
        beta[t]=A@(B[:,sequence[t+1]]*beta[t+1])
        if scale:
            beta[t]/=np.sum(beta[t])
        t-=1
    return(np.sum(pi*B[:,sequence[0]]*beta[0]),beta)

def Viterbi(A,B,pi,sequence):
    '''
    Viterbi decoding of HMM.
    
    Return: Most likely hidden states.

    Parameters
    ----------
    A: np.ndarray
        Stochastic transition matrix.
    B: np.ndarray
        Emission matrix.
    pi: np.1darray
        Initial state distribution.
    sequence: array-like
        The observed sequence.
        Need to be converted to integer coded.    
    '''

    N=A.shape[0]
    M=B.shape[1]
    T=len(sequence)
    delta=np.zeros([T,N])
    psi=np.zeros([T,N])
    delta[0]=pi*B[:,sequence[0]]
    t=1
    while True:
        if t==T:
            break
        delta_A=delta[t-1,np.newaxis].T*A
        delta[t]=np.max(delta_A,axis=0)*B[:,sequence[t]]
        psi[t]=np.argmax(delta_A,axis=0)
        t+=1
    psi=psi.astype(int)
    q=np.zeros(T).astype(int)
    q[T-1]=np.argmax(delta[T-1])
    t=T-2
    while True:
        if t<0:
            break
        q[t]=psi[t+1,q[t+1]]
        t-=1
    return(q)

def Baum_Welch(A,B,pi,sequence,max_iter,threshold=1e-15,scale=False):
    '''
    Baum-Welch algorithm of HMM. 
    See https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm.
    
    Returns: estimated transition matrix, estimated emission matrix, estimated initial state distribution 
    
    Parameters
    ----------
    A: np.ndarray
        Initial stochastic transition matrix.
    B: np.ndarray
        Emission matrix.
    pi: np.1darray
        Initial state distribution.
    sequence: array-like
        The observed sequence.
        Need to be converted to integer coded.   
    max_iter: int
        Number of iteration for EM.
    threshold: float
        Covergence check thershold for likelihood.
        Notice: when `scale` is set at True (for long chain), threshold will be set to 0 (run `max_iter` iterations) because the `forward` and `backward` will not return true likelihood. 
    scale: boolean, default False
        If set as True, use scaled alpha and beta.
        This is useful to avoid overflow when dealing with long chains.
    '''
    if scale:
        threshold=0
    N=A.shape[0]
    M=B.shape[1]
    T=len(sequence)
    likelihood,alpha=forward(A,B,pi,sequence,scale)
    for i in range(max_iter):
        beta=backward(A,B,pi,sequence,scale)[1]
        #temporary variables
        gamma=alpha*beta/np.sum(alpha*beta,axis=1).reshape((T,1))
        #Non-vectorized version for xi
        #xi=np.zeros([N,N,T-1])
        #for t in range(T-1):
        #    xi[:,:,t]=alpha[t].reshape((N,1))*A*beta[t+1]*B[:,sequence[t+1]]
        #    xi[:,:,t]=xi[:,:,t]/np.sum(xi[:,:,t])
        xi=alpha.T[:,np.newaxis,:-1]*A[:,:,np.newaxis]*(beta*B[:,sequence].T).T[np.newaxis,:,1:]
        xi=xi/np.sum(xi,axis=(0,1))
        pi=gamma[0]
        A=np.sum(xi,axis=2)/np.sum(gamma[:-1],axis=0).reshape([N,1])
        B=np.zeros([N,M])
        for t in range(T):
            B[:,sequence[t]]+=gamma[t]
        B=B/np.sum(gamma,axis=0).reshape([N,1])
        likelihood_new,alpha=forward(A,B,pi,sequence,scale)
        if abs(likelihood-likelihood_new)<threshold:
            break
        likelihood=likelihood_new
    return(A,B,pi)

def Baum_Welch_linear_memory(A,B,pi,sequence,max_iter,threshold=1e-15):
    '''
    Baum-Welch algorithm in linear memory.
    Implemented according to Churbanov, A., & Winters-Hilt, S. (2008).

    Returns: estimated transition matrix, estimated emission matrix, estimated initial state distribution
    
    ####Warning####
    
    This function is not correctly implemnted! Do NOT use this function!
    
    Parameters
    ----------
    A: np.ndarray
        Initial stochastic transition matrix.
    B: np.ndarray
        Emission matrix.
    pi: np.1darray
        Initial state distribution.
    sequence: array-like
        The observed sequence.
        Need to be converted to integer coded.   
    max_iter: int
        Number of iteration for EM.
    threshold: float
        Covergence check thershold for likelihood.
        Notice: when `scale` is set at True (for long chain), threshold will be set to 0 (run `max_iter` iterations) because the `forward` and `backward` will not return true likelihood. 
    scale: boolean, default False
        If set as True, use scaled alpha and beta.
        This is useful to avoid overflow when dealing with long chains.
    '''

    N=A.shape[0]
    M=B.shape[1]
    T=len(sequence)
    ###########################
    for z in range(max_iter):
        ##Beta_t+1
        beta_tilt_old=np.zeros(N)
        ##Beta_t
        beta_tilt_new=np.zeros(N)
        #T_t+1
        T_tilt_old=np.zeros([N,N,N])
        #T_t
        T_tilt_new=np.zeros([N,N,N])
        #E_t+1
        E_tilt_old=np.zeros([N,M,N])
        #E_t
        E_tilt_new=np.zeros([N,M,N])
        beta_tilt_old+=1
        d=1/np.sum(beta_tilt_old)
        beta_tilt_old=d*beta_tilt_old
        for m in range(N):
            for i in range(N):
                for gamma in range(M):
                    E_tilt_old[m,gamma,m]=beta_tilt_old[i]*int(sequence[T-1]==gamma)
        for t in range(T-2,-1,-1):
            beta_tilt_new=A@(B[:,sequence[t+1]]*beta_tilt_old)
            dt=1/np.sum(beta_tilt_new)
            for m in range(N):
                for i in range(N):
                    for j in range(N):
                        partial=0
                        for n in range(N):
                            partial+=A[m,n]*T_tilt_old[i,j,n]*B[n,sequence[t+1]]
                        T_tilt_new[i,j,m]=dt*(beta_tilt_old[j]*A[m,j]*B[j,sequence[t+1]]*int(i==m)+partial)
                    for gamma in range(M):
                        partial=0
                        for n in range(N):
                            partial+=B[n,sequence[t+1]]*A[m,n]*E_tilt_old[i,gamma,n]
                        E_tilt_new[i,gamma,m]=dt*(partial+beta_tilt_new[m]*int(sequence[t]==gamma)*int(m==i))
            beta_tilt_new=dt*beta_tilt_new
            beta_tilt_old=beta_tilt_new
            T_tilt_old=T_tilt_new
            E_tilt_old=E_tilt_new
        E_end=np.zeros([N,M])
        T_end=np.zeros([N,N])
        for m in range(N):
            E_end+=E_tilt_old[:,:,m]*pi[m]*B[m,sequence[0]]
            T_end+=T_tilt_old[:,:,m]*pi[m]*B[m,sequence[0]]
        alpha=pi*B[:,sequence[0]]
        pi=alpha*beta_tilt_old
        pi=pi/np.sum(pi)
        B=E_end/np.sum(E_end,axis=1).reshape((N,1))
        A=T_end/np.sum(T_end,axis=1).reshape((N,1))
    return(A,B,pi)

def Viterbi_linear_memory(A,B,pi,sequence):
    '''
    Viterbi decoding of HMM in linear memory by using `deque` in `collections`.
    
    Return: Most likely hidden states.

    Parameters
    ----------
    A: np.ndarray
        Stochastic transition matrix.
    B: np.ndarray
        Emission matrix.
    pi: np.1darray
        Initial state distribution.
    sequence: array-like
        The observed sequence.
        Need to be converted to integer coded.    
    '''

    N=A.shape[0]
    M=B.shape[1]
    T=len(sequence)
    delta=np.zeros([T,N])
    psi=np.zeros([T,N])
    d = deque([], maxlen=T)
    delta[0]=pi*B[:,sequence[0]]
    t=1
    while True:
        if t==T:
            break
        delta_A=delta[t-1,np.newaxis].T*A
        delta[t]=np.max(delta_A,axis=0)*B[:,sequence[t]]
        d.append(np.argmax(delta_A,axis=0).astype(int))
        t+=1

    q=np.zeros(T).astype(int)
    q[T-1]=np.argmax(delta[T-1])

    t=T-2
    while d:
        q[t]=d.pop().reshape([N,1])[q[t+1]]
        t-=1  
    return q

def sim_HMM(A,B,pi,length):
    '''
    Simulate a HMM.
    
    Return: hidden states, observed sequence

    Parameters
    ----------
    A: np.ndarray
        Stochastic transition matrix.
    B: np.ndarray
        Emission matrix.
    pi: np.1darray
        Initial state distribution.
    length: int
        The length of the chain.
    '''
    states=np.arange(A.shape[0])
    outcomes=np.arange(B.shape[1])
    out_states=np.zeros(length)
    out_emission=np.zeros(length)
    out_states=np.zeros(length).astype(int)
    out_emission=np.zeros(length).astype(int)
    out_states[0]=random.choice(states,size=1,p=pi)
    out_emission[0]=random.choice(outcomes,size=1,p=B[out_states[0]])
    for i in range(1,length):
        out_states[i]=random.choice(states,size=1,p=A[out_states[i-1]])
        out_emission[i]=random.choice(outcomes,size=1,p=B[out_states[i]])
    return(out_states,out_emission)