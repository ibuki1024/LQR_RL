import numpy as np
from scipy import linalg

class LqrAgent():
    def __init__(self,A,B,E,F,initial_gain):
        self.A = A
        self.B = B
        self.E = E
        self.F = F
        self.initial_gain = initial_gain

    def fit(self, iter_RLS=400, iter_Gain=1000, sn=100):
        A = self.A
        B = self.B
        E = self.E
        F = self.F
        U = self.initial_gain
        U_opt = - lqr(A,B,E,F)[1]
        gamma = 1.

        n,p = B.shape
        n_th = int((n+p)*(n+p+1)/2)

        x_hist = []
        U_hist = []
        Uerr_hist = []


        x = np.random.rand(n,)/10;
        P = np.eye(n_th)

        for k in range(1,iter_Gain+1):
            Uerr_hist.append(np.linalg.norm(U-U_opt, ord='fro'))
            for i in range(1,iter_RLS+1):
                u = np.dot(U,x) + np.random.randn(p,)*s_n
                c = quad(x,E) + quad(u,F)
                bar = H_to_theta(np.outer(np.hstack((x,u)),np.hstack((x,u))))
                x = np.dot(A,x) + np.dot(B,u)
                u = np.dot(U,x)
                barplus = H_to_theta(np.outer(np.hstack((x,u)),np.hstack((x,u))))

                phi = bar - np.dot(gamma, barplus)

                e = c - np.dot(phi,theta)
                denom = 1 + quad(phi,P)
                theta += np.dot(np.dot(P,phi),e)/denom
                P -= np.outer(np.dot(P,phi),np.dot(phi,P))/denom

            H = theta_to_H(theta,n+p)
            U = - np.dot(np.linalg.inv(H[n:n+p,n:n+p]), H[n:n+p,0:n])
            if max(np.abs(np.linalg.eig(A+np.dot(B,U))[0]))>1:
                print(k)
                break

        self.final_gain = U
        self.Uerr_hist = Uerr_hist

class DpAgent():
    def __init__(self,A,B,E,F,initial_gain):
        self.A = A
        self.B = B
        self.E = E
        self.F = F
        self.initial_gain = initial_gain

    def fit(self, iter_Gain=1000):
        A = self.A
        B = self.B
        E = self.E
        F = self.F
        U = self.initial_gain
        U_opt = - lqr(A,B,E,F)[1]

        n,p = B.shape

        Uerr_hist = []
        for i in range(iter_Gain):
            Uerr_hist.append(np.linalg.norm(U-U_opt, ord='fro'))
            sys_n = np.eye(n) #preserve (A+BU)^i
            K_U = np.zeros(A.shape)
            for i in range(5000):
                K_U = K_U + quad(sys_n, (E+quad(U,F)))
                sys_n = np.dot(sys_n,(A+np.dot(B,U)))

            H_11 = E + quad(A,K_U)
            H_12 = np.dot(np.dot(A.T,K_U),B)
            H_21 = H_12.T
            H_22 = F + quad(B,K_U)
            H_upper = np.hstack((H_11,H_12))
            H_under = np.hstack((H_21,H_22))
            H = np.vstack((H_upper, H_under))

            U = - np.dot(np.linalg.inv(H[n:n+p,n:n+p]), H[n:n+p,0:n])
        self.final_gain = U
        self.Uerr_hist = Uerr_hist



def initialGain(A,B,E,F):
    n,p = B.shape
    n_th = int((n+p)*(n+p+1)/2)
    while True:
        theta = np.random.randn(n_th,)
        H = theta_to_H(theta,n+p)
        U = - np.dot(np.linalg.inv(H[n:n+p,n:n+p]), H[n:n+p,0:n])
        if max(np.abs(np.linalg.eig(A+np.dot(B,U))[0]))<0.99:
            break
    return U

def H_to_theta(H):
    '''transform H to theta
    '''
    n = H.shape[0]
    theta = H[0,:]
    for i in range(2,n+1):
        theta = np.hstack((theta, H[i-1,(i-1):n]))
    return theta

def theta_to_H(theta,n):
    '''transform theta to H
    '''
    H = theta[0:n]
    k = n
    for i in range(2,n+1):
        H = np.vstack((H,np.hstack((np.zeros((1,i-1))[0], theta[k:k+n-i+1]))))
        k = k+n-i+1;
    H = (H+H.T)/2;
    return H

def quad(X,A):
    """ Returns X.T*A*X
    """
    assert A.shape[1] == X.shape[0], 'Dimension Error'
    tmp = np.dot(np.dot(X.T,A),X)
    return tmp

def lqr(A, B, Q, R):
    '''LQR for discrete time system
    '''
    P = linalg.solve_discrete_are(A, B, Q, R)
    K = np.dot(np.linalg.inv(quad(B,P)+R),(np.dot(np.dot(B.T,P),A)))
    e = linalg.eigvals(A - B.dot(K))
    return P, K, e
