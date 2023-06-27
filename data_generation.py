
import numpy as np
from pylab import *  # for plot
from scipy.special import factorial
from scipy import linalg as LA
import scipy.sparse as sps
from scipy.linalg import eigh
from scipy.special import eval_hermite
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.special import gamma, factorial
import tensorflow as tf

from sklearn.model_selection import train_test_split


# Compute the energy of a wave function
def empirical_energy1D(φ, 
                       potential, 
                       xmin=-8, 
                       xmax = 8,
                       N=100, 
                       hbar=1, 
                       m=1):
    '''
    Function to calculate the empirical energy of a wavefunction
    Args:
      φ (np.array): Wavefunctions
      potential (np.array): potential V(x)
      xmin (int): minimum value of x
      xmax (int): maximum value of x
      N (int): number of grid points
      hbar (float): h bar
      m (float): mass
    Returns:
      E (np.array): empirical energies
    '''
    # Normalize φ just in case
    h = (xmax - xmin)/N
    def energy(φ,potential,h):
        '''
          Calculates the empirical energy for one wavefunction
          Args:
            φ (np.array): Wavefunctions
            potential (np.array): potential V(x)
            h (float): lattice size
          Returns:
            E (float): empirical energy 
        '''
        C = 1./np.sqrt(np.sum(φ*φ*h))
        φ = C*φ
        # We first calculate the second derivative of φ
        φr = np.concatenate(( φ[1:], np.zeros(1)), axis=0) # We add 0 at the extrema. 
        # It makes sense because φ(x)->0 at x->+-inf
        φl = np.concatenate(( np.zeros(1), φ[:-1]), axis=0)

        deriv = (φr - 2.0*φ + φl)/(h*h)
        return np.sum((-hbar*hbar/(2*m)*φ*deriv + potential*(φ*φ))*h)

    E = np.array([energy(φ[i,:], potential[i,:], h) for i in range(φ.shape[0])])   
    return E



class eigen_state_potential:
    def __init__(self, alpha_min=None, alpha_max=None, N=None, xmin=None, xmax=None):
        '''
        Class to generate data (V(x) and φ(x) ground state) for potentials of the form
        V(x) = sum_i alpha_i x^i, using the plane wave basis
        Args:
            alpha_min: vector of length N, with the minimum value of the coefficients alpha
            alpha_max: vector of length N, with the maximum value of the coefficients alpha
            the values of alpha will be randomly distributed in [alpha_min, alpha_max]
        '''
        if len(alpha_min)!=len(alpha_max):
            print("Error. Inconsisten shapes")
        self.alpha_min = np.array(alpha_min)
        self.alpha_max = np.array(alpha_max)
        self.N =N  # Length of plane wave basis
        self.k = len(alpha_min) #Number of alphas for V(x)
        self.xmin = xmin #xmin
        self.xmax = xmax #xmax
        #print(self.alpha_min,self.alpha_max,self.N,self.k,self.x_min,self.x_max)

    def delta(self,n,m):
        """
        Define the Dirac's delta
        """
        if(n==m):
            delta=1
        else:
            delta=0      
        return delta

    def find_eigen_state(self,alphas, n_state):
        '''
        Finds the eigen state of a potential V(x) = sum_i alpha_i x^i
        Args:
            alphas(np array): size k. Coefficients of the potential V(x)
            n_state (int): Number of excited state (default n_state=0, ground state)
        Returns:
            E_a (float): Energy of the ground state for potential V
            a (np.array): size N. Coefficients in the basis of the H.O potential
        '''
        # 0. Generate matrix of C_nm
        N = self.N  # N==N
        xmin = self.xmin  # xmin
        xmax = self.xmax  # xmax

        T = np.zeros((N,N))
        delta_x=(xmax-xmin)/N
        L=delta_x*N
        mu=1.
        
        # 1. Build matrix T using the plane wave basis
#*********TODO***************
        for n in range(N):
            for m in range(N):
                if(n==m):
                    T[n,m] =(-1)**(n-m)*np.pi**2/(mu*L**2)*(N**2+2)/6
                elif(n!=m):
                    T[n,m] = (-1)**(n-m)*np.pi**2/(mu*L**2)*1./(np.sin(np.pi*(n-m)/N))**2


        # 2. Build matrix V
#*********TODO***************
        V = np.zeros((N,N))
        k = self.k  # k
        for n in range(N):
            x=xmin+n*delta_x
             #V[n,n]=0.5*x**2
            for i in range(k):
                V[n,n]+=alphas[i]*x**i

        # 3. Build matrix H
        H = np.zeros((N,N))
        veps= np.zeros((N,N))
        vaps=np.zeros(N)
        a= np.zeros((N,n_state+1))
        E_a=np.zeros(n_state+1)
        for n in range(N):
            for m in range(N):
                H[n,m] = T[n,m] + V[n,m]*self.delta(n,m)     

        # 4. Diagonalize matrix H
        vaps, veps = eigh(H)

        # 5. Set bound wave functions to be real and normalize them
        for n in range(N):
            norm=np.dot(veps[:,n],veps[:,n])
            macloc=np.argmax(abs(veps[:,n]))
            veps[:,n]=veps[:,n]*abs(veps[macloc,n])/veps[macloc,n]
            veps[:,n]=veps[:,n]/np.sqrt(delta_x*norm)

        # 6. We choose the vector with n_state-th lowest energy
        # as an approximation of the n_state state   
        a = veps[:,n_state] # Final value of eigenvalues for state n_state
        E_a =vaps[n_state] # Value of the energy
        return E_a, a

    def generate_data(self, n_samples, alpha=np.array([None]), n_state=None, display=100):
        '''
        Generates samples of potentials  with random coefficients and finds the n_state excited state for them
        Args:
            n_samples (int): Number of samples of potentials (alphas)
            alpha (np.array): Values of alpha. If you want to generate them randomly, don't provide anything
            n_state (int): Number of excited state (default n_state=0, ground state)
            display (int): Display step
        Returns:
            E (np.array): size n_samples. Ground energy for each V
            a (np.array): size n_samples x N. Coefficients in the H.O basis for each V
            alpha (np.array): size n_samples x k. Coefficients of the potentials V(x)
        '''
        data = np.zeros((n_samples, self.N))

        # Generate random value of alphas
        if (alpha==None).any():
            print("Random alphas")
            r_alpha = np.random.random((n_samples, self.k)) # Values between 0 and 1
            alpha = r_alpha*(self.alpha_max - self.alpha_min)+ self.alpha_min # random alpha

        # Prepare vectors of energies and coefficients
        E = np.zeros(n_samples)
        a = np.zeros((n_samples, self.N))
        # Find ground state for each sample
        for i in range(n_samples):
            E_new, a_new = self.find_eigen_state(alpha[i,:], n_state)
            if i%display==0:
                print("\rGenerating data: {}/{}".format(i,n_samples), end='')
            E[i] = E_new
            a[i,:] = a_new
        return E, a, alpha  

    def evaluate_potential(self, xmin, xmax, N, alpha):
        '''
        Given the coeefficients alphas, it evaluates the potential in V(x)
        Args:
            xmin(float): minimum value of x
            xmax (float): maximum value of x
            N (int): Number of points between xmin and xmax
            alpha (np.array): size N x k. Matrix of coefficients of V(x) (each row a different potential)
        Returns:
            V(np.array): size n_samples x N. V(x) for every sample
            x(np.array): size N. Values of x
        '''
        x = np.arange(xmin, xmax, (xmax - xmin)/N)
        n_samples, k = alpha.shape
        V = np.zeros((n_samples, N))
        x_mat = (x**np.arange(k)[:,None])# Matrix of powers of x: x^0, x^1, x^2, ..., x^N (in every row)
        V = np.zeros((n_samples, N))# V(x) in each row different alpha
        for i in range(n_samples):
            for j in range(N):
                V[i,j] = np.dot(alpha[i,:],x_mat[:,j])

        return V, x

    def final_wavefunction(self, xmin, xmax, N, a):
        '''
        Returns the final wavefunctions psi(x) = sum_i alpha_i φ_i(x) for each alpha.
        Args:
            xmin(float): minimum value of x
            xmax (float): maximum value of x
            N (int): Number of points between xmin and xmax
            a (np.array): size n_samples x N. Coefficients in the H.O basis for each V
        Returns:
            waves(np.array): size n_samples x N. psi(x) for each value of V (given by alpha)
        '''
        x = np.arange(xmin, xmax, (xmax - xmin)/N)
        n_samples, _ = a.shape
        # Construct matrix of φ_n   
        waves = np.zeros((n_samples, N))
        φs = np.zeros((n_samples, N))
        for i in range(n_samples):
            for j in range(N):
                waves[i,j] = a[i,j]
                φs[i,j] = waves[i,j]
                # convention: To choose the phase we make the maximums be first
            w = waves[i,:]
            maxi = argrelextrema(w, np.greater)[0]
            mini = argrelextrema(w, np.less)[0]
            idx2= np.abs(w[maxi])>5e-2
            maxi = maxi[idx2]
            idx2= np.abs(w[mini])>5e-2
            mini = mini[idx2]
            if len(maxi)==0 and len(mini)>0:
                waves[i,:] = -waves[i,:]
            elif len(mini)>0 and len(maxi)>0 and mini[0]<maxi[0]:
                waves[i,:] = -waves[i,:]
        return waves, x, φs