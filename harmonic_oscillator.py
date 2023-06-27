
import numpy as np
from pylab import *  # for plot
from scipy.special import factorial
from scipy import linalg as LA
import scipy.sparse as sps
from scipy.linalg import eigh
from scipy.special import eval_hermite
from scipy.signal import argrelextrema
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.special import gamma, factorial



# Class to create a harmonic oscillator

class Harmonic_Oscillator:
    '''
    Class to generate potential and wavefunctions of an harmonic oscillator.
    Attributes:
    omega_min (float): Minimum value of omega
    omega_max (float): Maximum value of omega
    x0_min (float): Minimum value of x0
    x0_max (float): Maximum value of x0
    hbar (float): h bar
    m (float): mass
    N (int): number of points of the grid
    omega (np.array): values of omega
    x0 (np.array): values of x0
    '''
    def __init__(self, omega_min = 0.001, omega_max = 1, x0_min = -0.5, x0_max = 0.5,
                     hbar = 1, m = 1, x_range = 0.8, N = 100):
        self.omega_min = omega_min # omega ~ U(omega_min, omega_max)
        self.omega_max = omega_max
        self.x0_min = x0_min
        self.x0_max = x0_max
        self.hbar = hbar
        self.m = m
        self.find_xrange(x_range) # x in [-3 smax, 3smax]
        self.N = N # Number of points of the grid 
        self.omega=None
        self.x0 = None

    def find_xrange(self, x_range):
        '''
        Find range of x values x \in [-x_range * sigma_max, + x_range * sigma_max]
        Args:
        x_range (float)
        '''
        smax = np.sqrt(self.hbar/(self.m*self.omega_min)) # Find sigma_max
        self.xmin = -x_range*smax
        self.xmax = x_range*smax

    def generate_omega(self, N):
        '''
        Generates new values of omega
        Args:
        N (int): number of samples
        '''
        if N==None:
            N = 100
        self.omega = np.random.uniform(self.omega_min, self.omega_max, N).reshape(-1,1)
        self.x0 = np.random.uniform(self.x0_min, self.x0_max, N).reshape(-1,1)
    
    def generate_data(self, N=None, n=None , new_omega=True):
        '''
        Generates N random data points from the energetic level n
        Args:
        N (int): number of samples
        n (int): energetic level
        new_omega (boolean): if True, new values of omega are generated
        Returns:
        φ_n (np.array): HO wavefunctions
        x (np.array): grid
        omega (np.array): omega values for each sample
        x0 (np.array): values of x0 for each sample
        potential (np.array): V(x)
        '''
        if new_omega or self.omega.any()==None or self.x0.any()==None:
            self.generate_omega(N)

        x = np.arange(self.xmin, self.xmax, (self.xmax - self.xmin)/self.N)
        sigma_inv = np.sqrt(self.m*self.omega/self.hbar).reshape(-1,1)
        ones = np.repeat(1, N)
        x_mat = np.tensordot(x, ones, axes=0).T
        all_x = (x_mat - self.x0)*sigma_inv # It is a matrix of dim (num_omega x num_x_points), 
        # where each row has the values of x times sqrt(m*omega/hbar. In each row we change the value of omega
        herm = eval_hermite(n, all_x) # H_n(x/sigma)
        exp = np.exp(- all_x**2/2) # Exponential term

        φ_n = exp*herm

        h = (self.xmax - self.xmin)/self.N
        C = 1./np.sqrt(np.sum(φ_n*φ_n*h, axis = 1)) #1/np.sqrt(2**n * math.factorial(int(n))) * np.sqrt(sigma_inv)# Normalization constant
        C = C.reshape(-1,1)
        φ_n = C*φ_n #exp*herm

        potential = (x_mat-self.x0)**2 * 1/2*self.m*self.omega**2

        return φ_n, x, self.omega, self.x0, potential

    def get_energy(self, n, omega=np.array([None])):
        '''
        Get theoretical energy
        Args:
        n (int): energetic level
        omega (np.array): values of omega
        Returns:
        E (np.array): energies
        '''
        if omega.any()==None:
            omega = self.omega
        E = self.hbar*omega*(n+1/2)
        return E.flatten()