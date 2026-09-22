import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math

# Load System Data
sys_data = sio.loadmat('./Assignment_data/system_q1a.mat', squeeze_me=True)
A = sys_data['A_q1a']
names_q1a = sys_data['names_q1a']
latex_names_q1a = sys_data['latex_names_q1a']

#------------------------------------------------------------------------------
#Q1.1.1: Manual Excitation, Calculating eignevalues of the system matrix and 
# frequency, damping of the oscillation modes
#------------------------------------------------------------------------------
print("------------------------------------------------------------------------------")
print("Q1.1.1: Calculating eigenvalues of the system matrix and frequency, damping")
eigen_vals, eigen_vec = np.linalg.eig(A)
for lam in eigen_vals:
    sigma = lam.real
    omega = lam.imag
    if abs(omega) > 10e-6:
        freq = abs(omega)/(2*math.pi)
        damp = -sigma/(np.sqrt(sigma**2 + omega**2))
        print(f"Eigenvalue: {lam:.4f}, Frequency: {freq:.4f} Hz, Damping: {damp:.4f}")
    else:
        print(f"Eigenvalue: {lam:.4f}")
print("------------------------------------------------------------------------------")

#------------------------------------------------------------------------------
#Q1.2.1: Manual Excitation, Calculating eignevalues of the system matrix and 
# frequency, damping of the oscillation modes
#------------------------------------------------------------------------------
