import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import os
import sys
sys.path.append('./Assignment_helpfunctions_part_I')
import P_matrix_write as pmw

# Create path for results
def result_path(subfolder, filename):
    folder = os.path.join("./Results", subfolder)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

# Time response to an initial rotor angle perturbation delta0 (rad)
def time_response(Phi, Psi, eigen_vals, t, delta0=0.087266):
    x_nul = np.zeros(Phi.shape[0], dtype=complex)
    x_nul[0] = delta0
    xt = np.zeros((Phi.shape[0], len(t)))
    for k in range(len(t)):
        xt[:,k] = np.real(Phi.dot(np.exp(eigen_vals*t[k])*Psi.dot(x_nul)))
    return xt

#------------------------------------------------------------------------------
#Q1.1.1: Manual Excitation, Calculating eignevalues of the system matrix and 
# frequency, damping of the oscillation modes
#------------------------------------------------------------------------------
sys_data = sio.loadmat('./Assignment_data/system_q1a.mat', squeeze_me=True)
matrix_A = sys_data['A_q1a']
names = sys_data['names_q1a']
latex_names = sys_data['latex_names_q1a']

print("------------------------------------------------------------------------------")
print("Q1.1.1: Calculating eigenvalues of the system matrix and frequency, damping")
eigen_vals, Phi = np.linalg.eig(matrix_A)
Psi = np.linalg.inv(Phi)
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

# Time response plotting
t = np.arange(0,5,0.001)
xt_1 = time_response(Phi, Psi, eigen_vals, t)

#------------------------------------------------------------------------------
#Q1.2.1: Effect of AVR, Calculating eignevalues of the system matrix and 
# frequency, damping of the oscillation modes
#------------------------------------------------------------------------------
sys_data = sio.loadmat('./Assignment_data/system_q1b.mat', squeeze_me=True)
matrix_A = sys_data['A_q1b']
names = sys_data['names_q1b']
latex_names = sys_data['latex_names_q1b']

print("------------------------------------------------------------------------------")
print("Q1.2.1: Considering AVR, calculating eigenvalues of the system matrix and frequency, damping")
eigen_vals, Phi = np.linalg.eig(matrix_A)
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

#Participation Matrix P_ij = phi_ij * psi_ji
A_size = matrix_A.shape
P = np.zeros(matrix_A.shape, dtype = complex)
Psi = np.linalg.inv(Phi)
for i in range(A_size[0]):
    for j in range(A_size[1]):
        P[i, j] = Phi[i, j] * Psi[j, i]
        
pmw.latex_P_matrix(P, names, False, result_path("Results_Q1_2", "P_matrix_Q1_2.tex"), A_size[0], 0.05)
pmw.excel_P_matrix(P, names, False, result_path("Results_Q1_2", "P_matrix_Q1_2.xls"), 0.05)

# Time response plotting
xt_2 = time_response(Phi, Psi, eigen_vals, t)

#------------------------------------------------------------------------------
#Q1.3.1: Effect of AVR+PSS, Calculating eignevalues of the system matrix and 
# frequency, damping of the oscillation modes
#------------------------------------------------------------------------------
sys_data = sio.loadmat('./Assignment_data/system_q1c.mat', squeeze_me=True)
matrix_A = sys_data['A_q1c']
names = sys_data['names_q1c']
latex_names = sys_data['latex_names_q1c']

print("------------------------------------------------------------------------------")
print("Q1.3.1: Considering PSS, calculating eigenvalues of the system matrix and frequency, damping")
eigen_vals, Phi = np.linalg.eig(matrix_A)
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

#Participation Matrix P_ij = phi_ij * psi_ji
A_size = matrix_A.shape
P = np.zeros(matrix_A.shape, dtype = complex)
Psi = np.linalg.inv(Phi)
for i in range(A_size[0]):
    for j in range(A_size[1]):
        P[i, j] = Phi[i, j] * Psi[j, i]
        
pmw.latex_P_matrix(P, names, False, result_path("Results_Q1_3", "P_matrix_Q1_3.tex"), A_size[0], 0.05)
pmw.excel_P_matrix(P, names, False, result_path("Results_Q1_3", "P_matrix_Q1_3.xls"), 0.05)

# Time response plotting
xt_3 = time_response(Phi, Psi, eigen_vals, t)

plt.plot(t, xt_1[0,:], label = "Manual")
plt.plot(t, xt_2[0,:], label = "AVR")
plt.plot(t, xt_3[0,:], label = "PSS + AVR")
plt.grid(True)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel(r"$\Delta\delta$ [rad]")
plt.savefig(result_path("Results_Q1_3", "delta_t_response.pdf"), bbox_inches='tight', dpi=300)
plt.show()