import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('./Assignment_helpfunctions_part_I')
import P_matrix_write as pmw

# Create path for results
def result_path(subfolder, filename):
    folder = os.path.join("./Results", subfolder)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

# Load the system matrix and the state names of one case ('q1a', 'q1b' or 'q1c')
def load_system(case):
    sys_data = sio.loadmat(f'./Assignment_data/system_{case}.mat', squeeze_me=True)
    return sys_data[f'A_{case}'], sys_data[f'names_{case}'], sys_data[f'latex_names_{case}']

# Eigenvalues, right (Phi) and left (Psi) eigenvectors and participation matrix P_ij = phi_ij * psi_ji
def eigen_analysis(A):
    eigen_vals, Phi = np.linalg.eig(A)
    Psi = np.linalg.inv(Phi)
    P = Phi * Psi.T
    return eigen_vals, Phi, Psi, P

# Print the eigenvalues, and the frequency and damping ratio of the oscillating modes
def print_modes(title, eigen_vals):
    print("------------------------------------------------------------------------------")
    print(title)
    for lam in eigen_vals:
        if abs(lam.imag) > 1e-5:
            freq = abs(lam.imag) / (2 * np.pi)
            damp = -lam.real / abs(lam)
            print(f"Eigenvalue: {lam:.4f}, Frequency: {freq:.4f} Hz, Damping: {damp:.4f}")
        else:
            print(f"Eigenvalue: {lam:.4f}")
    print("------------------------------------------------------------------------------")

# Time response to an initial rotor angle perturbation delta0 (rad)
def time_response(Phi, Psi, eigen_vals, t, delta0=0.087266):
    x_nul = np.zeros(Phi.shape[0], dtype=complex)
    x_nul[0] = delta0
    xt = np.zeros((Phi.shape[0], len(t)))
    for k in range(len(t)):
        xt[:,k] = np.real(Phi.dot(np.exp(eigen_vals*t[k])*Psi.dot(x_nul)))
    return xt

t = np.arange(0,5,0.001)

#------------------------------------------------------------------------------
#Q1.1.1: Manual excitation, eigenvalues of the system matrix and
# frequency, damping of the oscillating modes
#------------------------------------------------------------------------------
A, _, _ = load_system('q1a')
eigen_vals, Phi, Psi, P = eigen_analysis(A)
print_modes("Q1.1.1: Manual excitation", eigen_vals)
xt_1 = time_response(Phi, Psi, eigen_vals, t)

#------------------------------------------------------------------------------
#Q1.2.1: Effect of AVR, eigenvalues of the system matrix, frequency, damping
# of the oscillating modes and participation matrix
#------------------------------------------------------------------------------
A, names, latex_names = load_system('q1b')
eigen_vals, Phi, Psi, P = eigen_analysis(A)
print_modes("Q1.2.1: AVR", eigen_vals)

# LaTeX names are used for the .tex table, plain names for the Excel table
pmw.latex_P_matrix(P, latex_names, False, result_path("Results_Q1_2", "P_matrix_Q1_2.tex"), A.shape[0], 0.05)
pmw.excel_P_matrix(P, names, False, result_path("Results_Q1_2", "P_matrix_Q1_2.xls"), 0.05)
xt_2 = time_response(Phi, Psi, eigen_vals, t)

#------------------------------------------------------------------------------
#Q1.3.1: Effect of AVR+PSS, eigenvalues of the system matrix and
# frequency, damping of the oscillating modes
#------------------------------------------------------------------------------
A, names, latex_names = load_system('q1c')
eigen_vals, Phi, Psi, P = eigen_analysis(A)
print_modes("Q1.3.1: AVR + PSS", eigen_vals)

pmw.latex_P_matrix(P, latex_names, False, result_path("Results_Q1_3", "P_matrix_Q1_3.tex"), A.shape[0], 0.05)
pmw.excel_P_matrix(P, names, False, result_path("Results_Q1_3", "P_matrix_Q1_3.xls"), 0.05)
xt_3 = time_response(Phi, Psi, eigen_vals, t)

#------------------------------------------------------------------------------
#Q1.3.2: Rotor angle response of the three cases to delta(0) = 5 deg
#------------------------------------------------------------------------------
plt.figure(figsize=(7, 4), dpi=300)
plt.plot(t, xt_1[0,:], label = "Manual")
plt.plot(t, xt_2[0,:], label = "AVR")
plt.plot(t, xt_3[0,:], label = "PSS + AVR")
plt.grid(True)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel(r"$\Delta\delta$ [rad]")
plt.savefig(result_path("Results_Q1_3", "delta_t_response.pdf"), bbox_inches='tight', dpi=300)
plt.show()
