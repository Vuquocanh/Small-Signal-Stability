# 46711 Assignment 1 - Task 1

# import libraries
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# import provided functions
from P_matrix_write import latex_P_matrix, excel_P_matrix

# initial rotor angle disturbance and time vector
delta_0 = np.deg2rad(5)
t = np.linspace(0, 5, 2000)

# function to load the A matrix and the state names
def load_system(filename, matrix_name, name_variable):
    data = scipy.io.loadmat(filename)
    A = data[matrix_name]

    # unwrap state names so they arent nested arrays
    names = []
    for name in data[name_variable].flatten():
        while isinstance(name, np.ndarray):
            name = name.flatten()[0]
        names.append(str(name))

    return A, names

# function to find the eigenvalues, damping ratios, frequencies and participation matrix
def find_eigenvalues(A):
    # right eigenvectors are the columns of V and left eigenvectors are the rows of W
    eigenvalues, V = np.linalg.eig(A)
    W = np.linalg.inv(V)

    # damping ratio and frequency in hertz
    damping_ratios = -np.real(eigenvalues) / np.abs(eigenvalues)
    frequencies = np.abs(np.imag(eigenvalues)) / (2 * np.pi)

    # participation matrix
    P = V * W.T

    return eigenvalues, damping_ratios, frequencies, V, W, P

# function to print eigenvalue results
def print_eigenvalues(title, eigenvalues, damping_ratios, frequencies):
    print("\n" + title)
    print("Mode, Eigenvalue, Damping ratio, Frequency [Hz]")

    for i in range(len(eigenvalues)):
        print(i + 1, np.round(eigenvalues[i], 4), np.round(damping_ratios[i], 4), np.round(frequencies[i], 4))

# function to find rotor angle response using x(t) = V*exp(lambda*t)*W*x(0)
def find_delta_response(eigenvalues, V, W):
    x_0 = np.zeros(len(eigenvalues))
    x_0[0] = delta_0

    delta_values = np.zeros(len(t))

    for i in range(len(t)):
        x = V @ (np.exp(eigenvalues * t[i]) * (W @ x_0))
        delta_values[i] = np.real(x[0])

    return np.rad2deg(delta_values)

# load the systems, the latex names are used for the latex tables and the normal names for excel
A_q1a, names_q1a = load_system("Assignment_data/system_q1a.mat", "A_q1a", "names_q1a")
A_q1a, latex_names_q1a = load_system("Assignment_data/system_q1a.mat", "A_q1a", "latex_names_q1a")

A_q1b, names_q1b = load_system("Assignment_data/system_q1b.mat", "A_q1b", "names_q1b")
A_q1b, latex_names_q1b = load_system("Assignment_data/system_q1b.mat", "A_q1b", "latex_names_q1b")

A_q1c, names_q1c = load_system("Assignment_data/system_q1c.mat", "A_q1c", "names_q1c")
A_q1c, latex_names_q1c = load_system("Assignment_data/system_q1c.mat", "A_q1c", "latex_names_q1c")

# Q1.1 manual excitation
eig_q1a, damping_q1a, frequency_q1a, V_q1a, W_q1a, P_q1a = find_eigenvalues(A_q1a)
print_eigenvalues("Q1.1 - Manual excitation", eig_q1a, damping_q1a, frequency_q1a)

# Q1.2 AVR
eig_q1b, damping_q1b, frequency_q1b, V_q1b, W_q1b, P_q1b = find_eigenvalues(A_q1b)
print_eigenvalues("Q1.2 - AVR", eig_q1b, damping_q1b, frequency_q1b)

# Q1.3 AVR + PSS
eig_q1c, damping_q1c, frequency_q1c, V_q1c, W_q1c, P_q1c = find_eigenvalues(A_q1c)
print_eigenvalues("Q1.3 - AVR + PSS", eig_q1c, damping_q1c, frequency_q1c)

# write the participation matrices using the provided functions
latex_P_matrix(P_q1a, latex_names_q1a, False, "P_q1a.tex", P_q1a.shape[1], 0.5)
excel_P_matrix(P_q1a, names_q1a, False, "P_q1a.xlsx", 0.5)

latex_P_matrix(P_q1b, latex_names_q1b, False, "P_q1b.tex", P_q1b.shape[1], 0.5)
excel_P_matrix(P_q1b, names_q1b, False, "P_q1b.xlsx", 0.5)

latex_P_matrix(P_q1c, latex_names_q1c, False, "P_q1c.tex", P_q1c.shape[1], 0.5)
excel_P_matrix(P_q1c, names_q1c, False, "P_q1c.xlsx", 0.5)

# Q1.3.2 rotor angle response for the three cases
delta_q1a = find_delta_response(eig_q1a, V_q1a, W_q1a)
delta_q1b = find_delta_response(eig_q1b, V_q1b, W_q1b)
delta_q1c = find_delta_response(eig_q1c, V_q1c, W_q1c)

# plot the responses
plt.figure(figsize = (10, 6))
plt.plot(t, delta_q1a, label = "Manual excitation")
plt.plot(t, delta_q1b, label = "AVR")
plt.plot(t, delta_q1c, label = "AVR + PSS")
plt.xlabel("Time [s]")
plt.ylabel(r"$\Delta\delta$ [degrees]")
plt.title("Rotor-angle response to an initial 5° disturbance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("task1_rotor_angle_response.png", dpi = 300)
plt.show()