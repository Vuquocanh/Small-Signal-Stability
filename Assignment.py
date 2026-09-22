import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import P_matrix_write as Pmw
import plot_phasor_diagram_sss as ppd

# Load System Data
sys_data = sio.loadmat('system_q2.mat', squeeze_me=True)
A = sys_data['A']
names_q2a = sys_data['names_q2a']
latex_names_q2a = sys_data['latex_names_q2a']

# ----------------------------------------------------------------------------------------------------------------
# Q.2.1.1: Participation Matrix Calculation & Export
# ----------------------------------------------------------------------------------------------------------------
evals, right_evecs = np.linalg.eig(A)
left_evecs = np.linalg.inv(right_evecs)

# Participation Matrix P_ij = v_ij * w_ji
P = np.zeros(A.shape, dtype=complex)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        P[i, j] = right_evecs[i, j] * left_evecs[j, i]

# Export to LaTeX and Excel for Appendix
Pmw.latex_P_matrix(P, latex_names_q2a, False, 'appendix_P_matrix.tex', 5, 0.05)
Pmw.excel_P_matrix(P, names_q2a, False, 'appendix_P_matrix.xls', 0.05)
print("Q.2.1.1: Participation matrix generated and saved to files.")
# ----------------------------------------------------------------------------------------------------------------
# Q.2.1.2 & Q.2.1.3: Eigenvalues & Mode Summary Table
# ----------------------------------------------------------------------------------------------------------------
print("\nQ.2.1.2 & Q.2.1.3: System Modes Table")
print(f"{'Mode':<6}{'Eigenvalue (lambda)':<28}{'Frequency (Hz)':<16}{'Damping (zeta)':<14}{'Dominant States':<25}")
print("-" * 95)

em_indices = []

for i in range(len(evals)):
    lam = evals[i]
    real = np.real(lam)
    imag = np.imag(lam)

    # Identify dominant states (highest participation factors)
    p_mag = np.abs(P[:, i])
    top_states_idx = np.argsort(p_mag)[-2:][::-1]
    dom_states_str = ", ".join([str(names_q2a[idx]) for idx in top_states_idx])

    if np.abs(imag) > 1e-4:
        wn = np.abs(lam)
        fn = imag / (2 * np.pi)
        zeta = -real / wn

        # Identify electromechanical modes (dominated by rotor angle/speed states)
        is_em = any('delta' in str(names_q2a[idx]).lower() or 'omega' in str(names_q2a[idx]).lower() for idx in top_states_idx)
        if is_em and imag > 0:
            em_indices.append(i)

        print(f"lambda_{i+1:<4}{real:+.4f} +- {np.abs(imag):.4f}j{np.abs(fn):<16.4f}{zeta:<14.4f}{dom_states_str:<25}")
    else:
        print(f"lambda_{i+1:<4}{real:+.4f}{'N/A (Real)':<16}{'N/A (Real)':<14}{dom_states_str:<25}")

# ----------------------------------------------------------------------------------------------------------------
# Q.2.1.4: Plot Mode Shapes for Electromechanical Modes
# ----------------------------------------------------------------------------------------------------------------
delta_indices = [np.where(names_q2a == name)[0][0] for name in ['delta_G1', 'delta_G2', 'delta_G3', 'delta_G4']]

colors = ['#000099', '#ff0000', '#cc0099', '#33ccff']
labels = [r'$\Delta\delta_{G1}$', r'$\Delta\delta_{G2}$', r'$\Delta\delta_{G3}$', r'$\Delta\delta_{G4}$']

for idx in em_indices:
    mode_vec = right_evecs[delta_indices, idx]
    mode_vec = mode_vec / mode_vec[0]

    phasors = np.column_stack((np.real(mode_vec), np.imag(mode_vec)))

    plt.figure(figsize=(4, 4), dpi=300)
    plt.grid(True)
    ppd.plot_phasors(phasors, np.array(colors), np.array(labels))
    plt.title(f"Mode Shape for Mode lambda_{idx+1}")
    plt.savefig(f'mode_shape_lambda_{idx+1}.pdf', bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------
# Q.2.1.5: Inter-Area Mode Time Response Simulation
# ----------------------------------------------------------------------------------------------------------------
inter_area_mode_idx = min(em_indices, key=lambda k: np.abs(np.imag(evals[k])))
print(f"\nQ.2.1.5: Inter-Area Mode identified at lambda_{inter_area_mode_idx+1} = {evals[inter_area_mode_idx]:.4f}")

x0 = np.real(right_evecs[:, inter_area_mode_idx])

t = np.linspace(0, 10, 1000)
x_t = np.zeros((len(A), len(t)))
for i, ti in enumerate(t):
    x_t[:, i] = np.real(right_evecs @ (np.diag(np.exp(evals * ti)) @ np.linalg.solve(right_evecs, x0)))

plt.figure(figsize=(7, 4), dpi=300)
for idx, g_name in zip(delta_indices, ['G1', 'G2', 'G3', 'G4']):
    plt.plot(t, x_t[idx, :], label=rf'$\Delta\delta_{{{g_name}}}$')

plt.xlabel('Time [s]')
plt.ylabel(r'Rotor Angle Deviation $\Delta\delta$ [rad]')
plt.title('Time Response - Inter-Area Mode Excitation')
plt.legend()
plt.grid(True)
plt.savefig('inter_area_time_response.pdf', bbox_inches='tight')
plt.show()