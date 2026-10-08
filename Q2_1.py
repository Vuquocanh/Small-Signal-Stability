import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('./Assignment_helpfunctions_part_I')
import P_matrix_write as Pmw
import plot_phasor_diagram_sss as ppd
sys.path.append('./Assignment_helpfunctions_part_II')

# Load System Data
sys_data = sio.loadmat('./Assignment_data/system_q2.mat', squeeze_me=True)
A = sys_data['A']
names_q2a = sys_data['names_q2a']
latex_names_q2a = sys_data['latex_names_q2a']

# Create path for results
def result_path(subfolder, filename):
    folder = os.path.join("./Results", subfolder)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

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
Pmw.latex_P_matrix(P, latex_names_q2a, False, result_path("Results_Q2_1", "P_matrix.tex"), 5, 0.05)
Pmw.excel_P_matrix(P, names_q2a, False, result_path("Results_Q2_1", "P_matrix.xls"), 0.05)
print("Q.2.1.1: Participation matrix generated and saved to files.")
# ----------------------------------------------------------------------------------------------------------------
# Q.2.1.2 & Q.2.1.3: Eigenvalues & Mode Summary Table
# ----------------------------------------------------------------------------------------------------------------
print("\nQ.2.1.2 & Q.2.1.3: System Modes Table")
print(f"{'Mode':<6}{'Eigenvalue (lambda)':<28}{'Frequency (Hz)':<16}{'Damping (zeta)':<14}{'Dominant States':<25}")
print("-" * 95)

em_indices = []
em_zetas = {}
zero_indices = []

for i in range(len(evals)):
    lam = evals[i]
    real = np.real(lam)
    imag = np.imag(lam)

    # Identify dominant states (highest participation factors)
    p_mag = np.abs(P[:, i])
    top_states_idx = np.argsort(p_mag)[-2:][::-1]
    dom_states_str = ", ".join([str(names_q2a[idx]) for idx in top_states_idx])

    # Zero-modes show up numerically as |lambda| ~ 0, possibly as a tiny complex pair
    if np.abs(lam) < 1e-2:
        zero_indices.append(i)
        print(f"lambda_{i+1:<4}{real:+.4f} +- {np.abs(imag):.4f}j{'Zero-mode':<16}{'Zero-mode':<14}{dom_states_str:<25}")
    elif np.abs(imag) > 1e-4:
        wn = np.abs(lam)
        fn = imag / (2 * np.pi)
        zeta = -real / wn

        # Identify electromechanical modes (dominated by rotor angle/speed states)
        is_em = any('delta' in str(names_q2a[idx]).lower() or 'omega' in str(names_q2a[idx]).lower() for idx in top_states_idx)
        if is_em and imag > 0:
            em_indices.append(i)
            em_zetas[i] = zeta

        print(f"lambda_{i+1:<4}{real:+.4f} +- {np.abs(imag):.4f}j{np.abs(fn):<16.4f}{zeta:<14.4f}{dom_states_str:<25}")
    else:
        print(f"lambda_{i+1:<4}{real:+.4f}{'N/A (Real)':<16}{'N/A (Real)':<14}{dom_states_str:<25}")

# Flag electromechanical modes with critically low damping (zeta <= 0.1)
print("\nElectromechanical modes with critically low damping (zeta <= 0.1):")
critical = [i for i in em_indices if em_zetas[i] <= 0.1]
if critical:
    for i in critical:
        print(f"  lambda_{i+1}: zeta = {em_zetas[i]:.4f}")
else:
    print("  None")

# Zero-modes: eigenvalues ~0 arise because the model has no absolute angle
# reference, so a uniform shift of all rotor angles leaves the system unchanged.
print("\nZero-modes (lambda approx. 0):")
if zero_indices:
    for i in zero_indices:
        print(f"  lambda_{i+1} = {evals[i]:.4f} -- dominant states: "
              f"{', '.join([str(names_q2a[idx]) for idx in np.argsort(np.abs(P[:, i]))[-2:][::-1]])}")
else:
    print("  None")

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
    plt.savefig(result_path("Results_Q2_1", f"mode_shape_lambda_{idx+1}.pdf"), bbox_inches='tight')
    # plt.show()
    
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
plt.savefig(result_path("Results_Q2_1", "inter_area_time_response.pdf"), bbox_inches='tight')
# plt.show()
print("------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------
# Q.2.2.1: Determining Suitable Locations for PSS Installation
# ----------------------------------------------------------------------------------------------------------------
print("\nQ.2.2.1: PSS location analysis for the electromechanical modes")

# Input/output and state indices (strsps uses MATLAB 1-based indexing)
B = sys_data['B']
C = sys_data['C']
strsps = sys_data['strsps']
vref_indices = np.asarray(strsps['vref'].item(), dtype=int).ravel() - 1    # PSS input: V_ref of G1..G4 (columns of B)
speed_indices = np.asarray(strsps['speed'].item(), dtype=int).ravel() - 1  # PSS measurement: speed of G1..G4 (rows of C)
omega_indices = [np.where(names_q2a == name)[0][0] for name in ['omega_G1', 'omega_G2', 'omega_G3', 'omega_G4']]

# Participation of the generator speed states in each electromechanical mode,
# normalized so that the most participating generator of each mode equals 1
P_red = np.abs(P[np.ix_(omega_indices, em_indices)])
P_red = P_red / P_red.max(axis=0)

# Mode observability from the speed outputs and controllability from the V_ref inputs
obs = C[speed_indices, :] @ right_evecs[:, em_indices]   # (generators x modes)
contr = left_evecs[em_indices, :] @ B[:, vref_indices]   # (modes x generators)

# Residue R_ki = (C_k phi_i)(psi_i B_k): effect of a PSS at generator k on mode i
residue = obs * contr.T
residue_mag = np.abs(residue)
residue_angle = np.angle(residue, deg=True)   
residue_norm = residue_mag / residue_mag.max(axis=0)

gen_labels = ['G1', 'G2', 'G3', 'G4']
mode_labels = [f"lambda_{i+1}" for i in em_indices]
tables = [("Normalized participation factors |P| (speed states)", P_red, ".3f"),
          ("Normalized residues |R|", residue_norm, ".3f"),
          ("Residue angles [deg]", residue_angle, ".1f")]

for title, table, fmt in tables:
    print(f"\n{title}")
    print(f"{'Gen':<6}" + "".join(f"{m:>12}" for m in mode_labels))
    for k, g in enumerate(gen_labels):
        print(f"{g:<6}" + "".join(f"{v:>12{fmt}}" for v in table[k]))

print("\nMost effective PSS location per mode (largest residue):")
for col, i in enumerate(em_indices):
    print(f"  lambda_{i+1} = {evals[i]:.4f}: {gen_labels[np.argmax(residue_norm[:, col])]}")
