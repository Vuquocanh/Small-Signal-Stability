import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
import control.statesp as stsp
import matplotlib.pyplot as plt
import itertools
import sys
import os
sys.path.append('./Assignment_helpfunctions_part_II')
import stsp_functions as f_stsp

# Load System Data
sys_data = sio.loadmat('./Assignment_data/system_q2.mat', squeeze_me=True)
A = sys_data['A']
B = sys_data['B']
C = sys_data['C']
D = sys_data['D']
strsps = sys_data['strsps']
statename = sys_data['StateName']
names_q2a = sys_data['names_q2a']

# Create path for results
def result_path(subfolder, filename):
    folder = os.path.join("./Results", subfolder)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

# ----------------------------------------------------------------------------------------------------------------
# Indices, PSS parameters and helper functions
# ----------------------------------------------------------------------------------------------------------------
gen_labels = ['G1', 'G2', 'G3', 'G4']
delta_indices = [np.where(names_q2a == f'delta_{g}')[0][0] for g in gen_labels]
omega_indices = [np.where(names_q2a == f'omega_{g}')[0][0] for g in gen_labels]

# Input/output indices (strsps uses MATLAB 1-based indexing)
vref_indices = np.asarray(strsps['vref'].item(), dtype=int).ravel() - 1    # PSS input: V_ref of G1..G4 (columns of B)
speed_indices = np.asarray(strsps['speed'].item(), dtype=int).ravel() - 1  # PSS measurement: speed of G1..G4 (rows of C)
pe_indices = np.asarray(strsps['pe'].item(), dtype=int).ravel() - 1        # electrical power of G1..G4 (rows of C)

# Rotor angle and speed states of the original 28-state system
mech = np.logical_or(statename == 'angle', statename == 'speed')
keep = np.logical_not(mech)

# PSS design parameters
Ks=20;Tw=10;Tn1=.05;Td1=.02;Tn2=3;Td2=5.4; #initial PSS parameters provided in the assignment

def system_with_pss(gens, params):
    sys_pss = stsp.StateSpace(A, B, C, D)
    for g in gens:
        sys_pss = f_stsp.addPSS(sys_pss, f_stsp.pss_stsp(Ks, *params), strsps, g)
    return sys_pss

def electromechanical_modes(A_sys):
    # Oscillatory modes (f >= 0.1 Hz, which excludes the zero-mode) whose most
    # participating state is a rotor angle or speed of the original system
    ev, V = np.linalg.eig(A_sys)
    P_sys = np.abs(V * np.linalg.inv(V).T)
    modes = [k for k in range(len(ev))
             if np.imag(ev[k]) / (2 * np.pi) >= 0.1 and mech[np.argmax(P_sys[:len(mech), k])]]
    return sorted(modes, key=lambda k: np.imag(ev[k])), ev, V

def print_modes(label, A_sys):
    modes, ev, _ = electromechanical_modes(A_sys)
    zetas = [-np.real(ev[k]) / np.abs(ev[k]) for k in modes]
    modes_str = "  ".join(f"{np.imag(ev[k]) / (2 * np.pi):.3f} Hz (zeta = {z:.3f})" for k, z in zip(modes, zetas))
    status = "OK" if min(zetas) > 0.1 else "zeta <= 0.1"
    print(f"  {label:<32}{modes_str}   -> {status}")

# Eigen-analysis of the system without PSS (same system as in Q.2.1)
em_indices, evals, right_evecs = electromechanical_modes(A)
left_evecs = np.linalg.inv(right_evecs)
P = right_evecs * left_evecs.T   # participation matrix P_ij = phi_ij * psi_ji

# ----------------------------------------------------------------------------------------------------------------
# Q.2.2.1: Determining Suitable Locations for PSS Installation
# ----------------------------------------------------------------------------------------------------------------
print("\nQ.2.2.1: PSS location analysis for the electromechanical modes")

# Participation of the generator speed states in each electromechanical mode,
# normalized so that the most participating generator of each mode equals 1
P_abs = np.abs(P[np.ix_(omega_indices, em_indices)])
P_red = P_abs / P_abs.max(axis=0)

# Mode observability from the speed outputs and controllability from the V_ref inputs
obs = C[speed_indices, :] @ right_evecs[:, em_indices]   # (generators x modes)
contr = left_evecs[em_indices, :] @ B[:, vref_indices]   # (modes x generators)

# Residue R_ki = (C_k phi_i)(psi_i B_k): effect of a PSS at generator k on mode i
residue = obs * contr.T
residue_mag = np.abs(residue)
residue_angle = np.angle(residue, deg=True)
residue_norm = residue_mag / residue_mag.max(axis=0)

mode_labels = [f"lambda_{i+1}" for i in em_indices]
tables = [("Participation factors |P| (speed states)", P_abs, ".4f"),
          ("Normalized participation factors |P| (speed states)", P_red, ".3f"),
          ("Residues |R|", residue_mag, ".4f"),
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

# ----------------------------------------------------------------------------------------------------------------
# Q.2.2.2: Obtain the phase response of the system
# ----------------------------------------------------------------------------------------------------------------
# Parameters for frequency sweep
fmin = 0 ; fmax = 2
f = np.arange(fmin,fmax,.001)

# Phase of the transfer function V_ref -> P_e of the same generator, for every candidate location.
# The rotor angle and speed states are removed, so the response only contains the
# excitation/electrical path that the PSS has to compensate (no shaft dynamics)
sys_phase = np.zeros((len(gen_labels), len(f)))
for k in range(len(gen_labels)):
    sys_red = stsp.StateSpace(A[np.ix_(keep, keep)],
                              B[np.ix_(keep, [vref_indices[k]])],
                              C[np.ix_([pe_indices[k]], keep)],
                              D[np.ix_([pe_indices[k]], [vref_indices[k]])])
    mag, phase, omega = sys_red.frequency_response(2 * np.pi * f)
    sys_phase[k, :] = np.degrees(np.squeeze(phase))

# The PSS has to provide the inverted system phase (phase lead) to give a torque in phase with speed
plt.figure(figsize=(7, 4), dpi=300)
for k, g in enumerate(gen_labels):
    plt.plot(f, -sys_phase[k, :], label=g)
for i in em_indices:
    plt.axvline(np.imag(evals[i]) / (2 * np.pi), color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Required phase compensation [deg]')
plt.title(r'Inverted phase of $V_{ref} \rightarrow P_e$ (dashed: electromechanical modes)')
plt.legend()
plt.grid(True)
plt.savefig(result_path("Results_Q2_2", "system_phase_response.pdf"), bbox_inches='tight')
# plt.show()

# ----------------------------------------------------------------------------------------------------------------
# Q.2.2.3: Tuning of the PSS phase compensation
# ----------------------------------------------------------------------------------------------------------------
print("\nQ.2.2.3: PSS phase compensation tuning")

# PSS locations chosen from the residues in Q.2.2.1 (generator numbers, 1-based as used by addPSS):
# G2 -> local mode area 1, G4 -> local mode area 2 and inter-area mode, G1 -> area 1 and inter-area mode
pss_gens = [1, 2, 4]
param_names = ['Tw', 'Tn1', 'Td1', 'Tn2', 'Td2']
params_init = (Tw, Tn1, Td1, Tn2, Td2)

def pss_phase(params, freqs):
    pss = f_stsp.pss_stsp(Ks, *params)
    _, phase, _ = pss.frequency_response(2 * np.pi * freqs)
    return np.degrees(np.squeeze(phase))

# Required compensation = inverted V_ref -> P_e phase, averaged over the chosen locations
band = f >= 0.1
f_band = f[band]
required_phase = -sys_phase[[g - 1 for g in pss_gens]][:, band].mean(axis=0)

# Least-squares fit of the PSS phase to the required compensation over the relevant frequency range
# (Tw limited to 10-30 s by the assignment, lead-lag time constants kept >= 0.01 s)
fit = least_squares(lambda p: pss_phase(p, f_band) - required_phase,
                    x0=[10, 0.15, 0.05, 0.15, 0.05],
                    bounds=([10, 0.01, 0.01, 0.01, 0.01], [30, 2, 2, 2, 2]))
params_tuned = tuple(fit.x)

print(f"{'':<10}" + "".join(f"{n:>9}" for n in param_names))
for label, params in [("Initial", params_init), ("Tuned", params_tuned)]:
    mismatch = np.abs(pss_phase(params, f_band) - required_phase).max()
    print(f"{label:<10}" + "".join(f"{v:>9.4f}" for v in params) + f"   max phase mismatch = {mismatch:.1f} deg")

plt.figure(figsize=(7, 4), dpi=300)
plt.plot(f_band, required_phase, 'k', label='Required compensation')
plt.plot(f_band, pss_phase(params_init, f_band), '--', label='PSS, initial settings')
plt.plot(f_band, pss_phase(params_tuned, f_band), label='PSS, tuned settings')
for i in em_indices:
    plt.axvline(np.imag(evals[i]) / (2 * np.pi), color='gray', linestyle=':', linewidth=0.8)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [deg]')
plt.title('PSS phase compensation (dotted: electromechanical modes)')
plt.legend()
plt.grid(True)
plt.savefig(result_path("Results_Q2_2", "pss_phase_compensation.pdf"), bbox_inches='tight')
# plt.show()

# ----------------------------------------------------------------------------------------------------------------
# Q.2.2.4: Performance of the power system stabilizers
# ----------------------------------------------------------------------------------------------------------------
print("\nQ.2.2.4: Electromechanical modes with PSS")

print_modes("No PSS", A)

print("\n1) Single PSS (tuned settings):")
for g in pss_gens:
    print_modes(f"PSS at G{g}", system_with_pss([g], params_tuned).A)

print("\n2) Two PSS (tuned settings):")
for gens in itertools.combinations(pss_gens, 2):
    print_modes("PSS at " + " + ".join(f"G{g}" for g in gens), system_with_pss(gens, params_tuned).A)

# Two PSS (one per area) are sufficient; G1 + G4 gives the largest margin on the inter-area mode
final_gens = [1, 4]

print("\n3) Final solution, initial vs tuned settings:")
gens_str = " + ".join(f"G{g}" for g in final_gens)
print_modes(f"{gens_str}, initial", system_with_pss(final_gens, params_init).A)
print_modes(f"{gens_str}, tuned", system_with_pss(final_gens, params_tuned).A)

# ----------------------------------------------------------------------------------------------------------------
# Q.2.2.5: Inter-area mode time response with the final PSS solution
# ----------------------------------------------------------------------------------------------------------------
A_final = system_with_pss(final_gens, params_tuned).A
modes_final, evals_final, evecs_final = electromechanical_modes(A_final)
inter_area_final = modes_final[0]   # lowest frequency electromechanical mode
print(f"\nQ.2.2.5: Inter-area mode with PSS at lambda = {evals_final[inter_area_final]:.4f}")

# Excite only the inter-area mode and propagate x(t) = Phi exp(Lambda t) Psi x0
t = np.linspace(0, 10, 1000)   # same time axis as Q.2.1.5
x0_final = np.real(evecs_final[:, inter_area_final])
coeff = np.linalg.solve(evecs_final, x0_final)
x_t_final = np.real(evecs_final @ (np.exp(np.outer(evals_final, t)) * coeff[:, None]))

plt.figure(figsize=(7, 4), dpi=300)
for idx, g_name in zip(delta_indices, gen_labels):
    plt.plot(t, x_t_final[idx, :], label=rf'$\Delta\delta_{{{g_name}}}$')
plt.xlabel('Time [s]')
plt.ylabel(r'Rotor Angle Deviation $\Delta\delta$ [rad]')
plt.title('Time Response - Inter-Area Mode Excitation with PSS')
plt.legend()
plt.grid(True)
plt.savefig(result_path("Results_Q2_2", "inter_area_time_response_pss.pdf"), bbox_inches='tight')
# plt.show()
