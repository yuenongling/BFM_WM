import numpy as np
import matplotlib.pyplot as plt

import sys
grid = sys.argv[1] if len(sys.argv) > 1 else "N"
model = sys.argv[2] if len(sys.argv) > 2 else "bfm"

if model == "bfm":
    tauw_start_tf = 1700
    tauw_start    = 4000
elif model == "eqwm":
    tauw_start_tf = 100
    tauw_start    = 800
else:
    raise ValueError("Model must be 'bfm' or 'eqwm'.")
    

# Velocity profile data from DNS and CH simulations
DNS_data = np.loadtxt("./DNS_stats/Re4200.prof", skiprows=21)
utau = 0.37145600E-01
tauw_dns = utau**2
y_dns = DNS_data[:, 0]
u_dns = DNS_data[:, 2] * utau
u_dns = u_dns / np.max(u_dns)  # Normalize u_dns by its maximum value

TF_data = np.loadtxt(f"./CH_stats/umean_CH4200{grid}_{model}_tf.cp", skiprows=6)
umean_bfm_tf = TF_data[:, 4]
y_bfm_tf = TF_data[:, 2]
umean_bfm_tf = umean_bfm_tf / np.max(umean_bfm_tf)

data = np.loadtxt(f"./CH_stats/umean_CH4200{grid}_{model}.cp", skiprows=6)
umean_bfm = data[:, 4]
y_bfm = data[:, 2]
umean_bfm = umean_bfm / np.max(umean_bfm)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(umean_bfm_tf, y_bfm_tf, '-x', label=f"{model} Time Filtered", color='blue', linewidth=2)
ax.plot(umean_bfm, y_bfm, '-o', label=model, color='red', linewidth=2)
ax.plot(u_dns, y_dns, '--', label="DNS", color='black', linewidth=1)

ax.set_xlabel(r"$u/U_c$", fontsize=14)
ax.set_ylabel(r"$y/h$", fontsize=14)
ax.legend()

ax.set_ylim(0, 1)

plt.show()

# Wall shear stress calculation
area = 157.914
TF_force = np.loadtxt(f"./CH_stats/force_CH4200{grid}_{model}_tf.dat", skiprows=3)
tauw_bfm_tf = TF_force[tauw_start_tf:, 12] / area
force = np.loadtxt(f"./CH_stats/force_CH4200{grid}_{model}.dat"  , skiprows=3)
tauw_bfm = force[tauw_start:, 12] / area
# BFMBFM_force = np.loadtxt(f"./CH_stats/force_CH4200{grid}_bfmbfm.dat"  , skiprows=3)
# tauw_bfmbfm = BFMBFM_force[500:-20, 12] / area

fig_tau, ax_tau = plt.subplots(figsize=(8, 6))
ax_tau.plot(tauw_bfm_tf, color='blue', label=f'{model} Time Filtered')
ax_tau.axhline(np.mean(tauw_bfm_tf), color='blue', linestyle='--', label=f'{model} Time Filtered, error = {(np.mean(tauw_bfm_tf) - tauw_dns) / tauw_dns * 100:.2f}%')
ax_tau.plot(tauw_bfm, color='red', label=model)
ax_tau.axhline(np.mean(tauw_bfm), color='red', linestyle='--', label=f'{model}, error = {(np.mean(tauw_bfm) - tauw_dns) / tauw_dns * 100:.2f}%')
# ax_tau.plot(tauw_bfmbfm, color='green', label='bfmbfm')
# ax_tau.axhline(np.mean(tauw_bfmbfm), color='green', linestyle='--', label=f'bfmbfm, error = {(np.mean(tauw_bfmbfm) - tauw_dns) / tauw_dns * 100:.2f}%')
ax_tau.axhline(tauw_dns, color='black', linestyle='--', label='DNS')

ax_tau.set_xlabel("Time step", fontsize=14)
ax_tau.set_ylabel(r"$\tau_w$", fontsize=14)
ax_tau.legend()

plt.show()
