import numpy as np
import matplotlib.pyplot as plt

import sys
grid = sys.argv[1] if len(sys.argv) > 1 else "N"

# Velocity profile data from DNS and CH simulations
DNS_data = np.loadtxt("./DNS_stats/Re4200.prof", skiprows=21)
utau = 0.37145600E-01
tauw_dns = utau**2
y_dns = DNS_data[:, 0]
u_dns = DNS_data[:, 2] * utau
u_dns = u_dns / np.max(u_dns)  # Normalize u_dns by its maximum value

EQWM_data = np.loadtxt(f"./CH_stats/umean_CH4200{grid}_eqwm_tf.cp", skiprows=6)
umean_eqwm = EQWM_data[:, 4]
y_eqwm = EQWM_data[:, 2]
umean_eqwm = umean_eqwm / np.max(umean_eqwm)

BFM_data = np.loadtxt(f"./CH_stats/umean_CH4200{grid}_bfm_tf.cp", skiprows=6)
umean_bfm = BFM_data[:, 4]
y_bfm = BFM_data[:, 2]
umean_bfm = umean_bfm / np.max(umean_bfm)

# bfmbfm_data = np.loadtxt(f"./CH_stats/umean_CH4200{grid}_bfmbfm.cp", skiprows=6)
# umean_bfmbfm = bfmbfm_data[:, 4]
# y_bfmbfm = bfmbfm_data[:, 2]
# umean_bfmbfm = umean_bfmbfm / np.max(umean_bfmbfm)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(umean_eqwm, y_eqwm, 'x', label="EQWM", color='blue', linewidth=2)
ax.plot(umean_bfm, y_bfm, 'o', label="BFM", color='red', linewidth=2)
# ax.plot(umean_bfmbfm, y_bfmbfm, '-o', label="BFM+BFM", color='green', linewidth=2)
ax.plot(u_dns, y_dns, '--', label="DNS", color='black', linewidth=1)

ax.set_xlabel(r"$u/U_c$", fontsize=14)
ax.set_ylabel(r"$y/h$", fontsize=14)
ax.legend()

ax.set_ylim(0, 1)

plt.show()

# Wall shear stress calculation
area = 157.914
EQWM_force = np.loadtxt(f"./CH_stats/force_CH4200{grid}_eqwm_tf.dat", skiprows=3)
tauw_eqwm = EQWM_force[200:-10, 12] / area
BFM_force = np.loadtxt(f"./CH_stats/force_CH4200{grid}_bfm_tf.dat"  , skiprows=3)
tauw_bfm = BFM_force[1990:, 12] / area
# BFMBFM_force = np.loadtxt(f"./CH_stats/force_CH4200{grid}_bfmbfm.dat"  , skiprows=3)
# tauw_bfmbfm = BFMBFM_force[500:-20, 12] / area

fig_tau, ax_tau = plt.subplots(figsize=(8, 6))
ax_tau.plot(tauw_eqwm, color='blue', label='EQWM')
ax_tau.axhline(np.mean(tauw_eqwm), color='blue', linestyle='--', label=f'EQWM, error = {(np.mean(tauw_eqwm) - tauw_dns) / tauw_dns * 100:.2f}%')
ax_tau.plot(tauw_bfm, color='red', label='BFM')
ax_tau.axhline(np.mean(tauw_bfm), color='red', linestyle='--', label=f'BFM, error = {(np.mean(tauw_bfm) - tauw_dns) / tauw_dns * 100:.2f}%')
# ax_tau.plot(tauw_bfmbfm, color='green', label='bfmbfm')
# ax_tau.axhline(np.mean(tauw_bfmbfm), color='green', linestyle='--', label=f'bfmbfm, error = {(np.mean(tauw_bfmbfm) - tauw_dns) / tauw_dns * 100:.2f}%')
ax_tau.axhline(tauw_dns, color='black', linestyle='--', label='DNS')

ax_tau.set_xlabel("Time step", fontsize=14)
ax_tau.set_ylabel(r"$\tau_w$", fontsize=14)
ax_tau.legend()

plt.show()
