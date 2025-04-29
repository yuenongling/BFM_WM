import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')
sys.path.append('../../src')
from src.wall_model import WallModel


model_path = "../../models/NN_wm_CH1_G0_S1_TBL1_tn543759_vn135940_fds0_lds0_customw1_inputs2_final_ep8000_tl0.03098575_vl0.03911465.pth"
wall_model = WallModel.load_compact(model_path, device="cpu")


nu = 1/0.11250000E+06
utau = 0.37145600E-01

Y1 = 0.03756574
U1 = 0.65163226
Y2 = 0.11092545
U2 = 0.75515148

input_mean = torch.tensor([U1*Y1/nu, 0, U2*Y1/nu], dtype=torch.float32).unsqueeze(0)

output_mean = wall_model.model(
    torch.tensor(input_mean, dtype=torch.float32))
utau_mean = output_mean[0, 0].item() * nu / Y1

sigma_U = np.linspace(0.01, 1, 100)
# sigma_U = [0.2]
sample_size = 50000

output_mean_vs_sigma = np.zeros(len(sigma_U))
output_std_vs_sigma = np.zeros(len(sigma_U))

#######################
# Test joint Gaussian #
#######################

for i in range(len(sigma_U)):
    sample_gaussian_U1 = np.random.normal(loc=0, scale=sigma_U[i]*U1, size=sample_size)
    U1_sample = sample_gaussian_U1 + U1

    sample_gaussian_U2 = np.random.normal(loc=0, scale=sigma_U[i]*U2, size=sample_size)
    U2_sample = sample_gaussian_U2 + U2

    # Stack the samples for prediction
    Inputs = np.column_stack((U1_sample*Y1/nu, np.zeros(sample_size), U2_sample*Y1/nu))
    Inputs = torch.tensor(Inputs, dtype=torch.float32)

    Output = (wall_model.model(Inputs).detach().numpy() * nu / Y1)**2

    output_mean_vs_sigma[i] = np.mean(Output)
    output_std_vs_sigma[i] = np.std(Output)

# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(sigma_U,(output_mean_vs_sigma-tauw_mean)/tauw_mean*100, label='Mean')
# ax.set_xlabel(r'$\sigma_{U_1} / U_1$, $\sigma_{U_2} / U_2$')
# ax.set_ylabel('Deviation from True mean value (%)')
# # ax[1].plot(sigma_U,output_std_vs_sigma/tauw_mean, label='Std')
# plt.show()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sigma_U,(output_mean_vs_sigma-utau_mean**2)/utau_mean**2*100, '-o')
ax.axhline((utau**2-utau_mean**2)/utau_mean**2 * 100 , color='r', linestyle='--', label='DNS')
ax.set_xlabel(r'$\sigma_{U_1} / U_1$, $\sigma_{U_2} / U_2$')
ax.grid()
ax.set_ylabel('Deviation from True mean value (%)')
ax.set_title('Independent Gaussian')
ax.legend()
# ax[1].plot(sigma_U,output_std_vs_sigma/tauw_mean, label='Std')
plt.show()


#######################
# Test joint Gaussian #
#######################

for i in range(len(sigma_U)):

    sample_gaussian_U = np.random.multivariate_normal(
            mean=[0, 0],
            cov=[[sigma_U[i]**2 * U1**2, sigma_U[i]**2 * U1*U2], [sigma_U[i]**2 * U1*U2, sigma_U[i]**2 * U2**2]],
            size=sample_size
                )
    U1_sample = sample_gaussian_U[:, 0] + U1
    U2_sample = sample_gaussian_U[:, 1] + U2

    # Stack the samples for prediction
    Inputs = np.column_stack((U1_sample*Y1/nu, np.zeros(sample_size), U2_sample*Y1/nu))
    Inputs = torch.tensor(Inputs, dtype=torch.float32)

    Output = (wall_model.model(Inputs).detach().numpy() * nu / Y1)**2

    output_mean_vs_sigma[i] = np.mean(Output)
    output_std_vs_sigma[i] = np.std(Output)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sigma_U,(output_mean_vs_sigma-utau_mean**2)/utau_mean**2*100, '-o')
ax.axhline((utau**2-utau_mean**2)/utau_mean**2 * 100, color='r', linestyle='--', label='DNS')
ax.axhline(0, color='k', linestyle='--', label='Mean prediction')
ax.set_xlabel(r'$\sigma_{U_1} / U_1$, $\sigma_{U_2} / U_2$')
ax.grid()
ax.set_ylabel('Deviation from True mean value (%)')
ax.legend()
ax.set_title('Joint Gaussian')
# ax[1].plot(sigma_U,output_std_vs_sigma/tauw_mean, label='Std')
plt.show()
