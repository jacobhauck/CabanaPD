import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters
folder = 'dogbone_tensile_test'
area = 0.003 * 0.003  # width * height
time_range = slice(15, 70)
velocity = 0.5  # None indicates you should load the positions
dt = 3.0e-8
delta = 0.0005
width = 0.015

# Get total raw force data (not scaled by area)
f_x = np.loadtxt(os.path.join(folder, 'output_force_x.txt'))[:, 0]

# Calculate stress
stress = -f_x / area

# Get average left and right positions
if velocity is None:
    x_left = np.loadtxt(os.path.join(folder, 'output_left_position.txt'))[:, 1]
    x_right = np.loadtxt(os.path.join(folder, 'output_right_position.txt'))[:, 1]
else:
    # Or calculate them
    x_left = np.full(stress.shape, delta)
    x_right = np.arange(len(stress), dtype=float) * dt * velocity + width - delta


# Calculate strain
base_length = x_right[0] - x_left[0]
strain = (x_right - x_left - base_length) / base_length

# Display stress-strain curve
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].plot(strain)
axes[0].set_title('Strain')
axes[0].set_xlabel('Time step index')
axes[0].set_ylabel('Strain')

axes[1].plot(stress)
axes[1].set_title('Stress')
axes[1].set_xlabel('Time step index')
axes[1].set_ylabel('Stress (Pa)')

axes[2].plot(strain[time_range], stress[time_range])
axes[2].set_title('Stress-Strain')
axes[2].set_xlabel('Strain')
axes[2].set_ylabel('Stress (Pa)')
fig.savefig('stress_strain.png', bbox_inches='tight')
plt.show()
