import numpy as np
import matplotlib.pyplot as plt
from acceleration_cooking import *
import glob
import os

# Assuming that the functions reshape_acceleration_matrix, calculate_coefficients, and compute_displacement_velocity are defined as in the previous script.


# Function to load all .txt files from a directory
def load_txt_files(directory):
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    data = {}
    for file_path in file_paths:
        data[os.path.basename(file_path)] = np.loadtxt(file_path)
    return data


def main():
    # Load all .txt files from the specified directory
    directory = os.path.curdir
    data = load_txt_files(directory)

    time_step = 0.01  # s

    # Iterate over all loaded data files
    for file_name, accel_matrix in data.items():
        # Reshape the acceleration matrix into vectors
        time_vector, accel_vector = reshape_acceleration_matrix(accel_matrix, time_step)

        # Define structural and material properties
        E = 2.1e6 * 9.81  # N/cm^2
        zeta = 0.05  # Dimensionless damping ratio
        I = 19270  # cm^4
        h = 300  # cm
        L = 4  # m
        k_column = 1200 * E * I / h**3
        k = 2 * k_column  # Total stiffness N/m
        m = np.array([20, 21, 22, 23, 25, 25, 25, 20]) * 1000  # Mass vector in kg
        DoF = len(m)

        # Compute undamped natural frequencies
        Omega = np.sqrt(k / m)

        # Create mass and stiffness matrices
        M = np.diag(m)
        K = np.zeros((DoF, DoF))
        for i in range(DoF):
            if i == DoF - 1:
                K[i, i] = k
                K[i - 1, i] = -k
            elif i == 0:
                K[0, 0] = 2 * k
                K[1, 0] = -k
            else:
                K[i, i] = 2 * k
                K[i - 1, i] = -k
                K[i + 1, i] = -k

        # Solve the eigenvalue problem
        eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.inv(M), K))
        omegas = np.sqrt(eigvals)

        # Plot eigenvectors (mode shapes)
        y = np.arange(1, DoF + 1)
        for i in range(DoF):
            plt.plot(eigvecs[:, i], y, label=f"Mode {i+1}")

        plt.title("Mode Shapes")
        plt.xlabel("Amplitude")
        plt.ylabel("Degree of Freedom")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Participation factors
        partfac = np.zeros(DoF)
        for i in range(DoF):
            vec = eigvecs[:, i]
            partfac[i] = -np.sum(m * vec)

        # Damped natural frequencies
        Wd = omegas * np.sqrt(1 - zeta**2)

        # Compute maximum displacements
        max_displacement = np.zeros(DoF)
        for i in range(DoF):
            Omega = omegas[i]
            mass = m[i]
            Coefs = calculate_coefficients(zeta, Omega, Wd[i], time_step)
            displacements, velocities = compute_displacement_velocity(
                accel_vector * partfac[i], Coefs
            )
            plt.plot(time_vector, displacements, label=f"DOF {i+1}")
            max_displacement[i] = np.max(displacements)

        plt.title(f"Displacement Response for {file_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
