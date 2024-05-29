import numpy as np
import matplotlib.pyplot as plt
import glob
import os


# Function to load all .txt files from a directory
def load_txt_files(directory):
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    if len(file_paths) < 3:
        raise ValueError(
            "Not enough .txt files in the directory. Need at least 3 files."
        )
    data = [np.loadtxt(file_path) for file_path in file_paths[:3]]
    return data


# Function to reshape the acceleration matrix into a vector and create a corresponding time vector
def reshape_acceleration_matrix(accel_matrix, time_step):
    accel_vector = accel_matrix.flatten()
    time_vector = np.arange(len(accel_vector)) * time_step
    return time_vector, accel_vector


# Function to calculate the coefficients needed for computing displacement and velocity
def calculate_coefficients(damping_ratio, angular_freq, damped_angular_freq, time_step):
    coefficients = [
        np.exp(-damping_ratio * angular_freq * time_step)
        * (
            damping_ratio
            * angular_freq
            / damped_angular_freq
            * np.sin(damped_angular_freq * time_step)
            + np.cos(damped_angular_freq * time_step)
        ),
        np.exp(-damping_ratio * angular_freq * time_step)
        * (1 / damped_angular_freq * np.sin(damped_angular_freq * time_step)),
        1
        / angular_freq**2
        * (
            2 * damping_ratio / (angular_freq * time_step)
            + np.exp(-damping_ratio * angular_freq * time_step)
            * (
                (
                    (1 - 2 * damping_ratio**2) / (damped_angular_freq * time_step)
                    - (damping_ratio / np.sqrt(1 - damping_ratio**2))
                )
                * np.sin(damped_angular_freq * time_step)
                - (1 + 2 * damping_ratio / (angular_freq * time_step))
                * np.cos(damped_angular_freq * time_step)
            )
        ),
        1
        / angular_freq**2
        * (
            1
            - 2 * damping_ratio / (angular_freq * time_step)
            + np.exp(-damping_ratio * angular_freq * time_step)
            * (
                (2 * damping_ratio**2 - 1)
                / (angular_freq * time_step)
                * np.sin(damped_angular_freq * time_step)
                + 2
                * damping_ratio
                / (angular_freq * time_step)
                * np.cos(damped_angular_freq * time_step)
            )
        ),
        -np.exp(-damping_ratio * angular_freq * time_step)
        * (
            angular_freq**2
            / damped_angular_freq
            * np.sin(damped_angular_freq * time_step)
        ),
        np.exp(-damping_ratio * angular_freq * time_step)
        * np.cos(damped_angular_freq * time_step)
        - damping_ratio
        * angular_freq
        / damped_angular_freq
        * np.sin(damped_angular_freq * time_step),
        1
        / angular_freq**2
        * (
            np.exp(-damping_ratio * angular_freq * time_step)
            * (
                (
                    angular_freq**2 / damped_angular_freq
                    + angular_freq * damping_ratio / (damped_angular_freq * time_step)
                )
                * np.sin(damped_angular_freq * time_step)
                + 1 / time_step * np.cos(damped_angular_freq * time_step)
            )
            - 1 / time_step
        ),
        1
        / (angular_freq**2 * time_step)
        * (
            -np.exp(-damping_ratio * angular_freq * time_step)
            * (
                angular_freq
                * damping_ratio
                / damped_angular_freq
                * np.sin(damped_angular_freq * time_step)
                + np.cos(damped_angular_freq * time_step)
            )
            + 1
        ),
    ]
    return coefficients


# Function to compute displacement and velocity over time for given acceleration data
def compute_displacement_velocity(acceleration, coefficients):
    displacement = np.zeros_like(acceleration)
    velocity = np.zeros_like(acceleration)
    for i in range(len(acceleration) - 1):
        displacement[i + 1], velocity[i + 1] = compute_next_displacement_velocity(
            coefficients,
            displacement[i],
            velocity[i],
            acceleration[i],
            acceleration[i + 1],
        )
    return displacement, velocity


# Function to compute the next displacement and velocity based on current states and ground accelerations
def compute_next_displacement_velocity(
    constants, displacement_prev, velocity_prev, ground_accel_current, ground_accel_next
):
    A, B, C, D, AA, BB, CC, DD = constants
    force_current = -ground_accel_current
    force_next = -ground_accel_next
    displacement_next = (
        A * displacement_prev + B * velocity_prev + C * force_current + D * force_next
    )
    velocity_next = (
        AA * displacement_prev
        + BB * velocity_prev
        + CC * force_current
        + DD * force_next
    )
    return displacement_next, velocity_next


def main():
    # Load Bam earthquake data from the current directory
    directory = os.path.curdir
    data = load_txt_files(directory)

    # Ensure the correct keys are used
    eq1, eq2, eq3 = data[0], data[1], data[2]

    time_step = 0.01  # s
    _, eq1vec = reshape_acceleration_matrix(eq1, time_step)
    _, eq2vec = reshape_acceleration_matrix(eq2, time_step)
    eq_time_vector, eq3vec = reshape_acceleration_matrix(eq3, time_step)

    # Peak Ground Acceleration (PGA) scaling
    PGA1 = np.max(eq1vec)
    PGA2 = np.max(eq2vec)
    SF1 = 1 / max(PGA1, PGA2)
    eq1vec *= SF1
    eq2vec *= SF1

    # Generate 2800 spectrum
    T0 = 0.15
    Ts = 0.7
    S = 1.75
    S0 = 1.1
    t_step = 0.01
    Tmax = 4
    tvec = np.arange(0, 5 + t_step, t_step)
    B = np.zeros_like(tvec)
    j = 0

    for T in tvec:
        if T <= Ts:
            N = 1
        elif T <= 4:
            N = 0.7 / (4 - Ts) * (T - Ts) + 1
        else:
            N = 1.7
        if T <= T0:
            B[j] = S0 + (S - S0 + 1) * (T / T0)
        elif T <= Ts:
            B[j] = S + 1
        else:
            B[j] = (S + 1) * (Ts / T)
        B[j] *= N
        j += 1

    plt.plot(tvec, B, label="2800 Spectrum")
    plt.legend()
    plt.show()

    # Bam spectrum computation
    su = [0]
    sv = [0]
    zeta = 0.05
    for T in np.arange(0.1, 5.1, 0.1):
        Omega = 2 * np.pi / T
        Wd = Omega * np.sqrt(1 - zeta**2)
        Coefs = calculate_coefficients(zeta, Omega, Wd, time_step)
        Bam1u, Bam1v = compute_displacement_velocity(eq1vec, Coefs)
        Bam2u, Bam2v = compute_displacement_velocity(eq2vec, Coefs)

        plt.plot(eq_time_vector, Bam1u, label=f"Bam1u {T:.2f}")
        plt.plot(eq_time_vector, Bam2u, label=f"Bam2u {T:.2f}")
        su.append(Omega**2 * np.max(Bam1u))
        sv.append(Omega**2 * np.max(Bam2u))

    plt.legend()
    plt.show()

    su = np.array(su[1:])
    sv = np.array(sv[1:])
    sfinal = np.sqrt(su**2 + sv**2)
    Btmp = 1.3 * B

    T = np.arange(0.1, 5.1, 0.1)
    plt.plot(T, su, label="su")
    plt.plot(T, sv, label="sv")
    plt.plot(T, sfinal, label="sfinal")
    plt.plot(tvec, 1.3 * B, label="1.3*B")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
