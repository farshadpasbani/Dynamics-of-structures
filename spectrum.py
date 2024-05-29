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
def reshape_acceleration_matrix(acceleration_matrix, time_step):
    # Flatten the 2D acceleration matrix into a 1D vector
    acceleration_vector = acceleration_matrix.flatten()
    # Create a time vector based on the number of data points and the time step
    time_vector = np.arange(len(acceleration_vector)) * time_step
    return time_vector, acceleration_vector


# Function to calculate the coefficients needed for computing displacement and velocity
def calculate_coefficients(
    damping_ratio, angular_frequency, damped_angular_frequency, time_step
):
    coefficients = [
        np.exp(-damping_ratio * angular_frequency * time_step)
        * (
            damping_ratio
            * angular_frequency
            / damped_angular_frequency
            * np.sin(damped_angular_frequency * time_step)
            + np.cos(damped_angular_frequency * time_step)
        ),
        np.exp(-damping_ratio * angular_frequency * time_step)
        * (1 / damped_angular_frequency * np.sin(damped_angular_frequency * time_step)),
        1
        / angular_frequency**2
        * (
            2 * damping_ratio / (angular_frequency * time_step)
            + np.exp(-damping_ratio * angular_frequency * time_step)
            * (
                (
                    (1 - 2 * damping_ratio**2) / (damped_angular_frequency * time_step)
                    - (damping_ratio / np.sqrt(1 - damping_ratio**2))
                )
                * np.sin(damped_angular_frequency * time_step)
                - (1 + 2 * damping_ratio / (angular_frequency * time_step))
                * np.cos(damped_angular_frequency * time_step)
            )
        ),
        1
        / angular_frequency**2
        * (
            1
            - 2 * damping_ratio / (angular_frequency * time_step)
            + np.exp(-damping_ratio * angular_frequency * time_step)
            * (
                (2 * damping_ratio**2 - 1)
                / (angular_frequency * time_step)
                * np.sin(damped_angular_frequency * time_step)
                + 2
                * damping_ratio
                / (angular_frequency * time_step)
                * np.cos(damped_angular_frequency * time_step)
            )
        ),
        -np.exp(-damping_ratio * angular_frequency * time_step)
        * (
            angular_frequency**2
            / damped_angular_frequency
            * np.sin(damped_angular_frequency * time_step)
        ),
        np.exp(-damping_ratio * angular_frequency * time_step)
        * np.cos(damped_angular_frequency * time_step)
        - damping_ratio
        * angular_frequency
        / damped_angular_frequency
        * np.sin(damped_angular_frequency * time_step),
        1
        / angular_frequency**2
        * (
            np.exp(-damping_ratio * angular_frequency * time_step)
            * (
                (
                    angular_frequency**2 / damped_angular_frequency
                    + angular_frequency
                    * damping_ratio
                    / (damped_angular_frequency * time_step)
                )
                * np.sin(damped_angular_frequency * time_step)
                + 1 / time_step * np.cos(damped_angular_frequency * time_step)
            )
            - 1 / time_step
        ),
        1
        / (angular_frequency**2 * time_step)
        * (
            -np.exp(-damping_ratio * angular_frequency * time_step)
            * (
                angular_frequency
                * damping_ratio
                / damped_angular_frequency
                * np.sin(damped_angular_frequency * time_step)
                + np.cos(damped_angular_frequency * time_step)
            )
            + 1
        ),
    ]
    return coefficients


# Function to compute displacement and velocity over time for given acceleration data
def compute_displacement_velocity(acceleration, coefficients):
    # Initialize displacement and velocity arrays with zeros
    displacement = np.zeros_like(acceleration)
    velocity = np.zeros_like(acceleration)
    # Iterate through the acceleration data to compute displacement and velocity
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
    coefficients,
    displacement_prev,
    velocity_prev,
    ground_accel_current,
    ground_accel_next,
):
    # Unpack the coefficients
    A, B, C, D, AA, BB, CC, DD = coefficients
    # Compute the forces based on the ground acceleration
    force_current = -ground_accel_current
    force_next = -ground_accel_next
    # Compute the next displacement and velocity
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
    # Load earthquake data from the current directory
    directory = os.path.curdir
    data = load_txt_files(directory)

    # Assign the first three loaded data arrays to earthquake1_data, earthquake2_data, earthquake3_data
    earthquake1_data, earthquake2_data, earthquake3_data = data[0], data[1], data[2]

    # Define the time step for the data
    time_step = 0.01  # s

    # Reshape the acceleration matrices into vectors
    _, earthquake1_vector = reshape_acceleration_matrix(earthquake1_data, time_step)
    _, earthquake2_vector = reshape_acceleration_matrix(earthquake2_data, time_step)
    time_vector, earthquake3_vector = reshape_acceleration_matrix(
        earthquake3_data, time_step
    )

    # Peak Ground Acceleration (PGA) scaling
    PGA1 = np.max(earthquake1_vector)
    PGA2 = np.max(earthquake2_vector)
    scaling_factor = 1 / max(PGA1, PGA2)
    earthquake1_vector *= scaling_factor
    earthquake2_vector *= scaling_factor

    # Generate 2800 spectrum
    initial_period = 0.15
    transition_period = 0.7
    spectrum_amplitude = 1.75
    initial_spectrum_amplitude = 1.1
    time_increment = 0.01
    max_time = 4
    time_values = np.arange(0, 5 + time_increment, time_increment)
    spectrum_values = np.zeros_like(time_values)
    index = 0

    for period in time_values:
        # Compute the amplification factor based on the period
        if period <= transition_period:
            amplification_factor = 1
        elif period <= 4:
            amplification_factor = (
                0.7 / (4 - transition_period) * (period - transition_period) + 1
            )
        else:
            amplification_factor = 1.7
        # Compute the spectral value based on the period
        if period <= initial_period:
            spectrum_values[index] = initial_spectrum_amplitude + (
                spectrum_amplitude - initial_spectrum_amplitude + 1
            ) * (period / initial_period)
        elif period <= transition_period:
            spectrum_values[index] = spectrum_amplitude + 1
        else:
            spectrum_values[index] = (spectrum_amplitude + 1) * (
                transition_period / period
            )
        spectrum_values[index] *= amplification_factor
        index += 1

    # Plot the 2800 spectrum
    plt.plot(time_values, spectrum_values, label="2800 Spectrum")
    plt.xlabel("Time (s)")
    plt.ylabel("Spectral Amplitude")
    plt.title("2800 Spectrum")
    plt.legend()
    plt.show()

    # Compute and plot the response spectrum for the earthquake data
    spectral_displacement1 = [0]
    spectral_displacement2 = [0]
    damping_ratio = 0.05
    for period in np.arange(0.1, 5.1, 0.1):
        # Compute the natural frequency and damped natural frequency
        natural_frequency = 2 * np.pi / period
        damped_natural_frequency = natural_frequency * np.sqrt(1 - damping_ratio**2)
        # Calculate the coefficients for the current period
        coefficients = calculate_coefficients(
            damping_ratio, natural_frequency, damped_natural_frequency, time_step
        )
        # Compute the displacement and velocity for the given acceleration data
        displacement1, velocity1 = compute_displacement_velocity(
            earthquake1_vector, coefficients
        )
        displacement2, velocity2 = compute_displacement_velocity(
            earthquake2_vector, coefficients
        )

        # Plot the displacement response for the current period
        plt.plot(
            time_vector, displacement1, label=f"Displacement for Period {period:.2f} s"
        )
        plt.plot(
            time_vector, displacement2, label=f"Displacement for Period {period:.2f} s"
        )
        # Append the maximum displacement values to the spectral displacement lists
        spectral_displacement1.append(natural_frequency**2 * np.max(displacement1))
        spectral_displacement2.append(natural_frequency**2 * np.max(displacement2))

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("Displacement Response for Different Periods")
    plt.legend()
    plt.show()

    # Compute the final spectrum values
    spectral_displacement1 = np.array(spectral_displacement1[1:])
    spectral_displacement2 = np.array(spectral_displacement2[1:])
    final_spectrum_values = np.sqrt(
        spectral_displacement1**2 + spectral_displacement2**2
    )
    scaled_spectrum_values = 1.3 * spectrum_values

    # Plot the final spectrum values
    period_values = np.arange(0.1, 5.1, 0.1)
    plt.plot(period_values, spectral_displacement1, label="Spectral Displacement 1")
    plt.plot(period_values, spectral_displacement2, label="Spectral Displacement 2")
    plt.plot(period_values, final_spectrum_values, label="Final Combined Spectrum")
    plt.plot(time_values, scaled_spectrum_values, label="Scaled 2800 Spectrum")
    plt.xlabel("Period (s)")
    plt.ylabel("Spectral Displacement (m)")
    plt.title("Final Spectral Displacements and Combined Spectrum")
    plt.legend()
    plt.show()


# Execute the main function
if __name__ == "__main__":
    main()
