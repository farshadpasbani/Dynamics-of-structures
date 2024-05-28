import numpy as np
import matplotlib.pyplot as plt


# Function to compute the Duhamel integral response
def duhamel_integral(
    t_initial,
    time_array,
    angular_freq,
    ground_accel_start,
    ground_accel_end,
    damping_ratio,
):
    response = (
        -1
        / angular_freq
        * (
            (ground_accel_end + ground_accel_start)
            / 2
            * np.exp(-damping_ratio * angular_freq * (time_array - t_initial))
            * np.sin(angular_freq * (time_array - t_initial))
        )
    )
    return response


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


# Main function to demonstrate usage of the above functions
def main():
    # Example data for demonstration
    t_initial = 0
    time_array = np.linspace(0, 5, 500)
    angular_freq = 10
    ground_accel_start = np.random.rand(len(time_array))
    ground_accel_end = np.random.rand(len(time_array))
    damping_ratio = 0.05

    # Compute the Duhamel integral response
    response = duhamel_integral(
        t_initial,
        time_array,
        angular_freq,
        ground_accel_start,
        ground_accel_end,
        damping_ratio,
    )

    # Plot the response
    plt.plot(time_array, response)
    plt.title("Duhamel Integral Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Response")
    plt.show()

    # Additional code for other functions can be similarly added here


if __name__ == "__main__":
    main()
