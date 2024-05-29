# Earthquake Data Analysis

This Python script analyzes earthquake data from text files, generates a 2800 spectrum, and computes the displacement response spectrum for the provided earthquake data. The script processes the acceleration data, scales it, calculates spectral values, and visualizes the results using matplotlib.

## Requirements

- Python 3.x
- numpy
- matplotlib
- glob
- os

You can install the required Python packages using pip:

```bash
pip install numpy matplotlib
```
## Usage
### Prepare the Data:

- Ensure that you have at least three text files containing acceleration data in the same directory as the script. The text files should contain 2D arrays of acceleration values.

### Run the Script:

- Execute the script to load the data, process it, and generate the plots.
```bash
python earthquake_analysis.py
```
## Script Details
### Functions
- **load_txt_files(directory)**: Loads all .txt files from the specified directory and returns their content as a list of arrays.
- **reshape_acceleration_matrix(acceleration_matrix, time_step)**: Reshapes a 2D acceleration matrix into a 1D vector and creates a corresponding time vector.
- **calculate_coefficients(damping_ratio, angular_frequency, damped_angular_frequency, time_step)**: Calculates the coefficients needed for computing displacement and velocity.
- **compute_displacement_velocity(acceleration, coefficients)**: Computes displacement and velocity over time for given acceleration data.
- **compute_next_displacement_velocity(coefficients, displacement_prev, velocity_prev, ground_accel_current, ground_accel_next)**: Computes the next displacement and velocity based on current states and ground accelerations.
### Main Workflow

1. Load Earthquake Data:
- Loads the first three text files in the directory.
- Reshapes the acceleration matrices into vectors.
- Scales the acceleration vectors based on Peak Ground Acceleration (PGA).

2. Generate 2800 Spectrum:
- Computes the spectral values for the 2800 spectrum.
- Plots the 2800 spectrum.

3. Compute and Plot Displacement Response Spectrum:
- Calculates the displacement and velocity for each period.
- Plots the displacement response for different periods.

4. Compute and Plot Final Spectrum:
- Computes the final combined spectrum values.
- Plots the final spectral displacements and combined spectrum.

### Example Plots

1. 2800 Spectrum Plot:
- Shows the spectral amplitude over time.

2. Displacement Response Plot:
- Displays the displacement response for different periods.

3. Final Spectral Displacements and Combined Spectrum Plot:
- Visualizes the final spectral displacements and the combined spectrum.

## Contact
For any questions or issues, please open an issue on the repository or contact the author.