import numpy as np
import matplotlib.pyplot as plt

def calculate_arrhenius_params(temperatures, diffusion_coefficients):
    """
    Calculates the activation energy and pre-exponential factor
    for diffusion based on the Arrhenius equation and plots the Arrhenius graph.

    Parameters:
    - temperatures: np.array, temperature values in Kelvin.
    - diffusion_coefficients: np.array, diffusion coefficients in cm²/s.

    Returns:
    - slope: float, slope of the linear fit.
    - intercept: float, intercept of the linear fit.
    - Ea: float, activation energy in eV.
    - D0: float, pre-exponential factor in cm²/s.
    """
    # Constants
    k_B = 8.617333262145e-5  # eV/K

    # Calculate 1/T and ln(D)
    inverse_temperatures = 1 / temperatures
    ln_diffusion_coefficients = np.log(diffusion_coefficients)

    # Perform linear regression to obtain the slope and intercept
    fit = np.polyfit(inverse_temperatures, ln_diffusion_coefficients, 1)
    slope = fit[0]
    intercept = fit[1]

    # Calculate the activation energy (Ea)
    Ea = -slope * k_B

    # Calculate the pre-exponential factor (D0)
    D0 = np.exp(intercept)

    # Plot the Arrhenius graph
    plt.figure(figsize=(10, 8))
    plt.scatter(inverse_temperatures, ln_diffusion_coefficients, color='blue', label='Data', s=100)
    plt.plot(inverse_temperatures, np.polyval(fit, inverse_temperatures), color='red', linestyle='--', linewidth=2, label='Fit')
    plt.xlabel('1/T (K$^{-1}$)', fontsize=14)
    plt.ylabel('ln(D) (cm$^2$/s)', fontsize=14)
    plt.title('Arrhenius Plot for Diffusion Constant', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Output result
    print(f"Slope (m): {slope}")
    print(f"Intercept (b): {intercept}")
    print(f"Activation Energy (Ea): {Ea:.2e} eV")
    print(f"Pre-exponential Factor (D0): {D0:.2e} cm^2/s")

    return slope, intercept, Ea, D0

def main():
    # Example data
    temperatures = np.array([500, 1000, 1250, 1500, 1750])
    diffusion_coefficients = np.array([7.6201e-04, 1.1173e-03, 1.4915e-03, 1.7833e-03, 1.8387e-03])

    # Call the calculation function
    calculate_arrhenius_params(temperatures, diffusion_coefficients)

if __name__ == "__main__":
    main()