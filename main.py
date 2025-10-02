from models.bicycle_model import BicycleModel, plot_bicycle_trajectory

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Fahrzeug initialisieren
    bike = BicycleModel(L=2.5, x0=0.0, y0=0.0, theta0=0.0, v0=10.0)
    
    # Simulation über 10 Sekunden
    t = np.linspace(0, 20, 101)
    a = np.zeros_like(t)  # Konstante Beschleunigung 1 m/s²
    delta = 0.1 * np.ones_like(t)  # Konstanter Lenkungswinkel 0.1 rad
    states = bike.simulate(a, delta, t)
    
    # Plot erstellen
    fig = plot_bicycle_trajectory(t, states, show_theta=True, show_velocity=True)
    plt.show()


if __name__ == "__main__":
    main()
