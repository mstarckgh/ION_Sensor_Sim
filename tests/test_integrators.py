import numpy as np
from utils.integrators import euler_step, rk4_step


def test_euler_step():
    print("Testing euler_step()")

    # Define deviation function f(y, u) = -y
    def f(y, u):
        return -y
    
    y0 = np.array([1.0])
    u0 = np.array([0.0])
    dt = 0.1

    # expected result y + dt * f(y, u) = 1.0 + 0.1*(-1.0) = 0.9
    expected = np.array([0.9])

    # execute calculation
    result = euler_step(f, y0, u0, dt)

    # check result with tolerance
    tolerance = 1e-10
    assert np.allclose(result, expected, atol=tolerance), f"Expected {expected}; Got {result}"

    print(f"Expected {expected}; Got {result}")    
    print("Test passed")


def test_rk4_step():
    """
    Testfunktion für die Runge-Kutta-4-Implementierung.
    Testet skalare und vektorielle Differentialgleichungen sowie Genauigkeit.
    """
    # Test 1: Skalare Differentialgleichung y' = -2y, y(0) = 1
    def scalar_f(y, u):
        return -2 * y  # u wird hier ignoriert

    y0 = 1.0
    u = 0.0  # Keine Steuerung
    dt = 0.1
    y = rk4_step(scalar_f, y0, u, dt)
    exact_y = y0 * np.exp(-2 * dt)  # Exakte Lösung: y(t) = e^(-2t)
    assert np.abs(y - exact_y) < 1e-4, f"Skalarer Test fehlgeschlagen: {y} != {exact_y}"

    # Test 2: Vektorielles System (harmonischer Oszillator)
    def oscillator_f(y, u):
        # y[0] = x, y[1] = v; dx/dt = v, dv/dt = -x + u
        return np.array([y[1], -y[0] + u])

    y0 = np.array([1.0, 0.0])  # Start: x=1, v=0
    u = 0.0  # Keine externe Kraft
    dt = 0.1
    y = rk4_step(oscillator_f, y0, u, dt)
    # Exakte Lösung: x(t) = cos(t), v(t) = -sin(t)
    exact_y = np.array([np.cos(dt), -np.sin(dt)])
    assert np.all(np.abs(y - exact_y) < 1e-3), f"Vektortest fehlgeschlagen: {y} != {exact_y}"

    # Test 3: Konstante Steuerung u
    def controlled_f(y, u):
        return -2 * y + u  # y' = -2y + u

    y0 = 1.0
    u = 1.0  # Konstante Steuerung
    dt = 0.1
    y = rk4_step(controlled_f, y0, u, dt)
    # Exakte Lösung: y(t) = 0.5 + 0.5*e^(-2t) für u=1, y(0)=1
    exact_y = 0.5 + 0.5 * np.exp(-2 * dt)
    assert np.abs(y - exact_y) < 1e-4, f"Steuerungstest fehlgeschlagen: {y} != {exact_y}"

    # Test 4: Numerische Genauigkeit bei kleiner Schrittweite
    dt = 0.01
    y = rk4_step(scalar_f, y0, u, dt)
    exact_y = y0 * np.exp(-2 * dt)
    error = np.abs(y - exact_y)
    assert error < 1e-6, f"Genauigkeitstest fehlgeschlagen: Fehler {error} zu groß"

    print("Alle Tests erfolgreich bestanden!")