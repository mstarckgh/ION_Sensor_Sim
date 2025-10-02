import numpy as np
import matplotlib.pyplot as plt
from utils.integrators import rk4_step


class BicycleModel:
    """
    Dynamisches Fahrradmodell für ein Fahrzeug in der Ebene.
    Zustand: [x, y, theta, v] (Position x, y, Orientierung theta, Geschwindigkeit v).
    Steuerung: [a, delta] (Beschleunigung a, Lenkungswinkel delta).
    Nutzt RK4 zur numerischen Integration der Zustandsänderungen.
    
    Parameter:
    -----------
    L : float
        Radstand des Fahrzeugs in Metern (Abstand zwischen Vorder- und Hinterachse).
    x0 : float, optional
        Anfangsposition x (default: 0.0).
    y0 : float, optional
        Anfangsposition y (default: 0.0).
    theta0 : float, optional
        Anfangsorientierung in Radiant (default: 0.0).
    v0 : float, optional
        Anfangsgeschwindigkeit in m/s (default: 0.0).
    """
    
    def __init__(self, L=2.5, x0=0.0, y0=0.0, theta0=0.0, v0=0.0):
        self.L = float(L)  # Radstand
        if self.L <= 0:
            raise ValueError("Radstand L muss positiv sein.")
        self.state = np.array([x0, y0, theta0, v0], dtype=float)  # [x, y, theta, v]

    def dynamics(self, state, u):
        """
        Zustandsableitung des dynamischen Fahrradmodells: dy/dt = f(y, u).
        
        Parameter:
        -----------
        state : np.ndarray
            Zustand [x, y, theta, v].
        u : np.ndarray
            Steuerung [a, delta] (Beschleunigung, Lenkungswinkel).
        
        Rückgabe:
        ---------
        dydt : np.ndarray
            Ableitung [dx/dt, dy/dt, dtheta/dt, dv/dt].
        """
        a, delta = u
        x, y, theta, v = state
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            (v / self.L) * np.tan(delta),
            a
        ])

    def step(self, a, delta, dt):
        """
        Führt einen Zeitschritt mit RK4 aus.
        
        Parameter:
        -----------
        a : float
            Beschleunigung in m/s².
        delta : float
            Lenkungswinkel in Radiant.
        dt : float
            Zeitschritt in Sekunden.
        
        Rückgabe:
        ---------
        state : np.ndarray
            Neuer Zustand [x, y, theta, v] nach dem Zeitschritt.
        """
        if dt <= 0:
            raise ValueError("Zeitschritt dt muss positiv sein.")
        u = np.array([a, delta], dtype=float)
        self.state = rk4_step(self.dynamics, self.state, u, dt)
        return self.state.copy()

    def simulate(self, a, delta, t):
        """
        Simuliert die Trajektorie über eine Zeitspanne.
        
        Parameter:
        -----------
        a : np.ndarray oder float
            Beschleunigung(en) für jeden Zeitschritt in m/s².
        delta : np.ndarray oder float
            Lenkungswinkel für jeden Zeitschritt in Radiant.
        t : np.ndarray
            Array von Zeitpunkten (sortiert, aufsteigend).
        
        Rückgabe:
        ---------
        states : np.ndarray
            Array der Zustände [x, y, theta, v] für jeden Zeitpunkt.
            Form: (len(t), 4).
        """
        if np.isscalar(a):
            a = np.full_like(t, a)
        if np.isscalar(delta):
            delta = np.full_like(t, delta)
        if len(a) != len(t) or len(delta) != len(t):
            raise ValueError("a und delta müssen die gleiche Länge wie t haben.")
        
        states = np.zeros((len(t), 4))
        states[0] = self.state.copy()
        
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            self.step(a[i], delta[i], dt)
            states[i + 1] = self.state.copy()
        
        return states

    def get_state(self):
        """
        Gibt den aktuellen Zustand zurück.
        """
        return self.state.copy()
    


def plot_bicycle_trajectory(t, states, show_theta=True, show_velocity=True):
    """
    Plottet die Trajektorie und Zustände eines dynamischen Fahrradmodells.
    
    Parameter:
    -----------
    t : np.ndarray
        Array der Zeitpunkte (n,).
    states : np.ndarray
        Array der Zustände [x, y, theta, v] für jeden Zeitpunkt, Form (n, 4).
    show_theta : bool, optional
        Wenn True, wird die Orientierung theta(t) geplottet (default: True).
    show_velocity : bool, optional
        Wenn True, wird die Geschwindigkeit v(t) geplottet (default: True).
    
    Rückgabe:
    ---------
    fig : matplotlib.figure.Figure
        Die erstellte Matplotlib-Figur.
    """
    # Validierung der Eingaben
    if states.shape[0] != len(t):
        raise ValueError("Anzahl der Zustände muss mit der Länge von t übereinstimmen.")
    if states.shape[1] != 4:
        raise ValueError("States muss die Form (n, 4) haben für [x, y, theta, v].")
    
    # Extrahiere Zustände
    x, y, theta, v = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    
    # Bestimme die Anzahl der Subplots
    num_subplots = 1 + show_theta + show_velocity
    fig_height = 4 * num_subplots
    
    # Erstelle die Figur
    fig = plt.figure(figsize=(10, fig_height))
    
    # Plot 1: Trajektorie in der xy-Ebene
    ax1 = fig.add_subplot(num_subplots, 1, 1)
    ax1.plot(x, y, 'b-', label='Trajektorie')
    ax1.plot(x[0], y[0], 'go', label='Start')  # Startpunkt
    ax1.plot(x[-1], y[-1], 'ro', label='Ende')  # Endpunkt
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Fahrzeugtrajektorie')
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect('equal')  # Gleiche Skalierung für x und y
    
    # Plot 2: Orientierung theta(t), falls gewünscht
    if show_theta:
        ax2 = fig.add_subplot(num_subplots, 1, 2)
        ax2.plot(t, theta, 'm-', label='Orientierung')
        ax2.set_xlabel('Zeit (s)')
        ax2.set_ylabel('θ (rad)')
        ax2.set_title('Orientierung über Zeit')
        ax2.grid(True)
        ax2.legend()
    
    # Plot 3: Geschwindigkeit v(t), falls gewünscht
    if show_velocity:
        ax3 = fig.add_subplot(num_subplots, 1, num_subplots)
        ax3.plot(t, v, 'c-', label='Geschwindigkeit')
        ax3.set_xlabel('Zeit (s)')
        ax3.set_ylabel('v (m/s)')
        ax3.set_title('Geschwindigkeit über Zeit')
        ax3.grid(True)
        ax3.legend()
    
    # Layout optimieren
    plt.tight_layout()
    
    return fig