

import numpy as np

def euler_step(f, y:np.array, u:np.array, dt:float):
    """
    Simple Euler Integration
    f: Deviation function f(y,u) -> dy/dt
    y: Current state
    u: Input (Steering/Acceleration)
    dt: Steppsize in s
    """

    return y + dt * f(y, u)


def rk4_step(f, y, u, dt):
    """
    Runge-Kutta 4th Order
    f: Deviation funciton f(y,u) -> dy/dt
    y: Current State
    u: Input (Steering/Acceleration)
    dt: Steppsize in s
    """

    k1 = f(y, u)
    k2 = f(y + 0.5*dt*k1, u)
    k3 = f(y + 0.5*dt*k2, u)
    k4 = f(y + dt*k3, u)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
