# Simple Class to display the current state of the Car

import numpy as np



class StateVector:
    def __init__(self, 
                 x=0.0, y=0.0, z=0.0,
                 dx=0.0, dy=0.0, dz=0.0,
                 ddx=0.0, ddy=0.0, ddz=0.0,
                 theta_x=0.0, theta_y=0.0, theta_z=0.0,
                 wx=0.0, wy=0.0, wz=0.0,
                 sa=0.0):
        # Position
        self.x = x
        self.y = y
        self.z = z

        # Velocity
        self.dx = dx
        self.dy = dy
        self.dz = dz

        # Acceleration
        self.ddx = ddx
        self.ddy = ddy
        self.ddz = ddz

        # Orientation
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_z = theta_z  # Yaw (Heading)

        # Angular velocity
        self.wx = wx
        self.wy = wy
        self.wz = wz

        # Steering angle
        self.sa = sa

    def to_array(self):
        return np.array([
            self.x, self.y, self.z,
            self.dx, self.dy, self.dz,
            self.ddx, self.ddy, self.ddz,
            self.theta_x, self.theta_y, self.theta_z,
            self.wx, self.wy, self.wz,
            self.sa
        ])
    

