# Simple test to check if the bicycle_model is working correctly

import numpy as np
from models.bicycle_model import BicycleModel
from utils.state_vector import StateVector


def test_straight_line():
    model = BicycleModel
    
    x = StateVector
    x.dx = 10.0

    trajectory= []

    for _ in range(2000):
        x = model.step(x, )