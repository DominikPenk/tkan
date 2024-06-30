import torch
import numpy as np


from .base import TestFunction

FEYMANN_EQUATIONS:list[TestFunction] = [
    TestFunction(lambda theta, sigma: torch.exp(-0.5 * theta.square() / sigma.square()) / torch.sqrt(2.0*np.pi*sigma.square())),
    TestFunction(lambda theta, theta_1, sigma: torch.exp(-0.5 * (theta-theta_1).square() / sigma.square()) / torch.sqrt(2.0*np.pi*sigma.square())),
    TestFunction(lambda a, b, c, d, e, f: a / ((b-1).square() + (c-d).square()+(e-f).square())),
    TestFunction(lambda a, theta: 1 + a*torch.sin(theta)),
    TestFunction(lambda a, b: a*(1.0/b - 1)),
    TestFunction(lambda a, b: (1.0-a)/torch.sqrt(1-b.square())),
    TestFunction(lambda a, b: (a+b)/(1+a*b)),
    TestFunction(lambda a, b: (1.0 + a*b) / (1.0 + a)),
    TestFunction(lambda n, theta_2: torch.arcsin(n * torch.sin(theta_2))),
    TestFunction(lambda a, b: 1.0 / (1.0 + a*b)), 
    # TestFunction(lambda a, theta_1, theta_2: torch.sqrt(1.0 + a**2 - 2.0 * torch.acos(theta_1 - theta_2))), 
    TestFunction(lambda n, theta: torch.sin(0.5*n*theta)**2/torch.sin(0.5*theta))
]

def get_feymann_equations() -> list[TestFunction]:
    return FEYMANN_EQUATIONS