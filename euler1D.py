"""
Core Physics
"""
import numpy as np

def primitive2conserved(rho, u, p, gamma):
    
    U = np.zeros((len(rho), 3))
    U[:,0] = rho
    U[:,1] = rho * u
    U[:, 2] = p / (gamma - 1) + 0.5 * rho * u**2

    return U

def conserved2primitive(U, gamma):
    rho = U[:, 0]
    u = U[:, 1] / rho
    E = U[:, 2] / rho
    p = (gamma - 1.0) * rho * (E - 0.5*u*u)

    return rho, u, p
 

# Flux
def flux(U, gamma):

    rho, u, p = conserved2primitive(U, gamma)
    e = p / ((gamma - 1.0) * rho) 
    F = np.zeros_like(U)

    F[:, 0] = rho * u 
    F[:, 1] = rho * u**2 + p
    F[:, 2] = u * (rho * (e + 0.5*u*u) + p)

    return F


    
