# Numerics glue
import numpy as np
from euler1D import flux, conserved2primitive

# CFL timestep
def computedt(rho, u, p, dx, CFL, gamma):
    a = np.sqrt(gamma * p / rho)
    smax = np.max(np.abs(u) + a)
    
    dt = CFL * dx / smax
    
    return dt
   
def applyBC(U, BCtype):

    """
    Simple 1D boundary conditions to conserved variables U.

    "outflow" - zero-gradient
    "reflective" - wall BC (reverse normal velocity)
    """

    if BCtype == "outflow":
        U[0,:] = U[1,:]
        U[-1,:] = U[-2,:]

    elif BCtype == "reflective":
        # Left boundary
        U[0, 0] = U[1, 0]
        U[0, 1] = -U[1, 1]
        U[0, 2] = U[1, 2]
        # Right boundary
        U[-1, 0] = U[-2, 0]
        U[-1, 1] = -U[-2, 1]
        U[-1, 2] = U[-2, 2]

    else:
        raise ValueError("Unknow BC type.")

    return U

"""
Optional 2nd order
UL, UR = muscl_reconstruction(U, limiter="minmod")
"""
def LF(U, dt, dx, gamma, BCtype):

    # Number of cells
    n = U.shape[0]

    # Apply BC's
    U = applyBC(U, BCtype="outflow")

    # Left and Right fluxes
    UL = U[:-1, :]
    UR = U[1:, :]

    # Physical fluxes at left and right
    FL = flux(UL, gamma)
    FR = flux(UR, gamma)

    # Primitive variables -> wave speeds
    rhoL, uL, pL = conserved2primitive(UL, gamma)
    rhoR, uR, pR = conserved2primitive(UR, gamma)

    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)

    # Rusanov dissipation @ each interface
    alpha = np.maximum(np.abs(uL) + aL, np.abs(uR) + uR)

    # Numerical flux at interfaces
    Fhalf = 0.5 * (FL + FR) - 0.5 * alpha[:, None] * (UR - UL)

    # Updates
    Unew = U.copy()
    Unew[1:-1, :] = U[1:-1, :] - (dt/dx) * (Fhalf[1:, :] - Fhalf[:-1, :])

    return Unew

def MacCormack(U, dt, dx, gamma, BCtype):
    # Predictor
    Up = applyBC(U.copy(), BCtype = "outflow")
    F = flux(Up, gamma)

    Ustar = Up.copy()

    # Forward difference
    Ustar[:-1, :] = Up[:-1, :] - (dt/dx) * (F[1:, :] - F[:-1, :])

    # Corrector step
    Ustar = applyBC(Ustar, BCtype = "outflow")
    Fstar = flux(Ustar, gamma)

    Unew = U.copy()

    # Backward difference
    Unew[1:, :] = 0.5 * (Up[1:, :] + Ustar[1:, :] - (dt / dx) * (Fstar[1:, :] - Fstar[:-1, :]))

    return Unew

