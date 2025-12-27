import numpy as np
import matplotlib.pyplot as plt

from euler1D import primitive2conserved, conserved2primitive
from schemes import computedt, LF, MacCormack
from ioPlot import plotState

def makeGrid(xmin, xmax, N):
    x = np.linspace(xmin, xmax, N)
    dx = x[1] - x[0]
    return x, dx

def initSod(x, x0, left, right):
    rhoL, uL, pL = left
    rhoR, uR, pR = right

    rho = np.where(x < x0, rhoL, rhoR)
    u = np.where(x < x0, uL, uR)
    p = np.where(x < x0, pL, pR)

    return rho, u, p

def runSolver(scheme, x, dx, U, gamma, CFL, tEnd, plotTimes, BCtype="outflow"):

    t = 0.0
    plotTimes = sorted(plotTimes)
    k = 0 # time index of next plot

    print(f"Running scheme: {scheme}")

    while t < tEnd:
        rho, u, p = conserved2primitive(U, gamma)

        dt = computedt(rho, u, p, dx, CFL, gamma)
        dt = min(dt, tEnd - t)

        if scheme.lower() in ["lf", "lax-friedrichs", "rusanov"]:
            U = LF(U, dt, dx, gamma, BCtype=BCtype)

        elif scheme.lower() in ["maccormack", "mac"]:
                U = MacCormack(U, dt, dx, gamma, BCtype=BCtype)

        else:
            raise ValueError(f"Unknown scheme {scheme}")

        t += dt

        if k < len(plotTimes) and t >= plotTimes[k]:
            print(f"Plotting at t = {t:.3f}")
            plotState(x, U, gamma, t=t, which=("rho", "u", "p"))
            plt.show()
            k += 1

    print(f"Finished {scheme} at t = {t:.3f}")
    return u

def main():
    gamma = 1.4
    CFL = 0.1

    xmin, xmax = 0.0, 1.0
    N = 400
    x0 = 0.5
    tEnd = 0.2

    leftState = (1.0, 0.0, 1.0)
    rightState = (0.125, 0.0, 0.1)

    plotTimes = [0.05, 0.1, 0.2]

    x, dx = makeGrid(xmin, xmax, N)
    rho0, u0, p0 = initSod(x, x0, leftState, rightState)
    U0 = primitive2conserved(rho0, u0, p0, gamma)

    schemes = ["LF"]

    for scheme in schemes:
        U = U0.copy()
        runSolver(scheme=scheme, x=x, dx=dx, U=U, gamma=gamma, CFL=CFL, tEnd=tEnd, plotTimes=plotTimes, BCtype="outflow")
        plt.show()

if __name__ == "__main__":
    main()
