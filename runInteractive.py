from navStokesSolver import NavierStokesSolver2D
import postProcessing as pp
import matplotlib.pyplot as plt


# ======================================================
# Interactive Simulation Settings
# ======================================================

nx = 400
ny = 300

Lx = 16.0
Ly = 10.5

Re = 100

dtMax = 0.01

geometry = "cylinder"

alpha = 0.0

nSteps = 2000

plotEvery = 20

use_sparse = True

use_sa = False


# ======================================================
# Create Solver
# ======================================================

solver = NavierStokesSolver2D(
    nx=nx,
    ny=ny,
    Lx=Lx,
    Ly=Ly,
    Re=Re,
    dtMax=dtMax,
    geometry_type=geometry,
    use_sparse_solver=use_sparse,
    alpha_deg=alpha
)


# ======================================================
# Interactive Plot
# ======================================================

plt.ion()

fig, ax = plt.subplots(figsize=(10,5))

timeHistory = []
dragHistory = []
liftHistory = []


# ======================================================
# Time Loop
# ======================================================

for step in range(nSteps):

    solver.time_step()

    Fx, Fy, Cd, Cl = solver.compute_forces()

    timeHistory.append(step*solver.dt)
    dragHistory.append(Cd)
    liftHistory.append(Cl)

    if step % plotEvery == 0:

        ax.clear()

        pp.plot_results(
            solver,
            ax=ax
        )

        ax.set_title(
            f"Step = {step}    Re = {solver.Re}"
        )

        plt.pause(0.001)


plt.ioff()

plt.show()