from navStokesSolver import NavierStokesSolver2D
import postProcessing as pp
import cProfile
import pstats
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
# ======================================================
# Simulation Parameters
# ======================================================

nx = 800
ny = 800

Lx = 16.0
Ly = 10.5

Re = 3.9e6

dtMax = 0.001

geometry = "naca0012"

alpha = 0.0

nSteps = 1000

use_sparse = True

useSa = True


# ======================================================
# Create Solver
# ======================================================

allResults = []
outputFolder = "Results"

os.makedirs(outputFolder, exist_ok=True)

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

#profiler = cProfile.Profile()
#profiler.enable()

# ======================================================
# Run Simulation
# ======================================================

times, dragHistory, liftHistory = solver.run_simulation(
    n_steps=nSteps
)


# Average over last 100 iterations
nAverage = min(100, len(dragHistory))

CdAverage = np.mean(dragHistory[-nAverage:])
ClAverage = np.mean(liftHistory[-nAverage:])

CdStd = np.std(dragHistory[-nAverage:])
ClStd = np.std(liftHistory[-nAverage:])

summaryFile = os.path.join(outputFolder, "summary.csv")

fileExists = os.path.exists(summaryFile)

with open(summaryFile, "a", newline="") as f:

    writer = csv.writer(f)

    if not fileExists:

        writer.writerow([
            "Re",
            "AoA",
            "Cl",
            "Cd",
            "ClStd",
            "CdStd"
        ])

    writer.writerow([
        Re,
        alpha,
        ClAverage,
        CdAverage,
        ClStd,
        CdStd
    ])

print("\n===== Converged Aerodynamic Coefficients =====")
print(f"Average Cl : {ClAverage:.6f} ± {ClStd:.6f}")
print(f"Average Cd : {CdAverage:.6f} ± {CdStd:.6f}")


#profiler.disable()

#stats = pstats.Stats(profiler)
#stats.sort_stats("cumtime")
#stats.print_stats(20)

pp.plot_pressure_distribution(solver)

pp.plot_surface_cp(solver)

pp.plot_vorticity(solver)

pp.plot_turbulence_ratio(solver)

plt.figure(figsize=(10,5))

# Lift
plt.figure(figsize=(10,4))
plt.plot(times, liftHistory)
plt.xlabel("Time")
plt.ylabel("Cl")
plt.title("Lift Coefficient History")
plt.grid(True)

# Drag
plt.figure(figsize=(10,4))
plt.plot(times, dragHistory)
plt.xlabel("Time")
plt.ylabel("Cd")
plt.title("Drag Coefficient History")
plt.grid(True)

plt.show()

omega = solver.compute_vorticity()

nuRatio = solver.compute_turbulence_ratio()

xUpper, cpUpper, xLower, cpLower = solver.compute_surface_cp()

caseFolder = os.path.join(
    outputFolder,
    f"Re_{int(Re)}_AoA_{alpha:.1f}"
)

os.makedirs(caseFolder, exist_ok=True)

cpFile = os.path.join(caseFolder, "cp.csv")

with open(cpFile, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Upper x/c",
        "Upper Cp",
        "Lower x/c",
        "Lower Cp"
    ])

    n = max(len(xUpper), len(xLower))

    for i in range(n):

        row = []

        if i < len(xUpper):
            row += [xUpper[i], cpUpper[i]]
        else:
            row += ["",""]

        if i < len(xLower):
            row += [xLower[i], cpLower[i]]
        else:
            row += ["",""]

        writer.writerow(row)

np.savetxt(
    os.path.join(caseFolder, "vorticity.csv"),
    omega,
    delimiter=","
)

np.savetxt(
    os.path.join(caseFolder, "turbulence_ratio.csv"),
    nuRatio,
    delimiter=","
)

np.savetxt(
    os.path.join(caseFolder, "LiftCoeff.csv"),
    liftHistory,
    delimiter=","
)

np.savetxt(
    os.path.join(caseFolder, "DragCoeff.csv"),
    dragHistory,
    delimiter=","
)

results = {
    "Re": Re,
    "AoA": alpha,

    "Cl": ClAverage,
    "Cd": CdAverage,

    "ClStd": ClStd,
    "CdStd": CdStd,

    "Vorticity": omega,

    "TurbulenceRatio": nuRatio,

    "CpUpperX": xUpper,
    "CpUpper": cpUpper,

    "CpLowerX": xLower,
    "CpLower": cpLower,
}

allResults.append(results)

