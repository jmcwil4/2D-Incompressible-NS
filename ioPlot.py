"""
Input/Output and plotting utilities for 1D Euler shock tube
"""
import numpy as np
import matplotlib.pyplot as plt

from euler1D import conserved2primitive

def plotState(x, U, gamma, t=None, which=("rho", "u", "u")):
    """
    Plot a single state U (density, velocity, pressure etc.) vs x
    x: 1D array, shape (N,), spatial grid
    U: (N, 3) array - conserved variables [rho, rho*u, rho*E]
    gamma: float - ratio of specific heats
    t: float or None - time
    which: tuple of str - any subset of ("rho", "u", "p", "E"), choose what to plot
    """

    rho, u, p = conserved2primitive(U, gamma)
    E = U[:,2] / rho

    fig, ax = plt.subplots()

    if "rho" in which:
        ax.plot(x, rho, label=r"$\rho$")
    if "u" in which:
        ax.plot(x, u, label=r"$u$")
    if "p" in which:
        ax.plot(x, p, label=r"$p$")
    if "E" in which:
        ax.plot(x, E, label=r"$E$")

    ax.set_xlabel("x")
    ax.set_ylabel("Primitive variables")
    if t is not None:
        ax.set_title(f"State at t = {t:.4f}")
    else:
        ax.set_title("State")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    return fig, ax

def plot_snapshots(x, snapshots, gamma, which=("rho", "u", "p"), times=None):
    
    # Choose which snapshots to plot
    if times is None:
        selected = snapshots
    else:
        allTimes = np.array([t for (t, _) in snapshots])
        selected = []
        for tTarget in times:
            idx = np.argmin(np.abs(allTimes - tTarget))
            selected.append(snapshots[idx])

    fig, ax = plt.subplots()

    for t, U in selected:
        rho, u, p = conserved2primitive(U, gamma)
        E = U[:, 2] / rho

        labelSuffix = f", t={t:.4f}"

        if "rho" in which:
            ax.plot(x, rho, label=r"$\rho$" + labelSuffix)
        if "u" in which:
            ax.plot(x, u, label=r"$u$" + labelSuffix)
        if "p" in which:
            ax.plot(x, p, label=r"$p$" + labelSuffix)
        if "E" in which:
            ax.plot(x, E, label=r"$E$" + labelSuffix)

    ax.set_xlabel("x")
    ax.set_ylabel("Primitive variables")
    ax.set_title("Shock tube evolution")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    return fig, ax

def plotComparison(x, states, labels, gamma, which=("rho", "u", "p"), t=None):

    fig, ax = plt.subplots()

    for U, lab in zip(states, labels):
        rho, u, p = conserved2primitive(U, gamma)
        E = U[:, 2] / rho

        if "rho" in which:
            ax.plot(x, rho, label=lab + r" : $\rho$")
        if "u" in which:
            ax.plot(x, u, label=lab + r" : $u$")
        if "p" in which:
            ax.plot(x, p, label=lab + r" : $p$")
        if "E" in which:
            ax.plot(x, E, label=lab + r" : $E$")

    ax.set_xlabel("x")
    ax.set_ylabel("Primitive variables")
    if t is not None:
        ax.set_title(f"Scheme comparison at t = {t:.4f}")
    else:
        ax.set_title("Scheme comparison")

    ax.grid(True)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    return fig, ax
