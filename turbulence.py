from scipy.ndimage import distance_transform_edt
import numpy as np
import time
from muscl import muscly
from muscl import musclx

def computeWallDistance(solver):
        """
        Distance (metres) from each fluid cell to the nearest solid cell.
        Uses Euclidean distance transform on the grid.
        """
        fluid = ~solver.solid_mask
        d = distance_transform_edt(fluid) * min(solver.dx, solver.dy)
        d[solver.solid_mask] = 0.0
        # Avoid division by zero in SA source terms
        
        return np.maximum(d, 1e-12)

def saConstants():
        # Standard SA constants

         return {
            "sigma": 2.0/3.0,
            "cb1": 0.1355,
            "cb2": 0.622,
            "kappa": 0.41,
            "cw2": 0.3,
            "cw3": 2.0,
            "cv1": 7.1,
        }

def  saUpdate_nuT(solver):
        """
        Compute nuT from nuTilde via fv1
        """
        c = saConstants()
        cv1 = c["cv1"]

        nu = solver.nu
        nuTilde = np.maximum(solver.nuTilde, 0.0)

        chi = nuTilde / (nu + 1e-12)
        fv1 = (chi**3) / (chi**3 + cv1**3 + 1e-12)

        solver.nuT = nuTilde * fv1

def saStep(solver):
        """
        One explicit time step of Spalart-Allmaras for nuTilde.
        Simplified out but useful for teaching / demos
        """
        t0 = time.perf_counter()
        

        c = saConstants()
        sigma = c["sigma"]
        cb1 = c["cb1"]
        cb2 = c["cb2"]
        kappa = c["kappa"]
        cw2 = c["cw2"]
        cw3 =  c["cw3"]
        cv1 = c["cv1"]

        # Derived constant
        cw1 = cb1 / (kappa**2) + (1.0 + cb2) / sigma

        nu = solver.nu
        d = solver.wallDist
        
        # Current nuTidle
        nuTilde = np.maximum(solver.nuTilde, 0.0)

        # Vorticity magnitude S = |dV/dx - dU/dy|
        dudy = (solver.u[1:-1, 2:] - solver.u[1:-1, :-2]) / (2.0 * solver.dy) 
        dvdx = (solver.v[2:, 1:-1] - solver.v[:-2, 1:-1]) / (2.0 * solver.dx)
        S = solver.sa_S
        S.fill(0.0)
        S[1:-1,1:-1] = np.abs(dvdx - dudy)
        
        t1 = time.perf_counter()
        solver.saTimers["vorticity"] += t1 - t0

        # fv1, fv2
        chi = nuTilde / (nu + 1e-12)

        chi2 = chi * chi
        chi3 = chi2 * chi

        cv13 = cv1**3  #scalar, compute it once

        fv1 = chi3 / (chi3 + cv13 + 1e-12)
        fv2 = 1.0 - chi / (1.0 + chi * fv1 + 1e-12)

        t2 = time.perf_counter()
        solver.saTimers["fv_terms"] += t2 - t1

        # Modified vorticity magnitude S~
        Stilde = S + (nuTilde / (kappa**2 * d**2 + 1e-12)) * fv2
        Stilde = np.maximum(Stilde, 1e-12)

        # Production term
        prod = cb1 * Stilde * nuTilde

        # Destruction term pieces (fw)
        r = nuTilde / (Stilde * (kappa**2) * d**2 + 1e-12)
        r = np.clip(r, 0.0, 10.0)

        # Compute powers by multiplication
        r2 = r * r
        r6 = r2 * r2 * r2

        g = r + cw2 * (r6 - r)

        g2 = g * g
        g6 = g2 * g2 * g2

        fw = g * (
            (1.0 + cw3**6) /
            (g6 + cw3**6 + 1e-12)
        )**(1.0 / 6.0)

        dest = cw1 * fw * (nuTilde / (d + 1e-12))**2

        t3 = time.perf_counter()
        solver.saTimers["production"] += t3 - t2

        # --------------------------------------------------
        # MUSCL / Rusanov advection of nuTilde
        # --------------------------------------------------

        nt = nuTilde

        # X-faces
        ntLx, ntRx = musclx(nt)

        uLx, uRx = musclx(solver.u)

        t4 = time.perf_counter()
        solver.saTimers["muscl_x"] += t4 - t3

        ax = 0.5 * (uLx + uRx)

        Fx_nt = solver.rusanovFlux(ax, ntLx, ntRx)

        # Y-faces
        ntLy, ntRy = muscly(nt)

        vLy, vRy = muscly(solver.v)

        t5 = time.perf_counter()
        solver.saTimers["muscl_y"] += t5 - t4

        ay = 0.5 * (vLy + vRy)

        Gy_nt = solver.rusanovFlux(ay, ntLy, ntRy)

        t6 = time.perf_counter()
        solver.saTimers["rusanov"] += t6 - t5

        # Flux divergence
        adv = solver.sa_adv
        adv.fill(0.0)

        dFx_nt = (Fx_nt[1:, :] - Fx_nt[:-1, :]) / solver.dx
        dGy_nt = (Gy_nt[:, 1:] - Gy_nt[:, :-1]) / solver.dy

        adv[2:-2, 2:-2] = (
            dFx_nt[:, 1:-1]
            +
            dGy_nt[1:-1, :]
        )

        t7 = time.perf_counter()
        solver.saTimers["flux_div"] += t7 - t6

        dnt_dx = solver.sa_dnt_dx
        dnt_dx.fill(0.0)
        dnt_dy = solver.sa_dnt_dy
        dnt_dy.fill(0.0)

        dnt_dx[1:-1,1:-1] = (
            nt[2:,1:-1] - nt[:-2,1:-1]
        )/(2.0*solver.dx)

        dnt_dy[1:-1,1:-1] = (
            nt[1:-1,2:] - nt[1:-1,:-2]
        )/(2.0*solver.dy)

        grad2 = dnt_dx**2 + dnt_dy**2

        t8 = time.perf_counter()
        solver.saTimers["gradients"] += t8 - t7

        # Diffusion term
        ntc = nt[1:-1, 1:-1]
        lap_nt  = ((nt[2:, 1:-1] - 2.0 * ntc + nt[:-2, 1:-1]) / (solver.dx**2) +
              (nt[1:-1, 2:] - 2.0 * ntc + nt[1:-1, :-2]) / (solver.dy**2))
        
        nuEff_nt = (nu + nt) / sigma
        diff = solver.sa_diff
        diff.fill(0.0)
        diff[1:-1, 1:-1] = nuEff_nt[1:-1, 1:-1] * lap_nt

        # Cross-diffusion term
        cross = (cb2 / sigma) * grad2

        t9 = time.perf_counter()
        solver.saTimers["laplacian"] += t9 - t8

        # Explicit update
        rhs = -adv + diff + cross + prod - dest
        nuTilde_new = nuTilde + solver.dt * rhs

        # Enforce BC's - 0 on solid and clamp negative
        nuTilde_new[solver.solid_mask] = 0.0
        nuTilde_new = np.maximum(nuTilde_new, 0.0)

        # Simple inlet BC - freestream turbulence level
        nuTilde_new[0, :] = 3.0 * nu

        solver.nuTilde = nuTilde_new

        t10 = time.perf_counter()
        solver.saTimers["update"] += t10 - t9