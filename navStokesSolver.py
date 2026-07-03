import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from geometry import create_geometry_mask
from geometry import get_airfoil_surface 
from turbulence import computeWallDistance
from turbulence import saStep
from turbulence import saUpdate_nuT
from muscl import muscly
from muscl import musclx
import time


class NavierStokesSolver2D:
    """
    2D Navier-Stokes solver using finite volume with immersed boundary method
    for handling complex geometries like cylinders and airfoils
    """
    
    def __init__(self, nx, ny, Lx, Ly, Re, dtMax=0.01, geometry_type='cylinder', use_sparse_solver=True, alpha_deg=10.0):
        # Grid parameters
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.dtMax = dtMax
        self.dt = dtMax
        
        # Physical parameters
        self.Re = Re  # Reynolds number
        self.nu = 1.0 / Re  # Kinematic viscosity
        
        self.rho = 1.0
        self.U_inf = 1.0

        # Reference length for coefficient normalisation
        # Use chord for airfoils, diameter for cylinder
        if geometry_type == 'cylinder':
            self.L_ref = 2 * min(self.Lx, self.Ly) * 0.1   # set this equal to your cylinder diameter in nondimensional units
        elif geometry_type in ['naca0012', 'naca4412']:
            self.L_ref = min(self.Lx, self.Ly) * 0.4
        else:
            self.L_ref = 1.0  # fallback

        # Flow direction angle in radians
        # If freestream is horizontal, leave at 0.0
        self.flow_angle = 0.0
        # Solver options
        self.use_sparse_solver = use_sparse_solver
        
        # Create coordinate arrays
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Initialize flow fields
        self.u = np.zeros((nx, ny))  # x-velocity
        self.v = np.zeros((nx, ny))  # y-velocity
        self.p = np.zeros((nx, ny))  # pressure
        
        # Temporary arrays for time stepping
        self.u_star = np.zeros((nx, ny))
        self.v_star = np.zeros((nx, ny))
        
        # Geometry placement
        self.center_x = self.Lx * 0.3
        self.center_y = self.Ly * 0.5

        # Create geometry mask
        self.geometry_type = geometry_type
        self.alpha_deg = alpha_deg
        self.alpha = np.deg2rad(alpha_deg)
        self.solid_mask = create_geometry_mask(self)
        
        # Initialize boundary conditions
        self.set_initial_conditions()
        
        # Clear the pressure matrix cache
        self._pressure_matrix = None
        self.pressure_lu = None

        # RANS: Spalart-Allmaras toggle
        self.useSa = True # set False to revert to laminar
        self.sa_S = np.zeros((self.nx, self.ny))
        self.sa_adv = np.zeros((self.nx, self.ny))
        self.sa_diff = np.zeros((self.nx, self.ny))
        self.sa_dnt_dx = np.zeros((self.nx, self.ny))
        self.sa_dnt_dy = np.zeros((self.nx, self.ny))

        # SA working variable (nuTilde) and turbulent viscosity nuT
        self.nuTilde = np.zeros((nx,ny), dtype=np.float64)
        self.nuT = np.zeros((nx, ny), dtype=np.float64)

        # Precompute wall distance to the immersed body (solid mask)
        self.wallDist = computeWallDistance(self)

        # Optional: set a small freestream turbulence level (dimensionless-ish)
        # For a "turbulent" inlet, nuTildeInfinity ~ (3-5)*nu is a commonish as a starting point
        self.nuTilde[:,:] = 3.0 * self.nu
        self.nuTilde[self.solid_mask] = 0.0

        self.saTimers = {
            "vorticity": 0.0,
            "fv_terms": 0.0,
            "production": 0.0,
            "muscl_x": 0.0,
            "muscl_y": 0.0,
            "rusanov": 0.0,
            "flux_div": 0.0,
            "gradients": 0.0,
            "laplacian": 0.0,
            "update": 0.0,
        }

        self.musclTimers = {
            "gradients": 0.0,
            "minmod": 0.0,
            "reconstruct": 0.0,
        }

    def set_initial_conditions(self):
        """Set initial flow conditions"""
        # Uniform flow from left
        U_inf = 1.0
        self.u[:, :] = U_inf
        self.v[:, :] = 0.0
        self.p[:, :] = 0.0
        
        # Apply no-slip boundary condition at solid surfaces
        self.u[self.solid_mask] = 0.0
        self.v[self.solid_mask] = 0.0
    
    def apply_boundary_conditions(self, u, v, p):
        """Apply boundary conditions to velocity and pressure fields"""
        # Inlet (left boundary) - uniform flow
        u[0, :] = 1.0
        v[0, :] = 0.0
        
        # Outlet (right boundary) - zero gradient
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]
        p[-1, :] = 0.0  # Reference pressure
        
        # Top and bottom walls - slip conditions
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        
        # No-slip at solid boundaries
        u[self.solid_mask] = 0.0
        v[self.solid_mask] = 0.0
        
        return u, v, p
    
    #def minmod(self, a, b):
     #   """Minmod limiter (vectorised)"""

      #  return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a),np.abs(b))

    def rusanovFlux(self, a, qL, qR):
        """
        Local Lax-Friedrichs (Rusanov) flux for scala q transported by speed a.
        """

        return 0.5 * (a * qL + a * qR) - 0.5 * np.abs(a) * (qR - qL)

    def compute_convection_diffusion(self, u, v):
        """
        Compute convection (MUSCL upwind, flux form) and diffusion (central) terms
        Convection is computed via fluxes:
            u-momentum: d/dx(u*u) + d/dy(v*u)
            v-momentum: d/dx(u*v) + d/dy(v*v)

         Diffusion remains central:
            nu * (d2/dx2 + d2/dy2)
        """
        # Initialize arrays
        conv_u = np.zeros_like(u)
        conv_v = np.zeros_like(v)
        diff_u = np.zeros_like(u)
        diff_v = np.zeros_like(v)
        
        # MUSCL convection
        # X-faces (between i and i+1), j in 1,...,ny-2
        uLx, uRx = musclx(u)
        vLx, vRx = musclx(v)

        ax = 0.5 * (uLx + uRx)

        # Fluxes at x-faces:
        # u-equation 
        Fx_u = self.rusanovFlux(ax, uLx, uRx)
        # v-equation
        Fx_v = self.rusanovFlux(ax, vLx, vRx)

        # Y faces
        uLy, uRy = muscly(u)
        vLy, vRy = muscly(v)

        ay = 0.5 * (vLy + vRy)

        # Fluxes at y-faces
        Gy_u = self.rusanovFlux(ay, uLy, uRy)
        Gy_v = self.rusanovFlux(ay, vLy, vRy)

        # Fx_ (nx-3, ny-2)
        dFx_u = (Fx_u[1:, :] - Fx_u[:-1, :]) / self.dx
        dFx_v = (Fx_v[1:, :] - Fx_v[:-1, :]) / self.dx

        # Gy_ (nx-2, ny-3)
        dGy_u = (Gy_u[:, 1:] - Gy_u[:, :-1]) / self.dy
        dGy_v = (Gy_v[:, 1:] - Gy_v[:, :-1]) / self.dy

        # Align to common interior core
        conv_u[2:-2, 2:-2] = -(dFx_u[:, 1:-1] + dGy_u[1:-1, :])
        conv_v[2:-2, 2:-2] = -(dFx_v[:, 1:-1] + dGy_v[1:-1, :])

        # Central diffusion
        uc = u[1:-1, 1:-1]
        vc = v[1:-1, 1:-1]

        lap_u = ((u[2:, 1:-1] - 2.0 * uc + u[:-2, 1:-1]) / (self.dx ** 2) +
                (u[1:-1, 2:] - 2.0 * uc + u[1:-1, :-2]) / (self.dy ** 2))
        lap_v = ((v[2:, 1:-1] - 2.0 * vc + v[:-2, 1:-1]) / (self.dx ** 2) +
                (v[1:-1, 2:] - 2.0 * vc + v[1:-1, :-2]) / (self.dy ** 2))

        if self.useSa:
            nuEff = self.nu + self.nuT
        else:
            nuEff = self.nu

        
        diff_u[1:-1, 1:-1] = nuEff[1:-1, 1:-1] * lap_u
        diff_v[1:-1, 1:-1] = nuEff[1:-1, 1:-1] * lap_v

        # Solids: zero out and keep boundaries zero
        conv_u[self.solid_mask] = 0.0
        conv_v[self.solid_mask] = 0.0
        diff_u[self.solid_mask] = 0.0
        diff_v[self.solid_mask] = 0.0

        return conv_u, conv_v, diff_u, diff_v

    
    def build_pressure_matrix(self):
        """Build sparse matrix for 2D Poisson equation ∇²p = rhs"""
        n_total = self.nx * self.ny
        
        # Create arrays for sparse matrix construction
        row_indices = []
        col_indices = []
        data = []
        
        def get_index(i, j):
            """Convert 2D indices to 1D index"""
            return i * self.ny + j
        
        # Build matrix row by row
        for i in range(self.nx):
            for j in range(self.ny):
                idx = get_index(i, j)
                
                if self.solid_mask[i, j]:
                    # Solid point: p = 0 (Dirichlet condition)
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(1.0)
                    
                elif i == self.nx - 1:
                    # Right boundary: p = 0 (Dirichlet condition)
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(1.0)
                    
                elif i == 0 or j == 0 or j == self.ny - 1:
                    # Other boundaries: ∂p/∂n = 0 (Neumann condition)
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(1.0)
                    
                    if i == 0 and i + 1 < self.nx:
                        row_indices.append(idx)
                        col_indices.append(get_index(1, j))
                        data.append(-1.0)
                    elif j == 0 and j + 1 < self.ny:
                        row_indices.append(idx)
                        col_indices.append(get_index(i, 1))
                        data.append(-1.0)
                    elif j == self.ny - 1 and j - 1 >= 0:
                        row_indices.append(idx)
                        col_indices.append(get_index(i, self.ny - 2))
                        data.append(-1.0)
                    
                else:
                    # Interior point: ∇²p = rhs
                    # Central coefficient
                    coeff_center = -2.0 / self.dx**2 - 2.0 / self.dy**2
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(coeff_center)
                    
                    # x-direction neighbors
                    coeff_x = 1.0 / self.dx**2
                    row_indices.append(idx)
                    col_indices.append(get_index(i-1, j))
                    data.append(coeff_x)
                    
                    row_indices.append(idx)
                    col_indices.append(get_index(i+1, j))
                    data.append(coeff_x)
                    
                    # y-direction neighbors
                    coeff_y = 1.0 / self.dy**2
                    row_indices.append(idx)
                    col_indices.append(get_index(i, j-1))
                    data.append(coeff_y)
                    
                    row_indices.append(idx)
                    col_indices.append(get_index(i, j+1))
                    data.append(coeff_y)
        
        # Create sparse matrix
        A = csr_matrix((data, (row_indices, col_indices)), shape=(n_total, n_total))
        return A
    
    def solve_pressure_poisson_sparse(self, u_star, v_star):
        """Solve pressure Poisson equation using a cached sparse LU factorization.

        This is typically the biggest speedup: build+factor A once, then reuse each step.
        """
        # Build matrix + LU once
        if self._pressure_matrix is None or self._pressure_lu is None:
            print("Building & factoring pressure matrix...")
            # Use CSC for splu
            self._pressure_matrix = self.build_pressure_matrix().tocsc()
            self._pressure_lu = splu(self._pressure_matrix)

        # Build RHS as a 2D field, then ravel (C-order matches i*ny + j)
        rhs2d = np.zeros((self.nx, self.ny), dtype=np.float64)

        div_u_star = ((u_star[2:, 1:-1] - u_star[:-2, 1:-1]) / (2.0 * self.dx) +
                      (v_star[1:-1, 2:] - v_star[1:-1, :-2]) / (2.0 * self.dy))

        rhs2d[1:-1, 1:-1] = div_u_star / self.dt

        # Solids: enforce RHS=0 (consistent with your Dirichlet p=0 rows in A)
        rhs2d[self.solid_mask] = 0.0

        rhs = rhs2d.ravel()

        # Solve A p = rhs using cached LU
        p_flat = self._pressure_lu.solve(rhs)

        return p_flat.reshape(self.nx, self.ny)
    
    def solve_pressure_poisson(self, u_star, v_star):
        """Solve pressure Poisson equation using finite differences"""
        # Right-hand side of Poisson equation
        rhs = np.zeros((self.nx, self.ny))
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                if not self.solid_mask[i,j]:
                    rhs[i,j] = (1.0/self.dt) * (
                        (u_star[i+1,j] - u_star[i-1,j]) / (2*self.dx) +
                        (v_star[i,j+1] - v_star[i,j-1]) / (2*self.dy)
                    )
        
        # Solve using iterative method (Gauss-Seidel)
        p_new = self.p.copy()
        
        for iteration in range(500):  # Increased iterations
            p_old = p_new.copy()
            
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    if not self.solid_mask[i,j]:
                        p_new[i,j] = (
                            (p_new[i+1,j] + p_old[i-1,j]) / self.dx**2 +
                            (p_new[i,j+1] + p_old[i,j-1]) / self.dy**2 -
                            rhs[i,j]
                        ) / (2 * (1/self.dx**2 + 1/self.dy**2))
            
            # Apply pressure boundary conditions
            p_new[-1, :] = 0.0  # Reference pressure at outlet
            p_new[0, :] = p_new[1, :]  # Zero gradient at inlet
            p_new[:, 0] = p_new[:, 1]  # Zero gradient at walls
            p_new[:, -1] = p_new[:, -2]
            p_new[self.solid_mask] = 0.0  # Ensure solid points have zero pressure
            
            # Check convergence
            residual = np.max(np.abs(p_new - p_old))
            if residual < 1e-6:
                if iteration % 50 == 0:
                    print(f"Pressure solver converged in {iteration} iterations")
                break
        
        return p_new
    
    def computedtCFL(self, cfl=0.3, cflDiff=0.5, dtMin=1e-6, dtMax=1e-2):
        """
        Adaptive timestep based on:
            - advection CFL
            - explicit diffusion stability
        """
        # Consider only fluid cells
        fluid = ~self.solid_mask
        u_abs = np.abs(self.u[fluid])
        v_abs = np.abs(self.v[fluid])

        umax = float(u_abs.max()) if u_abs.size else 0.0
        vmax = float(v_abs.max()) if v_abs.size else 0.0

        eps = 1e-12

        # Advection CFL
        dtAdv = cfl * min(self.dx / (umax + eps), self.dy / (vmax + eps))

        # Diffusion stability

        inv_dx2 = 1.0 / (self.dx * self.dx)
        inv_dy2 = 1.0 / (self.dy * self.dy)

        # Include turbulent viscosity from SA

        if self.useSa:
            nuEffMax = np.max(self.nu + self.nuT)
        else:
            nuEffMax = self.nu

        dtDiff = cflDiff * (
            1.0 / (
                2.0 * nuEffMax * (inv_dx2 + inv_dy2)
                + eps
            )
        )
        dt = min(dtAdv, dtDiff, dtMax)
        dt = max(dt, dtMin)

        return dt

    def time_step(self):
        """Perform one time step using fractional step (projection) method."""
        
        if self.useSa:
            saStep(self)
            saUpdate_nuT(self)
        
        # Adaptive CFL timestep
        self.dt = self.computedtCFL(cfl=0.3, cflDiff=0.5, dtMin=1e-6, dtMax=1e-2)
        
        # Step 1: Solve momentum equations without pressure gradient
        conv_u, conv_v, diff_u, diff_v = self.compute_convection_diffusion(self.u, self.v)

        # Predictor step
        self.u_star = self.u + self.dt * (conv_u + diff_u)
        self.v_star = self.v + self.dt * (conv_v + diff_v)

        # Apply boundary conditions to predictor velocities
        self.u_star, self.v_star, _ = self.apply_boundary_conditions(self.u_star, self.v_star, self.p)

        # Step 2: Solve pressure Poisson equation
        if self.use_sparse_solver:
            self.p = self.solve_pressure_poisson_sparse(self.u_star, self.v_star)
        else:
            self.p = self.solve_pressure_poisson(self.u_star, self.v_star)

        # Step 3: Correct velocities with pressure gradient (vectorized)
        dp_dx = (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2.0 * self.dx)
        dp_dy = (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2.0 * self.dy)

        self.u[1:-1, 1:-1] = self.u_star[1:-1, 1:-1] - self.dt * dp_dx
        self.v[1:-1, 1:-1] = self.v_star[1:-1, 1:-1] - self.dt * dp_dy

        # Enforce solid region (no-slip)
        self.u[self.solid_mask] = 0.0
        self.v[self.solid_mask] = 0.0

        # Apply final boundary conditions
        self.u, self.v, self.p = self.apply_boundary_conditions(self.u, self.v, self.p)

    def interpolate_pressure(self, x, y):
        """
        Bilinear interpolation of pressure field at physical coordinates (x,y).
        """

        xi = (x - self.x[0]) / self.dx
        yi = (y - self.y[0]) / self.dy

        i = int(np.floor(xi))
        j = int(np.floor(yi))

        i = np.clip(i, 0, self.nx - 2)
        j = np.clip(j, 0, self.ny - 2)

        sx = xi - i
        sy = yi - j

        p00 = self.p[i, j]
        p10 = self.p[i + 1, j]
        p01 = self.p[i, j + 1]
        p11 = self.p[i + 1, j + 1]

        return (
            (1 - sx) * (1 - sy) * p00
            + sx * (1 - sy) * p10
            + (1 - sx) * sy * p01
            + sx * sy * p11
        )


    def compute_forces(self):
        """Compute drag and lift forces on the immersed body"""
        xu, yu, xl, yl = get_airfoil_surface(self)

        # Closed loop
        x = np.concatenate([xu, xl[::-1]])
        y = np.concatenate([yu, yl[::-1]])

        Fx = 0.0
        Fy = 0.0

        for k in range(len(x) - 1):

            x1 = x[k]
            y1 = y[k]

            x2 = x[k + 1]
            y2 = y[k + 1]

            dx = x2 - x1
            dy = y2 - y1

            ds = np.sqrt(dx**2 + dy**2)

            if ds < 1e-12:
                continue

            # Tangent
            tx = dx / ds
            ty = dy / ds

            # Outward normal
            nx = -ty
            ny = tx

            # Midpoint
            xm = 0.5 * (x1 + x2)
            ym = 0.5 * (y1 + y2)

            # Pressure at midpoint
            p = self.interpolate_pressure(xm, ym)

            # Pressure force
            dFx = -p * nx * ds
            dFy = -p * ny * ds

            Fx += dFx
            Fy += dFy

        Cd, Cl = self.compute_force_coefficients(Fx, Fy)

        return Fx, Fy, Cd, Cl
    
    def compute_force_coefficients(self, Fx, Fy):
        """
        Convert global force components to drag/lift coefficients.
        Drag/lift are defined relative to freestream direction self.flow_angle
        """

        theta = self.flow_angle

        # Unit vector along freestream (drag direction)
        ex = np.cos(theta)
        ey = np.sin(theta)

        # Unit vector perpendicular to freestream (lift stream)
        lx = -np.sin(theta)
        ly = np.cos(theta)

        # Project force vector onto drag/lift axes 
        F_drag = Fx * ex + Fy * ey
        F_lift = Fx * lx + Fy * ly

        q_inf = 0.5 * self.rho * self.U_inf**2
        denom = q_inf * self.L_ref

        if denom < 1e-14:
            return 0.0, 0.0
        
        Cd = F_drag / denom
        Cl = F_lift / denom

        return Cd, Cl
    
    def compute_vorticity(self):
            """
            Compute vorticity field:
                omega = dv/dx - du/dy
            Returns
            omega : ndarray (nx, ny)
            """
            omega = np.zeros_like(self.u)

            omega[1:-1,1:-1] = (
                (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2.0 * self.dx)
                - (self.u[1:-1,2:] - self.u[1:-1, :-2])/ (2.0 * self.dy)
            )
        
            omega[self.solid_mask] = np.nan

            return omega
        
    def compute_pressure_distribution(self):
            """
                Compute pressure coefficient field.

                Cp = (p - p_inf)/(0.5*rho*U_inf^2)

                Returns
                -------
                Cp : ndarray
            """
            
            q_inf = 0.5 * self.rho * self.U_inf**2

            p_inf = 0.0

            Cp = (self.p - p_inf) / q_inf

            Cp[self.solid_mask] = np.nan

            return Cp

    def compute_surface_cp(self):
            """
                Compute surface pressure coefficient distribution.

                Returns
                -------
                xU, cpU : upper surface
                xL, cpL : lower surface
            """

            q_inf = 0.5 * self.rho * self.U_inf**2

            # --------------------------------------------------
            # Exact airfoil coordinates
            # --------------------------------------------------

            xu, yu, xl, yl = get_airfoil_surface(self)

            # Scale by chord length
            xu = xu * self.L_ref
            yu = yu * self.L_ref

            xl = xl * self.L_ref
            yl = yl * self.L_ref

            # --------------------------------------------------
            # Rotate to physical AoA
            # --------------------------------------------------

            ca = np.cos(self.alpha)
            sa = np.sin(self.alpha)

            Xu = self.center_x + xu * ca - yu * sa
            Yu = self.center_y + xu * sa + yu * ca

            Xl = self.center_x + xl * ca - yl * sa
            Yl = self.center_y + xl * sa + yl * ca

            # --------------------------------------------------
            # Sample pressure field
            # --------------------------------------------------

            cpU = []
            cpL = []

            for xp, yp in zip(Xu, Yu):

                i = int((xp / self.Lx) * (self.nx - 1))
                j = int((yp / self.Ly) * (self.ny - 1))

                i = np.clip(i, 0, self.nx - 1)
                j = np.clip(j, 0, self.ny - 1)

                p_surface = self.p[i, j]

                cpU.append(
                    p_surface / q_inf
                )

            for xp, yp in zip(Xl, Yl):

                i = int((xp / self.Lx) * (self.nx - 1))
                j = int((yp / self.Ly) * (self.ny - 1))

                i = np.clip(i, 0, self.nx - 1)
                j = np.clip(j, 0, self.ny - 1)

                p_surface = self.p[i, j]

                cpL.append(
                    p_surface / q_inf
                )

            cpU = np.array(cpU)
            cpL = np.array(cpL)

            # x/c coordinate
            xU = xu / self.L_ref
            xL = xl / self.L_ref

            return xU, cpU, xL, cpL

    def compute_turbulence_ratio(self):
            """
            Compute turbulence ratio:

                nu_t / nu

            Returns
            -------
            ratio : ndarray
            """

            ratio = self.nuT / (self.nu + 1e-30)

            ratio[self.solid_mask] = np.nan

            return ratio
    
    def run_simulation(self, n_steps, plot_interval=50):
        """Run the CFD simulation"""
        print(f"Running {self.geometry_type} simulation...")
        print(f"Grid: {self.nx}x{self.ny}, Re = {self.Re}, dt = {self.dt}")
        
        # Storage for results
        times = []
        drags = []
        lifts = []
        
        # Setup visualization with better subplot arrangement
        #plt.ion()  # Turn on interactive mode
        #fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        #fig.suptitle(f'2D Navier-Stokes Simulation - {self.geometry_type.title()}', fontsize=16)
        
        # Initialize colorbar references
        self.cbar1 = None
        self.cbar2 = None

        t = 0.0
        for step in range(n_steps):
            # Perform time step
            self.time_step()
            t += self.dt
            
            # Compute forces
            Fx, Fy, Cd_p, Cl_p = self.compute_forces()

            times.append(t)
            drags.append(Cd_p)
            lifts.append(Cl_p)
            
            # Plot results periodically
            #if step % plot_interval == 0:
             #   self.plot_results(axes, step, times, drags, lifts)
              #  plt.draw()
               # plt.pause(0.001)  # Small pause to update display
            
            # Print progress with more details
            if step % 100 == 0:
                u_max = np.max(np.sqrt(self.u**2 + self.v**2))
                p_max = np.max(np.abs(self.p))
                print(f"Step {step:4d}/{n_steps}, Time = {step*self.dt:6.3f}, "
                      f"Max U = {Fx:.4f}, Lift = {Fy:.4f}")
        
        #plt.ioff()  # Turn off interactive mode
        input("Simulation done. Close when finished.")
        return times, drags, lifts
