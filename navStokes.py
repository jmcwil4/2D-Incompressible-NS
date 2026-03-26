import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve, splu
import time
from scipy.ndimage import distance_transform_edt

class NavierStokesSolver2D:
    """
    2D Navier-Stokes solver using finite differences with immersed boundary method
    for handling complex geometries like cylinders and airfoils
    """
    
    def __init__(self, nx, ny, Lx, Ly, Re, dtMax=0.01, geometry_type='cylinder', use_sparse_solver=True, alpha_deg=0.0):
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
        
        # Create geometry mask
        self.geometry_type = geometry_type
        self.alpha_deg = alpha_deg
        self.alpha = np.deg2rad(alpha_deg)
        self.solid_mask = self.create_geometry_mask()
        
        # Initialize boundary conditions
        self.set_initial_conditions()
        
        # Clear the pressure matrix cache
        self._pressure_matrix = None
        self.pressure_lu = None

        # RANS: Spalart-Allmaras toggle
        self.useSa = True # set False to revert to laminar

        # SA working variable (nuTilde) and turbulent viscosity nuT
        self.nuTilde = np.zeros((nx,ny), dtype=np.float64)
        self.nuT = np.zeros((nx, ny), dtype=np.float64)

        # Precompute wall distance to the immersed body (solid mask)
        self.wallDist = self.computeWallDistance()

        # Optional: set a small freestream turbulence level (dimensionless-ish)
        # For a "turbulent" inlet, nuTildeInfinity ~ (3-5)*nu is a commonish as a starting point
        self.nuTilde[:,:] = 3.0 * self.nu
        self.nuTilde[self.solid_mask] = 0.0
        
    def create_geometry_mask(self):
        """Create mask for solid boundaries (True = solid, False = fluid).

        Fast, vectorized geometry creation for:
        - 'cylinder'
        - 'naca0012'  (also aliased by 'airfoil')
        - 'naca4412'

        Optional angle-of-attack support:
        - If self.alpha_deg exists: degrees
        - Else if self.alpha exists: radians
        - Else: 0
        """

        mask = np.zeros((self.nx, self.ny), dtype=bool)

        # Geometry placement (keep your original intent)
        center_x, center_y = self.Lx * 0.3, self.Ly * 0.5

        # Optional AoA support (body rotated by +alpha relative to flow)
        if hasattr(self, "alpha_deg"):
            alpha = np.deg2rad(float(self.alpha_deg))
        elif hasattr(self, "alpha"):
            alpha = float(self.alpha)
        else:
            alpha = 0.0

        # Shift to body-centered coordinates
        x0 = self.X - center_x
        y0 = self.Y - center_y

        # Rotate grid into body frame by -alpha (so body appears at +alpha in lab frame)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        Xr = x0 * ca - y0 * sa
        Yr = x0 * sa + y0 * ca

        # -------------------------
        # Cylinder
        # -------------------------
        if self.geometry_type == "cylinder":
            radius = min(self.Lx, self.Ly) * 0.1
            mask = (Xr**2 + Yr**2) <= radius**2
            return mask

        # -------------------------
        # NACA airfoils
        # -------------------------
        geom = self.geometry_type.lower()
        if geom == "airfoil":   # your original alias
            geom = "naca0012"

        if geom not in ("naca0012", "naca4412"):
            # Unknown geometry -> no obstacle
            return mask

        # Use your existing chord definition (but vectorized)
        chord = min(self.Lx, self.Ly) * 0.4

        # Non-dimensional coordinates in body frame
        x = Xr / chord
        y = Yr / chord

        # Only consider points inside chord range (0..1)
        inside_chord = (x >= 0.0) & (x <= 1.0)

        # Thickness distribution (NACA 4-digit), half-thickness yt(x)
        # Using the common "closed trailing edge" coefficient -0.1015
        t = 0.12
        x_clip = np.clip(x, 1e-12, 1.0)  # avoid sqrt(0) issues
        yt = 5.0 * t * (0.2969 * np.sqrt(x_clip) - 0.1260 * x_clip
                        - 0.3516 * x_clip**2 + 0.2843 * x_clip**3 - 0.1015 * x_clip**4)

        if geom == "naca0012":
            # Symmetric: camber line is zero; thickness is vertical envelope
            mask = inside_chord & (np.abs(y) <= yt)
            return mask

        # -------------------------
        # NACA 4412 (cambered)
        # -------------------------
        m = 0.04
        p = 0.4

        # Camber line yc(x) and slope dyc/dx (vectorized)
        yc = np.zeros_like(x, dtype=float)
        dyc_dx = np.zeros_like(x, dtype=float)

        # Only defined meaningfully on chord
        x_in = np.clip(x, 0.0, 1.0)

        left = x_in < p
        right = ~left

        # Camber line
        yc[left] = (m / p**2) * (2.0 * p * x_in[left] - x_in[left]**2)
        yc[right] = (m / (1.0 - p)**2) * ((1.0 - 2.0 * p) + 2.0 * p * x_in[right] - x_in[right]**2)

        # Slope
        dyc_dx[left] = (2.0 * m / p**2) * (p - x_in[left])
        dyc_dx[right] = (2.0 * m / (1.0 - p)**2) * (p - x_in[right])

        theta = np.arctan(dyc_dx)

        # Build *proper* upper/lower surface coordinates (xu,yu) and (xl,yl)
        # xu = x - yt*sin(theta), yu = yc + yt*cos(theta)
        # xl = x + yt*sin(theta), yl = yc - yt*cos(theta)
        xu = x_in - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x_in + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # To test if a grid point is inside the airfoil, we need y between y_lower(x) and y_upper(x).
        # Because xu/xl are slightly shifted in x, we build interpolants yU(x) and yL(x).

        # Create 1D airfoil surface curves for interpolation (fast and stable)
        n_surf = 600
        xs = np.linspace(0.0, 1.0, n_surf)
        xs_clip = np.clip(xs, 1e-12, 1.0)

        yt_s = 5.0 * t * (0.2969 * np.sqrt(xs_clip) - 0.1260 * xs_clip
                        - 0.3516 * xs_clip**2 + 0.2843 * xs_clip**3 - 0.1015 * xs_clip**4)

        yc_s = np.empty_like(xs)
        dyc_s = np.empty_like(xs)

        left_s = xs < p
        right_s = ~left_s

        yc_s[left_s] = (m / p**2) * (2.0 * p * xs[left_s] - xs[left_s]**2)
        yc_s[right_s] = (m / (1.0 - p)**2) * ((1.0 - 2.0 * p) + 2.0 * p * xs[right_s] - xs[right_s]**2)

        dyc_s[left_s] = (2.0 * m / p**2) * (p - xs[left_s])
        dyc_s[right_s] = (2.0 * m / (1.0 - p)**2) * (p - xs[right_s])

        theta_s = np.arctan(dyc_s)

        xu_s = xs - yt_s * np.sin(theta_s)
        yu_s = yc_s + yt_s * np.cos(theta_s)
        xl_s = xs + yt_s * np.sin(theta_s)
        yl_s = yc_s - yt_s * np.cos(theta_s)

        # Sort by x for monotone interpolation
        iu = np.argsort(xu_s)
        il = np.argsort(xl_s)
        xu_s, yu_s = xu_s[iu], yu_s[iu]
        xl_s, yl_s = xl_s[il], yl_s[il]

        # Interpolate upper/lower y at the grid point x-locations
        x_flat = x.ravel()
        y_flat = y.ravel()
        inside_flat = inside_chord.ravel()

        yU = np.full_like(x_flat, np.nan, dtype=float)
        yL = np.full_like(x_flat, np.nan, dtype=float)

        # Only interpolate where inside chord to avoid meaningless extrapolation
        idx = np.where(inside_flat)[0]
        xq = x_flat[idx]

        # np.interp extrapolates by endpoints; we want NaN outside valid surface x-range
        # so we clip query to [min(x_surf), max(x_surf)] and then apply chord mask anyway.
        xq_clip = np.clip(xq, xu_s[0], xu_s[-1])
        yU[idx] = np.interp(xq_clip, xu_s, yu_s)

        xq_clip = np.clip(xq, xl_s[0], xl_s[-1])
        yL[idx] = np.interp(xq_clip, xl_s, yl_s)

        inside_airfoil = inside_flat & (y_flat >= yL) & (y_flat <= yU)
        mask = inside_airfoil.reshape(self.nx, self.ny)
        return mask


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
    
    def computeWallDistance(self):
        """
        Distance (metres) from each fluid cell to the nearest solid cell.
        Uses Euclidean distance transform on the grid.
        """
        fluid = ~self.solid_mask
        d = distance_transform_edt(fluid) * min(self.dx, self.dy)
        d[self.solid_mask] = 0.0
        # Avoid division by zero in SA source terms
        
        return np.maximum(d, 1e-12)
    

    def saConstants(self):
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

    def  saUpdate_nuT(self):
        """
        Compute nuT from nuTilde via fv1
        """
        c = self.saConstants()
        cv1 = c["cv1"]

        nu = self.nu
        nuTilde = np.maximum(self.nuTilde, 0.0)

        chi = nuTilde / (nu + 1e-12)
        fv1 = (chi**3) / (chi**3 + cv1**3 + 1e-12)

        self.nuT = nuTilde * fv1

    def saStep(self):
        """
        One explicit time step of Spalart-Allmaras for nuTilde.
        Simplified out but useful for teaching / demos
        """

        c = self.saConstants()
        sigma = c["sigma"]
        cb1 = c["cb1"]
        cb2 = c["cb2"]
        kappa = c["kappa"]
        cw2 = c["cw2"]
        cw3 =  c["cw3"]
        cv1 = c["cv1"]

        # Derived constant
        cw1 = cb1 / (kappa**2) + (1.0 + cb2) / sigma

        nu = self.nu
        d = self.wallDist
        
        # Current nuTidle
        nuTilde = np.maximum(self.nuTilde, 0.0)

        # Vorticity magnitude S = |dV/dx - dU/dy|
        dudy = (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2.0 * self.dy) 
        dvdx = (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2.0 * self.dx)
        S = np.zeros_like(self.u)
        S[1:-1,1:-1] = np.abs(dvdx - dudy)

        # fv1, fv2
        chi = nuTilde / (nu + 1e-12)
        fv1 = (chi**3) / (chi**3 + cv1**3 + 1e-12)
        fv2 = 1.0 - chi / (1.0 + chi * fv1 + 1e-12)

        # Modified vorticity magnitude S~
        Stilde = S + (nuTilde / (kappa**2 * d**2 + 1e-12)) * fv2
        Stilde = np.maximum(Stilde, 1e-12)

        # Production term
        prod = cb1 * Stilde * nuTilde

        # Destruction term pieces (fw)
        r = nuTilde / (Stilde * (kappa**2) * d**2 + 1e-12)
        r = np.clip(r, 0.0, 10.0)

        g = r + cw2 * (r**6 - r)
        fw = g * ((1.0 + cw3**6) / (g**6 + cw3**6 + 1e-12))**(1.0/6.0)

        dest = cw1 * fw * (nuTilde / (d + 1e-12))**2

        # Advection of nuTidle (Simple Central)
        nt = nuTilde
        dnt_dx = np.zeros_like(nt)
        dnt_dy = np.zeros_like(nt)
        dnt_dx[1:-1, 1:-1] = (nt[2:, 1:-1] - nt[:-2, 1:-1]) / (2.0 * self.dx)
        dnt_dy[1:-1, 1:-1] = (nt[1:-1, 2:] - nt[1:-1, :-2]) / (2.0 * self.dy)

        adv = self.u * dnt_dx + self.v * dnt_dy

        # Diffusion term
        ntc = nt[1:-1, 1:-1]
        lap_nt  = ((nt[2:, 1:-1] - 2.0 * ntc + nt[:-2, 1:-1]) / (self.dx**2) +
              (nt[1:-1, 2:] - 2.0 * ntc + nt[1:-1, :-2]) / (self.dy**2))
        
        nuEff_nt = (nu + nt) / sigma
        diff = np.zeros_like(nt)
        diff[1:-1, 1:-1] = nuEff_nt[1:-1, 1:-1] * lap_nt

        # Cross-diffusion term
        grad2 = dnt_dx**2 + dnt_dy**2
        cross = (cb2 / sigma) * grad2

        # Explicit update
        rhs = -adv + diff + cross + prod - dest
        nuTilde_new = nuTilde + self.dt * rhs

        # Enforce BC's - 0 on solid and clamp negative
        nuTilde_new[self.solid_mask] = 0.0
        nuTilde_new = np.maximum(nuTilde_new, 0.0)

        # Simple inlet BC - freestream turbulence level
        nuTilde_new[0, :] = 3.0 * nu

        self.nuTilde = nuTilde_new

    def minmod(self, a, b):
        """Minmod limiter (vectorised)"""

        return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a),np.abs(b))

    def musclx(self, q):
        """
        MUSCL left/right states at x faces using minmod limiter
        Returns qL, qR located at faces between i and i+1 for:
        i = 1,...,nx-3, j = 1,...,ny-2 (interior in y)
        Shapes:(nx-3, ny-2)
        """

        qc = q[1:-1, 1:-1] # Cell centres
        dqL = qc - q[:-2, 1:-1]
        dqR = q[2:, 1:-1] - qc
        slope = self.minmod(dqL, dqR)

        # Faces i+1/2 for i=1,nx-3 => between qc[:-1] and qc[1:]
        qL = qc[:-1, :] + 0.5 * slope[:-1, :]
        qR = qc[1:, :] - 0.5 * slope[1:, :]

        return qL, qR
    
    def muscly(self, q):
        """
        MUSCL left/right states at y-faces using minmod limiter
        Returns qL, qR located at faces between j and j+1 for:
            - i = 1,...,nx-2, j=1,...,ny-3 (interior)
        Shapes: (nx-2, ny-3)
        """
        qc = q[1:-1, 1:-1]
        dqB = qc - q[1:-1, :-2]
        dqT = q[1:-1, 2:] - qc
        slope = self.minmod(dqB, dqT)

        # Faces j+1/2 for j=1,...,ny-3 => between qc[:,:-1] and qc[:,1:]
        qL = qc[:,:-1] + 0.5 * slope[:, :-1]
        qR = qc[:, 1:] - 0.5 * slope[:, 1:]

        return qL, qR

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
        uLx, uRx = self.musclx(u)
        vLx, vRx = self.musclx(v)

        ax = 0.5 * (uLx + uRx)

        # Fluxes at x-faces:
        # u-equation 
        Fx_u = self.rusanovFlux(ax, uLx, uRx)
        # v-equation
        Fx_v = self.rusanovFlux(ax, vLx, vRx)

        # Y faces
        uLy, uRy = self.muscly(u)
        vLy, vRy = self.muscly(v)

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
        dtDiff = cflDiff * (1.0 / (2.0 * self.nu * (inv_dx2 + inv_dy2) + eps))

        dt = min(dtAdv, dtDiff, dtMax)
        dt = max(dt, dtMin)

        return dt

    def time_step(self):
        """Perform one time step using fractional step (projection) method."""
        
        if self.useSa:
            self.saStep()
            self.saUpdate_nuT()
        
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
        
    def compute_forces(self):
        """Compute drag and lift forces on the immersed body"""
        Fx = 0.0
        Fy = 0.0

        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                if not self.solid_mask[i, j]:
                    continue

                # Check whether this solid cell is on the boundary 
                right_fluid = not self.solid_mask[i + 1, j]
                left_fluid = not self.solid_mask[i - 1, j]
                top_fluid = not self.solid_mask[i, j + 1]
                bottom_fluid = not self.solid_mask[i, j - 1]

                if not (right_fluid or left_fluid or top_fluid or bottom_fluid):
                    continue

                # Outward normal estimate from solid to fluid neighbors
                nx = 0.0
                ny = 0.0

                if right_fluid:
                    nx += 1.0
                if left_fluid:
                    nx -= 1.0
                if top_fluid:
                    ny += 1.0
                if bottom_fluid:
                    ny-= 1.0
                
                norm = np.hypot(nx, ny)
                if norm < 1e-12:
                    continue

                nx /= norm
                ny /= norm

                # Estimate local boundary segment lengths ds
                # Mostly vertical normal -> vertical face -> ds ~ dy
                # Mostly horizontal normal -> horizontal face -> ds ~ dx
                # Diagonal/corner -> use doagonal length

                if abs(nx) > 0.9 and abs(ny) < 0.1:
                    ds = self.dy
                elif abs(ny) > 0.9 and abs(nx) < 0.1:
                    ds = self.dx
                else:
                    ds = np.sqrt(self.dx**2 + self.dy**2)

                # Pressure at boundary: average only adjacent fluid neighbors
                p_sum = 0.0
                count = 0

                if right_fluid:
                    p_sum += self.p[i + 1, j]
                    count+= 1 
                if left_fluid:
                    p_sum += self.p[i - 1, j]
                    count += 1
                if top_fluid:
                    p_sum += self.p[i, j + 1]
                    count += 1
                if bottom_fluid:
                    p_sum += self.p[i, j - 1]
                    count += 1

                if count == 0:
                    continue

                p_b = p_sum / count

                # Pressure traction on body = -p * n
                dFx = -p_b * nx * ds
                dFy = -p_b * ny * ds

                Fx += dFx 
                Fy += dFy

        Cd_p, Cl_p = self.compute_force_coefficients(Fx, Fy)

        return Fx, Fy, Cd_p, Cl_p
    
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
        
    def run_simulation(self, n_steps, plot_interval=50):
        """Run the CFD simulation"""
        print(f"Running {self.geometry_type} simulation...")
        print(f"Grid: {self.nx}x{self.ny}, Re = {self.Re}, dt = {self.dt}")
        
        # Storage for results
        times = []
        drags = []
        lifts = []
        
        # Setup visualization with better subplot arrangement
        plt.ion()  # Turn on interactive mode
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'2D Navier-Stokes Simulation - {self.geometry_type.title()}', fontsize=16)
        
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
            if step % plot_interval == 0:
                self.plot_results(axes, step, times, drags, lifts)
                plt.draw()
                plt.pause(0.001)  # Small pause to update display
            
            # Print progress with more details
            if step % 100 == 0:
                u_max = np.max(np.sqrt(self.u**2 + self.v**2))
                p_max = np.max(np.abs(self.p))
                print(f"Step {step:4d}/{n_steps}, Time = {step*self.dt:6.3f}, "
                      f"Max U = {Fx:.4f}, Lift = {Fy:.4f}")
        
        plt.ioff()  # Turn off interactive mode
        input("Simulation done. Close when finished.")
        return times, drags, lifts
    
    def plot_results(self, axes, step, times, drags, lifts):
        """Plot velocity field, pressure, and force coefficients"""
        # Clear axes but preserve colorbars
        axes[0,0].clear()
        axes[0,1].clear()
        axes[1,0].clear()
        axes[1,1].clear()
        
        # Velocity magnitude
        vel_mag = np.sqrt(self.u**2 + self.v**2)
        vel_mag_plot = vel_mag.copy()
        vel_mag_plot[self.solid_mask] = np.nan
        
        im1 = axes[0,0].contourf(self.X, self.Y, vel_mag_plot, levels=20, cmap='viridis')
        axes[0,0].contour(self.X, self.Y, self.solid_mask.astype(float), levels=[0.5], colors='white', linewidths=2)
        axes[0,0].set_title(f'Velocity Magnitude (Step {step})')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('y')
        axes[0,0].set_aspect('equal')
        
        # Handle colorbar for velocity - create only once
        if self.cbar1 is None:
            self.cbar1 = plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
        else:
            self.cbar1.update_normal(im1)
        
        # Pressure field
        p_plot = self.p.copy()
        p_plot[self.solid_mask] = np.nan
        
        im2 = axes[0,1].contourf(self.X, self.Y, p_plot, levels=20, cmap='RdBu_r')
        axes[0,1].contour(self.X, self.Y, self.solid_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
        axes[0,1].set_title('Pressure Field')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('y')
        axes[0,1].set_aspect('equal')
        
        # Handle colorbar for pressure - create only once
        if self.cbar2 is None:
            self.cbar2 = plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
        else:
            self.cbar2.update_normal(im2)
        
        # Streamlines
        u_stream = self.u.T.copy()
        v_stream = self.v.T.copy()
        
        # Mask velocities in solid regions
        solid_mask_T = self.solid_mask.T
        u_stream[solid_mask_T] = 0
        v_stream[solid_mask_T] = 0
        
        try:
            axes[1,0].streamplot(self.x, self.y, u_stream, v_stream, 
                               density=1.5, color='blue', linewidth=1.0, 
                               broken_streamlines=False)
        except:
            # Fallback: use quiver plot if streamplot fails
            skip = max(1, min(self.nx, self.ny) // 20)
            axes[1,0].quiver(self.X[::skip, ::skip], self.Y[::skip, ::skip], 
                           self.u[::skip, ::skip], self.v[::skip, ::skip], 
                           scale=20, color='blue', alpha=0.7)
        
        axes[1,0].contour(self.X, self.Y, self.solid_mask.astype(float), 
                         levels=[0.5], colors='red', linewidths=2)
        axes[1,0].set_title('Streamlines/Flow Vectors')
        axes[1,0].set_xlabel('x')
        axes[1,0].set_ylabel('y')
        axes[1,0].set_aspect('equal')
        
        # Force coefficients
        if len(times) > 1:
            axes[1,1].plot(times, drags, 'b-', label='Drag', linewidth=2)
            axes[1,1].plot(times, lifts, 'r-', label='Lift', linewidth=2)
            axes[1,1].set_title('Force Coefficients vs Time')
            axes[1,1].set_xlabel('Time')
            axes[1,1].set_ylabel('Force Coefficient')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            # Add current values as text
            if len(drags) > 0:
                axes[1,1].text(0.02, 0.98, f'Current Drag: {drags[-1]:.4f}\nCurrent Lift: {lifts[-1]:.4f}',
                              transform=axes[1,1].transAxes, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Example usage and main execution
if __name__ == "__main__":
    # Simulation parameters - reduced for faster testing
    nx, ny = 400, 200      # Reduced grid resolution for faster computation
    Lx, Ly = 16.0, 10.5      # Domain size
    Re = 1000                # Reynolds number
    dtMax = 1e-3             # Smaller time step for stability
    n_steps = 2000         # Number of time steps
    
    print("2D Navier-Stokes CFD Solver")
    print("=" * 40)
    
    # Choose geometry: 'cylinder' or 'airfoil'
    geometry = 'naca4412'  # Start with cylinder for easier testing
    
    # Choose solver type
    use_sparse = True  # Sparse solver is generally more robust
    
    print(f"Geometry: {geometry}")
    print(f"Solver type: {'Sparse Matrix' if use_sparse else 'Iterative Gauss-Seidel'}")
    
    # Create and run solver
    solver = NavierStokesSolver2D(nx, ny, Lx, Ly, Re, dtMax, 
                                geometry_type=geometry, 
                                use_sparse_solver=use_sparse, alpha_deg=0.0)
    
    # Run simulation
    start_time = time.time()
    times, drags, lifts = solver.run_simulation(n_steps, plot_interval=20)
    end_time = time.time()
    
    print(f"\nSolver performance: {end_time - start_time:.2f} seconds")
    
    # Final results
    print("\nSimulation completed!")
    if len(drags) > 0:
        print(f"Final drag coefficient: {drags[-1]:.4f}")
        print(f"Final lift coefficient: {lifts[-1]:.4f}")
        
        # Additional analysis
        if len(drags) > 100:
            avg_drag = np.mean(drags[-100:])  # Average over last 100 steps
            avg_lift = np.mean(lifts[-100:])
            print(f"Time-averaged drag (last 100 steps): {avg_drag:.4f}")
            print(f"Time-averaged lift (last 100 steps): {avg_lift:.4f}")
    else:
        print("No force data collected!")
