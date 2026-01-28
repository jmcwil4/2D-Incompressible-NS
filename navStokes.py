import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve, splu
import time

class NavierStokesSolver2D:
    """
    2D Navier-Stokes solver using finite differences with immersed boundary method
    for handling complex geometries like cylinders and airfoils
    """
    
    def __init__(self, nx, ny, Lx, Ly, Re, dtMax=0.01, geometry_type='cylinder', use_sparse_solver=True):
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
        self.solid_mask = self.create_geometry_mask()
        
        # Initialize boundary conditions
        self.set_initial_conditions()
        
        # Clear the pressure matrix cache
        self._pressure_matrix = None
        self.pressure_lu = None
        
    def create_geometry_mask(self):
        """Create mask for solid boundaries (True = solid, False = fluid)"""
        mask = np.zeros((self.nx, self.ny), dtype=bool)
        
        if self.geometry_type == 'cylinder':
            # Circular cylinder at center
            center_x, center_y = self.Lx * 0.3, self.Ly * 0.5
            radius = min(self.Lx, self.Ly) * 0.1
            
            for i in range(self.nx):
                for j in range(self.ny):
                    if (self.X[i,j] - center_x)**2 + (self.Y[i,j] - center_y)**2 <= radius**2:
                        mask[i,j] = True
                        
        elif self.geometry_type == 'naca0012':
            # NACA 0012 airfoil
            center_x, center_y = self.Lx * 0.3, self.Ly * 0.5
            chord = min(self.Lx, self.Ly) * 0.4  # Larger chord for visibility
            
            for i in range(self.nx):
                for j in range(self.ny):
                    x_rel = (self.X[i,j] - center_x) / chord
                    y_rel = (self.Y[i,j] - center_y) / chord
                    
                    # Check if point is within airfoil chord length
                    if 0 <= x_rel <= 1:
                        # NACA 0012 thickness distribution (half-thickness)
                        t = 0.12  # Maximum thickness ratio
                        yt = (t/0.2) * (0.2969*np.sqrt(x_rel) - 0.1260*x_rel - 
                              0.3516*x_rel**2 + 0.2843*x_rel**3 - 0.1015*x_rel**4)
                        
                        # Point is inside airfoil if within thickness envelope
                        if abs(y_rel) <= yt:
                            mask[i,j] = True

        elif self.geometry_type == 'naca4412':
            # NACA 4412 airfoil parameters
            center_x, center_y = self.Lx * 0.3, self.Ly * 0.5
            chord = min(self.Lx, self.Ly) * 0.4 # Chord length

            m = 0.04
            p = 0.4
            t = 0.12

            for i in range(self.nx):
                for j in range(self.ny):
                    x = (self.X[i, j] - center_x) / chord
                    y = (self.Y[i, j] - center_y) / chord

                    if 0 <= x <= 1:
                        # thickness distribution
                        yt =  5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x
                            - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

                        # Camber line and slope
                        if x < p:
                            yc = (m / p**2) * (2 * p * x - x**2)
                            dyc_dx = (2 * m / p**2) * (p - x)
                        else:
                            yc = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x - x**2)
                            dyc_dx = (2 * m / (1 - p)**2) * (p - x)

                        theta = np.arctan(dyc_dx)

                        # Rotate coordinates into airfoil frame
                        y_upper = yc + yt * np.cos(theta)
                        y_lower = yc - yt * np.cos(theta)

                        if y_lower <= y <= y_upper:
                            mask[i, j] = True


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
        qR = qc[1:, :] - 0.5 * slope[:1, :]

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

        diff_u[1:-1, 1:-1] = self.nu * lap_u
        diff_v[1:-1, 1:-1] = self.nu * lap_v
        
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
        drag, lift = 0.0, 0.0
        
        # Improved force calculation using pressure integration
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                if self.solid_mask[i,j]:
                    # Check if this is a boundary point (adjacent to fluid)
                    boundary_point = False
                    normal_x, normal_y = 0.0, 0.0
                    
                    if not self.solid_mask[i+1,j]:  # Right neighbor is fluid
                        boundary_point = True
                        normal_x += 1.0
                    if not self.solid_mask[i-1,j]:  # Left neighbor is fluid
                        boundary_point = True
                        normal_x -= 1.0
                    if not self.solid_mask[i,j+1]:  # Top neighbor is fluid
                        boundary_point = True
                        normal_y += 1.0
                    if not self.solid_mask[i,j-1]:  # Bottom neighbor is fluid
                        boundary_point = True
                        normal_y -= 1.0
                    
                    if boundary_point:
                        # Normalize normal vector
                        norm = np.sqrt(normal_x**2 + normal_y**2)
                        if norm > 0:
                            normal_x /= norm
                            normal_y /= norm
                            
                            # Get pressure at boundary (interpolate from nearby fluid points)
                            p_boundary = 0.0
                            count = 0
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    ni, nj = i + di, j + dj
                                    if (0 <= ni < self.nx and 0 <= nj < self.ny and 
                                        not self.solid_mask[ni, nj]):
                                        p_boundary += self.p[ni, nj]
                                        count += 1
                            
                            if count > 0:
                                p_boundary /= count
                                
                                # Force contribution (pressure * area * normal)
                                area_element = self.dx * self.dy
                                drag += p_boundary * normal_x * area_element
                                lift += p_boundary * normal_y * area_element
        
        return drag, lift
    
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
            drag, lift = self.compute_forces()
            times.append(t)
            drags.append(drag)
            lifts.append(lift)
            
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
                      f"Max U = {u_max:.3f}, Max |P| = {p_max:.3f}, "
                      f"Drag = {drag:.4f}, Lift = {lift:.4f}")
        
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
    Re = 10e5                # Reynolds number
    dtMax = 0.005             # Smaller time step for stability
    n_steps = 2000         # Number of time steps
    
    print("2D Navier-Stokes CFD Solver")
    print("=" * 40)
    
    # Choose geometry: 'cylinder' or 'airfoil'
    geometry = 'naca0012'  # Start with cylinder for easier testing
    
    # Choose solver type
    use_sparse = True  # Sparse solver is generally more robust
    
    print(f"Geometry: {geometry}")
    print(f"Solver type: {'Sparse Matrix' if use_sparse else 'Iterative Gauss-Seidel'}")
    
    # Create and run solver
    solver = NavierStokesSolver2D(nx, ny, Lx, Ly, Re, dtMax, 
                                geometry_type=geometry, 
                                use_sparse_solver=use_sparse)
    
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
