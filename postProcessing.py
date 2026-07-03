import numpy as np
import matplotlib.pyplot as plt

def plot_results(solver, axes, step, times, drags, lifts):
        """Plot velocity field, pressure, and force coefficients"""
        # Clear axes but preserve colorbars
        axes[0,0].clear()
        axes[0,1].clear()
        axes[1,0].clear()
        axes[1,1].clear()
        
        # Velocity magnitude
        vel_mag = np.sqrt(solver.u**2 + solver.v**2)
        vel_mag_plot = vel_mag.copy()
        vel_mag_plot[solver.solid_mask] = np.nan
        
        im1 = axes[0,0].contourf(solver.X, solver.Y, vel_mag_plot, levels=20, cmap='viridis')
        axes[0,0].contour(solver.X, solver.Y, solver.solid_mask.astype(float), levels=[0.5], colors='white', linewidths=2)
        axes[0,0].set_title(f'Velocity Magnitude (Step {step})')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('y')
        axes[0,0].set_aspect('equal')
        
        # Handle colorbar for velocity - create only once
        if solver.cbar1 is None:
            solver.cbar1 = plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
        else:
            solver.cbar1.update_normal(im1)
        
        # Pressure field
        p_plot = solver.p.copy()
        p_plot[solver.solid_mask] = np.nan
        
        im2 = axes[0,1].contourf(solver.X, solver.Y, p_plot, levels=20, cmap='RdBu_r')
        axes[0,1].contour(solver.X, solver.Y, solver.solid_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
        axes[0,1].set_title('Pressure Field')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('y')
        axes[0,1].set_aspect('equal')
        
        # Handle colorbar for pressure - create only once
        if solver.cbar2 is None:
            solver.cbar2 = plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
        else:
            solver.cbar2.update_normal(im2)
        
        # Streamlines
        u_stream = solver.u.T.copy()
        v_stream = solver.v.T.copy()
        
        # Mask velocities in solid regions
        solid_mask_T = solver.solid_mask.T
        u_stream[solid_mask_T] = 0
        v_stream[solid_mask_T] = 0
        
        try:
            axes[1,0].streamplot(solver.x, solver.y, u_stream, v_stream, 
                               density=1.5, color='blue', linewidth=1.0, 
                               broken_streamlines=False)
        except:
            # Fallback: use quiver plot if streamplot fails
            skip = max(1, min(solver.nx, solver.ny) // 20)
            axes[1,0].quiver(solver.X[::skip, ::skip], solver.Y[::skip, ::skip], 
                           solver.u[::skip, ::skip], solver.v[::skip, ::skip], 
                           scale=20, color='blue', alpha=0.7)
        
        axes[1,0].contour(solver.X, solver.Y, solver.solid_mask.astype(float), 
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

def plot_vorticity(solver):

        omega = solver.compute_vorticity()
        plt.figure(figsize=(10,6))

        vmax = np.nanmax(np.abs(omega))

        plt.contourf(
            solver.X,
            solver.Y,
            omega,
            levels=50,
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax
        )

        plt.colorbar(label='Vorticity')

        plt.contour(
            solver.X,
            solver.Y,
            solver.solid_mask.astype(float),
            levels=[0.5],
            colors='k'
        )

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Vorticity Field')

        plt.tight_layout()
        plt.show()

def plot_turbulence_ratio(solver):

        ratio = solver.compute_turbulence_ratio()

        plt.figure(figsize=(10,6))

        plt.contourf(
            solver.X,
            solver.Y,
            np.log10(np.maximum(ratio, 1e-6)),
            levels=50
        )

        plt.colorbar(label='log10(nu_t / nu)')

        plt.contour(
            solver.X,
            solver.Y,
            solver.solid_mask.astype(float),
            levels=[0.5],
            colors='k'
        )

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Turbulence Ratio')

        plt.tight_layout()
        plt.show()

def plot_surface_cp(solver):

        xU, cpU, xL, cpL = solver.compute_surface_cp()

        plt.figure(figsize=(10,6))

        plt.plot(
            xU,
            cpU,
            label="Upper Surface"
        )

        plt.plot(
            xL,
            cpL,
            label="Lower Surface"
        )

        plt.gca().invert_yaxis()

        plt.xlabel("x/c")
        plt.ylabel("Cp")

        plt.title(
            f"Cp Distribution\n"
            f"Re={solver.Re:.0f}, AoA={solver.alpha_deg:.1f}°"
        )

        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

def plot_pressure_distribution(solver):

        Cp = solver.compute_pressure_distribution()

        plt.figure(figsize=(10,6))

        plt.contourf(
            solver.X,
            solver.Y,
            Cp,
            levels=50,
            cmap='coolwarm'
        )

        plt.colorbar(label='Cp')

        plt.contour(
            solver.X,
            solver.Y,
            solver.solid_mask.astype(float),
            levels=[0.5],
            colors='k'
        )

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Pressure Coefficient Field')

        plt.tight_layout()
        plt.show()
