import numpy as np

def create_geometry_mask(solver):
        """Create mask for solid boundaries (True = solid, False = fluid).

        Fast, vectorized geometry creation for:
        - 'cylinder'
        - 'naca0012'  (also aliased by 'airfoil')
        - 'naca4412'

        Optional angle-of-attack support:
        - If solver.alpha_deg exists: degrees
        - Else if solver.alpha exists: radians
        - Else: 0
        """

        mask = np.zeros((solver.nx, solver.ny), dtype=bool)

        # Geometry placement (keep your original intent)
        center_x = solver.center_x
        center_y = solver.center_y

        # Optional AoA support (body rotated by +alpha relative to flow)
        if hasattr(solver, "alpha_deg"):
            alpha = np.deg2rad(float(solver.alpha_deg))
        elif hasattr(solver, "alpha"):
            alpha = float(solver.alpha)
        else:
            alpha = 0.0

        # Shift to body-centered coordinates
        x0 = solver.X - center_x
        y0 = solver.Y - center_y

        # Rotate grid into body frame by -alpha (so body appears at +alpha in lab frame)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        Xr = x0 * ca - y0 * sa
        Yr = x0 * sa + y0 * ca

        # -------------------------
        # Cylinder
        # -------------------------
        if solver.geometry_type == "cylinder":
            radius = min(solver.Lx, solver.Ly) * 0.1
            mask = (Xr**2 + Yr**2) <= radius**2
            return mask

        # -------------------------
        # NACA airfoils
        # -------------------------
        geom = solver.geometry_type.lower()
        if geom == "airfoil":   # your original alias
            geom = "naca0012"

        if geom not in ("naca0012", "naca4412"):
            # Unknown geometry -> no obstacle
            return mask

        # Use your existing chord definition (but vectorized)
        chord = 1.0

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
        mask = inside_airfoil.reshape(solver.nx, solver.ny)

        xu, yu, xl, yl = solver.get_airfoil_surface()

        return mask

def get_airfoil_surface(solver, npts=400):
    """
        Generate upper and lower surface coordinates for a NACA 4-digit airfoil.
        Returns
        -------
        xu, yu : upper surface coordinates (0 -> 1 chord)
        xl, yl : lower surface coordinates (0 -> 1 chord)
    """

    geometry = solver.geometry_type.lower()

    # ----------------------------
    # Select airfoil parameters
    # ----------------------------
    if geometry == "naca0012":
        m = 0.00
        p = 0.00
        t = 0.12

    elif geometry == "naca2412":
        m = 0.02
        p = 0.40
        t = 0.12

    elif geometry == "naca4412":
        m = 0.04
        p = 0.40
        t = 0.12

    else:
        raise ValueError(
            f"Unsupported airfoil: {solver.geometry_type}"
        )

    # ----------------------------
    # Chord coordinate
    # ----------------------------

    x = np.linspace(0.0, 1.0, npts)

    # Avoid sqrt singularity at x=0
    xs = np.clip(x, 1e-12, 1.0)

    # ----------------------------
    # Thickness distribution
    # ----------------------------

    yt = 5.0 * t * (
        0.2969 * np.sqrt(xs)
        - 0.1260 * xs
        - 0.3516 * xs**2
        + 0.2843 * xs**3
        - 0.1015 * xs**4
    )

    # ----------------------------
    # Symmetric airfoil
    # ----------------------------

    if m == 0.0:

        xu = x
        yu = yt

        xl = x
        yl = -yt

        return xu, yu, xl, yl
    # ----------------------------
    # Camber line
    # ----------------------------

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    left = x < p
    right = ~left

    yc[left] = (
        m / p**2
    ) * (
        2.0*p*x[left]
        - x[left]**2
    )

    yc[right] = (
        m / (1.0-p)**2
    ) * (
        (1.0 - 2.0*p)
        + 2.0*p*x[right]
        - x[right]**2
    )

    dyc_dx[left] = (
        2.0*m / p**2
    ) * (
        p - x[left]
    )

    dyc_dx[right] = (
        2.0*m / (1.0-p)**2
    ) * (
        p - x[right]
    )

    theta = np.arctan(dyc_dx)

    # ----------------------------
    # Upper surface
    # ----------------------------

    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)

    # ----------------------------
    # Lower surface
    # ----------------------------

    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)

    return xu, yu, xl, yl  