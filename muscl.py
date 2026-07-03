from numba import njit
import numpy as np

@njit(cache=True)
def musclx(q):
    """
    MUSCL reconstruction in x-direction using an inline minmod limiter.

    Input:
        q : (nx, ny)

    Returns:
        qL, qR : (nx-3, ny-2)
    """

    nx, ny = q.shape

    qL = np.empty((nx-3, ny-2), dtype=q.dtype)
    qR = np.empty((nx-3, ny-2), dtype=q.dtype)

    for i in range(1, nx-2):
        for j in range(1, ny-1):

            qc = q[i, j]

            dl = qc - q[i-1, j]
            dr = q[i+1, j] - qc

            # Inline minmod
            if dl * dr <= 0.0:
                slope = 0.0
            elif abs(dl) < abs(dr):
                slope = dl
            else:
                slope = dr

            qL[i-1, j-1] = qc + 0.5 * slope
            qR[i-1, j-1] = q[i+1, j] - 0.5 * slope

    return qL, qR

@njit(cache=True)
def muscly(q):
    """
    MUSCL reconstruction in y-direction using an inline minmod limiter.

    Input:
        q : (nx, ny)

    Returns:
        qL, qR : (nx-2, ny-3)
    """

    nx, ny = q.shape

    qL = np.empty((nx-2, ny-3), dtype=q.dtype)
    qR = np.empty((nx-2, ny-3), dtype=q.dtype)

    for i in range(1, nx-1):
        for j in range(1, ny-2):

            qc = q[i, j]

            db = qc - q[i, j-1]
            dt = q[i, j+1] - qc

            # Inline minmod
            if db * dt <= 0.0:
                slope = 0.0
            elif abs(db) < abs(dt):
                slope = db
            else:
                slope = dt

            qL[i-1, j-1] = qc + 0.5 * slope
            qR[i-1, j-1] = q[i, j+1] - 0.5 * slope

    return qL, qR
