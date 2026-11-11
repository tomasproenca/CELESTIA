from numba import cuda, float32
import math

from celestia.utils.config import THREADS_PER_BLOCK

@cuda.jit
def compute_gpu(mass, pos_x, pos_y, pos_z, next_pos_x, next_pos_y, next_pos_z,
                vel_x, vel_y, vel_z, dt, bounce, g_force, min_limit, max_limit,
                softening, N):
    """
    CUDA kernel to update particle positions and velocities in an N-body simulation.

    Each GPU thread updates one particle using Newtonian gravity, updates velocities
    and positions, and handles wall collisions.

    Parameters
    ----------
    mass, pos_x, pos_y, pos_z, next_pos_x, next_pos_y, next_pos_z,
    vel_x, vel_y, vel_z : 1D GPU arrays
        Input positions, output positions, masses and velocities.
    dt : float
        Simulation time step.
    bounce : float
        Wall collision damping factor.
    g_force : float
        Gravitational constant.
    min_limit, max_limit : float
        Simulation box boundaries.
    softening : float
        Softening factor to avoid singularities.
    N : int
        Number of particles
    """

    i = cuda.grid(1)
    active = i < N

    # -------------------------
    # Shared memory (por bloco)
    # -------------------------
    sh_pos_x = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float32)
    sh_pos_y = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float32)
    sh_pos_z = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float32)
    sh_mass  = cuda.shared.array(shape=THREADS_PER_BLOCK, dtype=float32)

    # -------------------------
    # Registers
    # -------------------------
    xi = pos_x[i] if active else 0.0
    yi = pos_y[i] if active else 0.0
    zi = pos_z[i] if active else 0.0

    ax = 0.0
    ay = 0.0
    az = 0.0

    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x

    # -------------------------
    # Gravity (tiled)
    # -------------------------
    for tile_start in range(0, N, block_size):
        j = tile_start + tid

        # Global → Shared
        if j < N:
            sh_pos_x[tid] = pos_x[j]
            sh_pos_y[tid] = pos_y[j]
            sh_pos_z[tid] = pos_z[j]
            sh_mass[tid]  = mass[j]
        else:
            sh_mass[tid] = 0.0

        cuda.syncthreads()

        # Compute using shared memory
        if active:
            for k in range(block_size):
                if tile_start + k < N and tile_start + k != i:
                    dx = sh_pos_x[k] - xi
                    dy = sh_pos_y[k] - yi
                    dz = sh_pos_z[k] - zi

                    dist2 = dx*dx + dy*dy + dz*dz + softening
                    inv_dist = 1.0 / math.sqrt(dist2)
                    inv_dist3 = inv_dist * inv_dist * inv_dist

                    factor = g_force * sh_mass[k] * inv_dist3

                    ax += dx * factor
                    ay += dy * factor
                    az += dz * factor

        cuda.syncthreads()

    # -------------------------
    # Update velocities
    # -------------------------
    if not active:
        return
    vel_x[i] += ax * dt
    vel_y[i] += ay * dt
    vel_z[i] += az * dt

    # -------------------------
    # Update positions
    # -------------------------
    next_pos_x[i] = xi + vel_x[i] * dt
    next_pos_y[i] = yi + vel_y[i] * dt
    next_pos_z[i] = zi + vel_z[i] * dt

    # -------------------------
    # Wall bounce
    # -------------------------
    if next_pos_x[i] < min_limit:
        next_pos_x[i] = min_limit
        vel_x[i] *= -bounce
    elif next_pos_x[i] > max_limit:
        next_pos_x[i] = max_limit
        vel_x[i] *= -bounce

    if next_pos_y[i] < min_limit:
        next_pos_y[i] = min_limit
        vel_y[i] *= -bounce
    elif next_pos_y[i] > max_limit:
        next_pos_y[i] = max_limit
        vel_y[i] *= -bounce

    if next_pos_z[i] < min_limit:
        next_pos_z[i] = min_limit
        vel_z[i] *= -bounce
    elif next_pos_z[i] > max_limit:
        next_pos_z[i] = max_limit
        vel_z[i] *= -bounce
