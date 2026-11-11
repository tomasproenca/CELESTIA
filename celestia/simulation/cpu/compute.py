import numpy as np
import nvtx

def compute_cpu(particles, dt, bounce, g_force, min_limit, max_limit, softening=1e-1):
    """
    Update particle positions and velocities on CPU using Newtonian gravity
    and wall collisions. Particle-particle collisions are not computed.

    Particles are represented as a dict of 1D np.ndarray:
        'mass', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z'

    Euler integration:
        v_new = v_old + a*dt
        x_new = x_old + v*dt
    """

    N = len(particles['mass'])

    ax = np.zeros(N, dtype=np.float32)
    ay = np.zeros(N, dtype=np.float32)
    az = np.zeros(N, dtype=np.float32)

    # ---------- Compute gravitational accelerations ----------
    with nvtx.annotate("Compute_Gravity", color="red"):
        for i in range(N):
            xi, yi, zi = particles['pos_x'][i], particles['pos_y'][i], particles['pos_z'][i]

            for j in range(N):
                if i == j:
                    continue
                dx = particles['pos_x'][j] - xi
                dy = particles['pos_y'][j] - yi
                dz = particles['pos_z'][j] - zi

                dist2 = dx*dx + dy*dy + dz*dz + softening
                dist = np.sqrt(dist2)
                mj = particles['mass'][j]
                a = g_force * mj / dist2
                ax[i] += a * dx / dist
                ay[i] += a * dy / dist
                az[i] += a * dz / dist

    # ---------- Update velocities ----------
    particles['vel_x'] += ax * dt
    particles['vel_y'] += ay * dt
    particles['vel_z'] += az * dt

    # ---------- Update positions and apply wall bounce ----------
    for coord, vel in zip(['pos_x','pos_y','pos_z'], ['vel_x','vel_y','vel_z']):
        particles[coord] += particles[vel] * dt
        mask_min = particles[coord] < min_limit
        mask_max = particles[coord] > max_limit
        np.clip(particles[coord], min_limit, max_limit, out=particles[coord])
        particles[vel][mask_min | mask_max] *= -bounce
