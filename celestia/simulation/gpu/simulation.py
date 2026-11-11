import time
import nvtx
import cupy as cp
from numba import cuda
from celestia.simulation.gpu.compute import compute_gpu
from celestia.utils.config import SIMULATION_CONFIG, THREADS_PER_BLOCK, SAVE
from celestia.utils.save import TrajectorySaver


def run_simulation(particles):
    """
    N-body simulation loop on GPU using CUDA (Numba) + CuPy.

    Particles live on the GPU for the entire simulation.
    CPU-GPU copies happen ONLY at render intervals.

    Parameters
    ----------
    particles : dict of 1D np.ndarray
        Particle dataset with keys:
            'mass', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z'
        Each array has length N (number of particles)

    Returns
    -------
    trajectory : list of dict of 1D np.ndarray
        List of snapshots of particle positions at each render interval.

    sim_time : float
        Total simulation time in seconds.
    """
    N = particles['mass'].size
    blocks = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    softening = cp.float32(1e-1)

    # --- Data setup ---
    with nvtx.annotate("Data_Setup", color="yellow"):
        d_particles = {key: cuda.as_cuda_array(arr) for key, arr in particles.items()}
        next_positions = {
            key: cp.empty_like(particles[key]) for key in ("pos_x", "pos_y", "pos_z")
        }

    trajectory = [] if SAVE else None
    file_path = None
    saver = TrajectorySaver()
    start_time = time.time()

    with nvtx.annotate("Simulation_Loop", color="orange"):
        for step in range(SIMULATION_CONFIG["num_steps"]):
            with nvtx.annotate(f"Step_{step}", color="blue"):
                step_start = time.perf_counter()

                # --- Kernel compute ---
                with nvtx.annotate("Kernel_Compute", color="green"):
                    compute_gpu[blocks, THREADS_PER_BLOCK](
                        d_particles['mass'],
                        d_particles['pos_x'],
                        d_particles['pos_y'],
                        d_particles['pos_z'],
                        next_positions['pos_x'],
                        next_positions['pos_y'],
                        next_positions['pos_z'],
                        d_particles['vel_x'],
                        d_particles['vel_y'],
                        d_particles['vel_z'],
                        SIMULATION_CONFIG["dt"],
                        SIMULATION_CONFIG["bounce"],
                        SIMULATION_CONFIG["g_force"],
                        SIMULATION_CONFIG["space_min"],
                        SIMULATION_CONFIG["space_max"],
                        softening,
                        N,
                    )
                    cuda.synchronize()
                    for key in next_positions:
                        particles[key], next_positions[key] = next_positions[key], particles[key]
                        d_particles[key] = cuda.as_cuda_array(particles[key])

                step_time = time.perf_counter() - step_start

                # --- Save & Transfer ---
                is_render_step = (
                    step % SIMULATION_CONFIG["render_interval"] == 0
                    or step == SIMULATION_CONFIG["num_steps"] - 1
                )

                if SAVE and is_render_step:
                    with nvtx.annotate("Save_and_Transfer_to_CPU", color="red"):
                        step_data = {
                            'pos_x': cp.asnumpy(d_particles['pos_x']),
                            'pos_y': cp.asnumpy(d_particles['pos_y']),
                            'pos_z': cp.asnumpy(d_particles['pos_z']),
                        }
                        file_path = saver.save_step(step_data, step_index=step, file_path=file_path)
                        trajectory.append(step_data)

                print(
                    f"[GPU] Step {step+1}/{SIMULATION_CONFIG['num_steps']} | "
                    f"Step time: {step_time*1000:.3f} ms"
                )

    sim_time = time.time() - start_time
    return trajectory, sim_time
