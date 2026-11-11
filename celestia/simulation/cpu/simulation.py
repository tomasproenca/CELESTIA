import time
import nvtx
from celestia.simulation.cpu.compute import compute_cpu
from celestia.utils.config import SIMULATION_CONFIG, SAVE
from celestia.utils.save import TrajectorySaver

def run_simulation(particles):
    """
    N-body simulation loop on CPU using dict-of-arrays format.

    Particles live in CPU memory for the entire simulation.

    Parameters
    ----------
    particles : dict of 1D np.ndarray
        Particle dataset with keys:
            'mass', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z'

    Returns
    -------
    trajectory : list of dict of 1D np.ndarray or None
        List of snapshots of particle positions at each render interval (if SAVE=True).
    sim_time : float
        Total simulation time in seconds.
    """
    saver = TrajectorySaver()

    softening = 1e-1
    trajectory = [] if SAVE else None
    file_path = None
    start_time = time.time()

    dt = SIMULATION_CONFIG["dt"]
    bounce = SIMULATION_CONFIG["bounce"]
    g_force = SIMULATION_CONFIG["g_force"]
    min_limit = SIMULATION_CONFIG["space_min"]
    max_limit = SIMULATION_CONFIG["space_max"]

    with nvtx.annotate("Simulation_Loop", color="yellow"):
        for step in range(SIMULATION_CONFIG["num_steps"]):
            with nvtx.annotate(f"Step_{step}", color="blue"):
                step_start = time.perf_counter()

                # --- Compute forces and integrate ---
                with nvtx.annotate("Compute_Step", color="green"):
                    compute_cpu(particles, dt, bounce, g_force, min_limit, max_limit, softening)

                step_time = time.perf_counter() - step_start

                is_render_step = (
                    step % SIMULATION_CONFIG["render_interval"] == 0
                    or step == SIMULATION_CONFIG["num_steps"] - 1
                )

                if SAVE and is_render_step:
                    with nvtx.annotate("Save_Trajectory", color="red"):
                        step_data = {
                            'pos_x': particles['pos_x'].copy(),
                            'pos_y': particles['pos_y'].copy(),
                            'pos_z': particles['pos_z'].copy(),
                        }
                        file_path = saver.save_step(step_data, step_index=step, file_path=file_path)
                        trajectory.append(step_data)

                if is_render_step:
                    print(
                        f"[CPU] Step {step+1}/{SIMULATION_CONFIG['num_steps']} | "
                        f"Step time: {step_time*1000:.3f} ms"
                    )

    sim_time = time.time() - start_time
    return trajectory, sim_time
