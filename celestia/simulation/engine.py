# Data setup
from celestia.utils.config import DEVICE

class EngineSimulation:
    """
    N-body simulation engine that executes a particle system on CPU or GPU.

    This engine runs the simulation using the selected device, returning the
    trajectory of particle positions over time and the total simulation duration.

    Parameters
    ----------
    particles : dict of 1D np.ndarray
        Particle dataset with keys:
            - 'mass':  particle masses
            - 'pos_x', 'pos_y', 'pos_z': particle positions
            - 'vel_x', 'vel_y', 'vel_z': particle velocities
        Each array has length N (number of particles).

    device : str, optional
        Device to run the simulation on: "CPU" or "GPU" (default taken from config).

    Raises
    ------
    ValueError
        If the selected device is not "CPU" or "GPU".
    """
    def __init__(self, particles):
        self.particles = particles
        self.device = DEVICE.upper()

        # Validation
        if self.device not in ("CPU", "GPU"):
            raise ValueError(f"Invalid device '{DEVICE}'. Choose 'CPU' or 'GPU'.")

    def run(self):
        """
        Run the N-body simulation on the selected device.

        Returns
        -------
        trajectory : list of dict of 1D np.ndarray
            Each element of the list represents the particle dataset at a simulation step,
            with the same keys as the input 'particles' dictionary.

        sim_time : float
            Total time taken for the simulation in seconds.
        """
        if self.device == "CPU":
            from celestia.simulation.cpu.simulation import run_simulation
            trajectory, sim_time = run_simulation(self.particles)
            return trajectory, sim_time

        elif self.device == "GPU":
            from celestia.simulation.gpu.simulation import run_simulation
            trajectory, sim_time = run_simulation(self.particles)
            return trajectory, sim_time
