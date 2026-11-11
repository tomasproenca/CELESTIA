import numpy as np
import nvtx

class GenerateParticles:
    """
    Generates a 3D particle system with random masses, positions, and velocities.

    Each particle property is stored as a 1D NumPy array.

    The final generated dataset is returned as a **dictionary of 1D arrays**:
        {
            "mass":  array of particle masses,
            "pos_x": array of x positions,
            "pos_y": array of y positions,
            "pos_z": array of z positions,
            "vel_x": array of x velocities,
            "vel_y": array of y velocities,
            "vel_z": array of z velocities
        }

    Configuration (config) is expected as a dictionary with:
        - num_particles: number of particles
        - mass_limits: [min_mass, max_mass]
        - space_limits: [min_space, max_space]
        - velocity_limits: [min_velocity, max_velocity]
        - seed: random seed for reproducibility
    """
    def __init__(self, config):
        self.config = config

    # =========
    # Mass
    # =========
    def _mass(self):
        with nvtx.annotate("CPU_Mass", color="red"):
            m_min, m_max = self.config["mass_limits"]
            return np.random.uniform(m_min, m_max, size=self.config["num_particles"]).astype(np.float32)

    # =========
    # Positions
    # =========
    def _pos_x(self):
        with nvtx.annotate("CPU_Pos_X", color="blue"):
            min_space, max_space = self.config["space_limits"]
            return (np.random.rand(self.config["num_particles"]) * (max_space - min_space) + min_space).astype(np.float32)

    def _pos_y(self):
        with nvtx.annotate("CPU_Pos_Y", color="blue"):
            min_space, max_space = self.config["space_limits"]
            return (np.random.rand(self.config["num_particles"]) * (max_space - min_space) + min_space).astype(np.float32)

    def _pos_z(self):
        with nvtx.annotate("CPU_Pos_Z", color="blue"):
            min_space, max_space = self.config["space_limits"]
            return (np.random.rand(self.config["num_particles"]) * (max_space - min_space) + min_space).astype(np.float32)

    # ============
    # Velocities
    # ============
    def _vel_x(self):
        with nvtx.annotate("CPU_Vel_X", color="purple"):
            v_min, v_max = self.config["velocity_limits"]
            return np.random.uniform(v_min, v_max, size=self.config["num_particles"]).astype(np.float32)

    def _vel_y(self):
        with nvtx.annotate("CPU_Vel_Y", color="purple"):
            v_min, v_max = self.config["velocity_limits"]
            return np.random.uniform(v_min, v_max, size=self.config["num_particles"]).astype(np.float32)

    def _vel_z(self):
        with nvtx.annotate("CPU_Vel_Z", color="purple"):
            v_min, v_max = self.config["velocity_limits"]
            return np.random.uniform(v_min, v_max, size=self.config["num_particles"]).astype(np.float32)

    # ================
    # Final generation
    # ================
    def generate(self):
        np.random.seed(self.config["seed"])
        with nvtx.annotate("CPU_Particle_Generation", color="green"):
            particles = {
                "mass": self._mass(),
                "pos_x": self._pos_x(),
                "pos_y": self._pos_y(),
                "pos_z": self._pos_z(),
                "vel_x": self._vel_x(),
                "vel_y": self._vel_y(),
                "vel_z": self._vel_z(),
            }
        return particles