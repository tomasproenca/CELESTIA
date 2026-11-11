import cupy as cp
import nvtx

class GenerateParticles:
    """
    Generates a 3D particle system with random masses, positions, and velocities
    DIRECTLY ON THE GPU using CuPy.

    All particle properties are stored as 1D CuPy arrays (GPU memory only).

    Returned structure:
        {
            "mass":  cupy.ndarray,
            "pos_x": cupy.ndarray,
            "pos_y": cupy.ndarray,
            "pos_z": cupy.ndarray,
            "vel_x": cupy.ndarray,
            "vel_y": cupy.ndarray,
            "vel_z": cupy.ndarray
        }
    """
    def __init__(self, config):
        self.config = config
        cp.random.seed(self.config["seed"])
        self.N = self.config["num_particles"]

    # =========
    # Mass
    # =========
    def _mass(self):
        with nvtx.annotate("GPU_Mass", color="red"):
            m_min, m_max = self.config["mass_limits"]
            return cp.random.uniform(m_min, m_max, size=self.N, dtype=cp.float32)

    # =========
    # Positions
    # =========
    def _pos(self):
        with nvtx.annotate("GPU_Pos", color="blue"):
            min_space, max_space = self.config["space_limits"]
            return cp.random.rand(self.N, dtype=cp.float32) * (max_space - min_space) + min_space

    # ============
    # Velocities
    # ============
    def _vel(self):
        with nvtx.annotate("GPU_Vel", color="purple"):
            v_min, v_max = self.config["velocity_limits"]
            return cp.random.uniform(v_min, v_max, size=self.N, dtype=cp.float32)

    # ================
    # Final generation
    # ================
    def generate(self):
        with nvtx.annotate("GPU_Particle_Generation", color="orange"):
            particles = {
                "mass": self._mass(),
                "pos_x": self._pos(),
                "pos_y": self._pos(),
                "pos_z": self._pos(),
                "vel_x": self._vel(),
                "vel_y": self._vel(),
                "vel_z": self._vel(),
            }
        return particles