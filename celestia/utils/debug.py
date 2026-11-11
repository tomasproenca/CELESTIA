# Data Setup
from celestia.utils.config import DEVICE, IDX_PARTICLES, PARTICLE_CONFIG
import numpy as np

class Debug:
    """
    Class to help track and verify the simulation.
    """
    def pre_simulation(self, particles):

        # ------------------------
        # Settings
        # ------------------------
        print(f" Number of particles: {PARTICLE_CONFIG['num_particles']}")
        print(f" Random seed: {PARTICLE_CONFIG['seed']}")
        print(f" Space limits: {PARTICLE_CONFIG['space_limits']}")
        print(f" Velocity limits: {PARTICLE_CONFIG['velocity_limits']}")
        print(f" Mass limits: {PARTICLE_CONFIG['mass_limits']}")

        # ------------------------
        # Basic validation
        # ------------------------
        if not isinstance(particles, dict):
            print(" Particles is not a dictionary!")
            return

        keys = list(particles.keys())
        sizes = {key: particles[key].shape[0] for key in keys}

        print("\n Particle attributes:")
        for key, size in sizes.items():
            print(f"  {key}: shape={particles[key].shape}, dtype={particles[key].dtype}")

        if len(set(sizes.values())) != 1:
            print("\n ⚠️ WARNING: Particle arrays have inconsistent sizes!")
            return

        num_particles = next(iter(sizes.values()))

        # ------------------------
        # Show a sample of particles
        # ------------------------
        print("\n Sample of generated particles (first 5):")

        header = ["mass", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]

        for i in range(min(5, num_particles)):
            values = []
            for key in header:
                val = particles[key][i]
                values.append(f"{val:.4f}")
            print(f"  Particle {i}: " + ", ".join(values))

        print(" ====================")

    def generation_time(self, gen_time):
        print(f" *** PARTICLE GENERATION TIME: {gen_time:.6f} s")

    def simulation_time(self, trajectory, sim_time):
        if trajectory is None:
            print(f" *** {DEVICE} SIMULATION DONE (no snapshots), took {sim_time:.3f} s")
        else:
            print(f" *** {DEVICE} SIMULATION DONE: {len(trajectory)} frames, took {sim_time:.3f} s")