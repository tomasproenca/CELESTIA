import time
from celestia.utils.config import PARTICLE_CONFIG, DEVICE

if DEVICE == "GPU":
    from celestia.generate.particles_gpu import GenerateParticles
elif DEVICE == "CPU":
    from celestia.generate.particles_cpu import GenerateParticles
else:
    raise ValueError(f"Unknown DEVICE: {DEVICE}")


class GenerateManager:
    """
    Manages the generation of particle datasets for the simulation.
    """

    def generate(self):
        start_time = time.time()

        gen = GenerateParticles(PARTICLE_CONFIG)
        particles = gen.generate()

        end_time = time.time()
        return particles, end_time - start_time
