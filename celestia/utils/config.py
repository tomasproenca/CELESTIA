# Data setup
import os
from dotenv import load_dotenv
from pathlib import Path

# ==========================
# Paths e .env
# ==========================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DOTENV_PATH = PROJECT_ROOT / "dotenv" / ".env"
OUTPUT_PATH = PROJECT_ROOT / "output"

load_dotenv(dotenv_path=DOTENV_PATH)

# ==========================
# Execution
# ==========================
DEVICE = os.getenv("DEVICE", "CPU").upper()                     # GPU or CPU
SAVE = os.getenv("SAVE", "true").lower() == "true"              # Enable/disable simulation saving
PROFILING = os.getenv("PROFILING", "false").lower() == "true"   # Skip interactive confirmation and rendering


# ==========================
# Main parameters
# ==========================
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))                 # Seed for reproducibility
NUM_STEPS = int(os.getenv("NUM_STEPS", 100))                    # Number of simulation steps
DT = float(os.getenv("DT", 0.01))                               # Time step between simulation updates
RENDER_INTERVAL = int(os.getenv("RENDER_INTERVAL", 10))         # Render every N simulation steps
BOUNCE = float(os.getenv("BOUNCE", 1.0))                        # Bounce coefficient (particle–particle and particle–wall)
G_FORCE = float(os.getenv("G_FORCE", 0.001))                    # Gravitational constant used for force calculations

THREADS_PER_BLOCK = int(os.getenv("THREADS_PER_BLOCK", 256))    # GPU-only: threads per block (depends on the GPU used)


# ==========================
# Particles
# ==========================
NUM_PARTICLES = int(os.getenv("NUM_PARTICLES", 10))     # Number of particles

SPACE_MIN = float(os.getenv("SPACE_MIN", 0.0))         # Minimum simulation space boundary
SPACE_MAX = float(os.getenv("SPACE_MAX", 1.0))         # Maximum simulation space boundary

MASS_MIN = float(os.getenv("MASS_MIN", 1.0))           # Minimum particle mass at generation
MASS_MAX = float(os.getenv("MASS_MAX", 1.0))           # Maximum particle mass at generation

VELOCITY_MIN = float(os.getenv("VELOCITY_MIN", 0.0))   # Minimum initial particle velocity
VELOCITY_MAX = float(os.getenv("VELOCITY_MAX", 0.0))   # Maximum initial particle velocity


# ==========================
# Configuration dictionaries
# ==========================
SIMULATION_CONFIG = {
    "num_steps": NUM_STEPS,
    "dt": DT,
    "bounce": BOUNCE,
    "g_force": G_FORCE,
    "render_interval": RENDER_INTERVAL,
    "space_min": SPACE_MIN,
    "space_max": SPACE_MAX,
    "threads_per_block": THREADS_PER_BLOCK
}

PARTICLE_CONFIG = {
    "num_particles": NUM_PARTICLES,
    "seed": RANDOM_SEED,
    "space_limits": [SPACE_MIN, SPACE_MAX],
    "velocity_limits": [VELOCITY_MIN, VELOCITY_MAX],
    "mass_limits": [MASS_MIN, MASS_MAX]
}

# ==========================
# Particle data layout (index mapping)
# ==========================
IDX_PARTICLES = {
    "mass": 0,
    "pos_x": 1,
    "pos_y": 2,
    "pos_z": 3,
    "vel_x": 4,
    "vel_y": 5,
    "vel_z": 6
}
