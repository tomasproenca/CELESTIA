# CELESTIA

Academic project for a high performance computing course. CELESTIA simulates an N-body particle system in a bounded 3D space, with CPU (NumPy) and GPU (Numba CUDA and CuPy) implementations. It can save particle positions to HDF5 and display an animation with Matplotlib. NVTX annotations support profiling. No performance benchmark is claimed here.

## Setup

Python 3.11 is recommended. Create and activate an environment, for example:

```powershell
conda create -n celestia python=3.11
conda activate celestia
python -m pip install -r requirements.txt
```

The default configuration uses `DEVICE=CPU`. For GPU execution with CUDA 13.x, install CuPy and the Numba CUDA toolkit dependencies, then set `DEVICE=GPU`:

```powershell
python -m pip install cupy-cuda13x "numba-cuda[cu13]"
```

Install only one CuPy variant. Other CUDA versions require matching packages. A compatible NVIDIA driver is also required. The project author confirmed a CPU run and a GPU run on Windows with an NVIDIA RTX 3060 Ti; this is not a formal compatibility test.

Copy `dotenv/.env.example` to `dotenv/.env` and edit the settings there. The `.env` file is local and excluded from Git. `SAVE=true` writes HDF5 trajectories to `output/`; large simulations can use substantial disk space. `PROFILING=true` skips the confirmation prompt and rendering.

## Run

```powershell
python main.py
```

To render a saved trajectory later:

```powershell
python -m celestia.render.render output/your_file.h5
```

The CPU and GPU paths use different random number generators, so the same seed does not imply identical initial particles.

## Repository contents

`main.py` runs generation, simulation, saving and optional rendering. `celestia/generate/` creates particles, `celestia/simulation/` contains the CPU and GPU engines, `celestia/render/` loads and visualizes trajectories, and `celestia/utils/` holds configuration and I/O helpers. Generated trajectories, profiler reports, caches and local configuration are excluded from this public version.

## License

MIT. See `LICENSE`.
