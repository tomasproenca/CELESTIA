from celestia.generate.manager import GenerateManager
from celestia.simulation.engine import EngineSimulation
from celestia.utils.debug import Debug
from celestia.utils.logger import Logger
from celestia.render.render import NBodyRenderer
from celestia.utils.config import PROFILING
import gc

"""
Main simulation script for N-body particle dynamics.

Workflow
--------

1. Generate:
   Creates the initial particle dataset on the selected device as a dictionary
   of 1D arrays:
       - 'mass', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z'
   Returns the GPU-resident dictionary and the generation timing metadata.

2. Confirmation:
   Displays a technical summary of particle properties and physics settings.
   User confirmation is required to proceed, unless PROFILING mode is active,
   which bypasses interactive prompts for automated testing.

3. Simulation:
   Executes the N-body physics engine on CPU or GPU.
   Returns:
       - trajectory: A list of snapshots (if SAVE is enabled).
       - sim_time: Execution duration of the integration loop.

4. Save (Optional):
   If SAVE is True, snapshots are stored in an HDF5 file. GPU snapshots
   are transferred to CPU first.

5. Render:
   Generates a visual animation of the particle trajectories using Matplotlib.
   This step is strictly disabled when PROFILING is True to avoid CPU-side
   rendering bottlenecks and GPU context interference.
"""

def main():
    debugger = Debug()
    logger = Logger()

    # ========
    # Generate
    # ========
    logger.generate()
    generate = GenerateManager()
    particles, gen_time = generate.generate()
    debugger.generation_time(gen_time)

    # ============
    # Confirmation
    # ============
    logger.confirmation()
    debugger.pre_simulation(particles)
    if not PROFILING:
        while True:
            print("")
            user_input = input(" *** Do you want to continue with the simulation? [Y/N]: ").strip().upper()
            if user_input in ("Y", "N"):
                break
            print("     Invalid response! Please enter only Y or N.")

        if user_input == "N":
            print("     Simulation canceled by the user.")
            del particles
            gc.collect()
            return
        print("     Continuing with the simulation...")

    # ==========
    # Simulation
    # ==========
    logger.simulation()
    engine = EngineSimulation(particles)
    trajectory, sim_time = engine.run()
    debugger.simulation_time(trajectory, sim_time)

    # ======
    # Render
    # ======
    if not PROFILING:
        if trajectory is not None:
            logger.render()
            renderer = NBodyRenderer(trajectory, title="N-Body Simulation")
            renderer.render()
    else:
        print("     PROFILING mode active, skipping render.")

    # Cleanup
    del particles
    gc.collect()

if __name__ == "__main__":
    main()
