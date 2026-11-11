import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from celestia.utils.config import SIMULATION_CONFIG


class NBodyRenderer:
    """
    Renderer for N-body simulations stored as HDF5 trajectories.
    """

    def __init__(self, trajectory, title="N-Body Simulation"):
        self.trajectory = trajectory
        self.title = title
        self.total_frames = len(trajectory)

        self.fig = None
        self.ax = None
        self.scatter = None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        """Render the trajectory as a 3D animation."""
        self._setup_figure()

        interval_ms = SIMULATION_CONFIG["dt"] * SIMULATION_CONFIG["render_interval"] * 1000

        self.anim = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self.total_frames,
            init_func=self._init_frame,
            blit=False,
            interval=interval_ms,
        )

        plt.show()

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.scatter = self.ax.scatter([], [], [], c="blue", s=3, alpha=0.5)

        min_bound = SIMULATION_CONFIG["space_min"]
        max_bound = SIMULATION_CONFIG["space_max"]

        self.ax.set_xlim(min_bound, max_bound)
        self.ax.set_ylim(min_bound, max_bound)
        self.ax.set_zlim(min_bound, max_bound)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(self.title)

    def _init_frame(self):
        self.scatter._offsets3d = ([], [], [])
        return (self.scatter,)

    def _update_frame(self, frame_index):
        frame = self.trajectory[frame_index]

        positions = np.stack(
            (frame["pos_x"], frame["pos_y"], frame["pos_z"]),
            axis=-1
        )

        self.scatter._offsets3d = (
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
        )

        self.ax.set_title(
            f"{self.title} — Step {frame_index + 1} / {self.total_frames}"
        )

        return (self.scatter,)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_h5(file_path):
        """
        Load trajectory from an HDF5 file.

        Returns
        -------
        list[dict]
        """
        file_path = Path(file_path)
        trajectory = []

        with h5py.File(file_path, "r") as h5:
            pos_x = h5["pos_x"][:]
            pos_y = h5["pos_y"][:]
            pos_z = h5["pos_z"][:]

            for step in range(pos_x.shape[0]):
                trajectory.append({
                    "pos_x": pos_x[step],
                    "pos_y": pos_y[step],
                    "pos_z": pos_z[step],
                })

        return trajectory

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @classmethod
    def run_cli(cls):
        parser = argparse.ArgumentParser(
            description="Render N-body simulation from an HDF5 output file."
        )
        parser.add_argument(
            "file",
            type=str,
            help="Path to .h5 trajectory file"
        )
        parser.add_argument(
            "--title",
            type=str,
            default="N-Body Simulation (Loaded)",
            help="Plot title"
        )

        args = parser.parse_args()

        trajectory = cls.load_h5(args.file)
        renderer = cls(trajectory, title=args.title)
        renderer.render()


if __name__ == "__main__":
    NBodyRenderer.run_cli()
