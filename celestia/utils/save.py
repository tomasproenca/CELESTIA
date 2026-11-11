from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
from celestia.utils.config import DEVICE, OUTPUT_PATH


class TrajectorySaver:
    """
    Utility to stream N-body simulation steps to an HDF5 file.
    """

    def __init__(self, output_path=OUTPUT_PATH, device=DEVICE):
        self.output_path = Path(output_path)
        self.device = device
        self.output_path.mkdir(parents=True, exist_ok=True)

    def save_step(self, step_data, step_index, file_path=None):
        """
        Save a single simulation step to an HDF5 file (streaming).

        Parameters
        ----------
        step_data : dict
            Dictionary with keys 'pos_x', 'pos_y', 'pos_z'.
        step_index : int
            Index of the current step.
        file_path : pathlib.Path or None
            File to save to. If None, a new file is created.

        Returns
        -------
        pathlib.Path
            Path of the file used for saving.
        """
        pos_x = step_data["pos_x"]
        pos_y = step_data["pos_y"]
        pos_z = step_data["pos_z"]
        num_particles = len(pos_x)

        if file_path is None:
            file_path = self._create_new_file(num_particles, pos_x, pos_y, pos_z)
        else:
            self._append_step(file_path, pos_x, pos_y, pos_z)

        return file_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_new_file(self, num_particles, pos_x, pos_y, pos_z):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{self.device}_{timestamp}.h5"
        file_path = self.output_path / filename

        with h5py.File(file_path, "w") as h5:
            for axis in ["pos_x", "pos_y", "pos_z"]:
                h5.create_dataset(
                    axis,
                    shape=(1, num_particles),
                    maxshape=(None, num_particles),
                    dtype="f4",
                    compression="gzip",
                )

            h5["pos_x"][0, :] = np.array(pos_x, dtype="f4")
            h5["pos_y"][0, :] = np.array(pos_y, dtype="f4")
            h5["pos_z"][0, :] = np.array(pos_z, dtype="f4")

        return file_path

    def _append_step(self, file_path, pos_x, pos_y, pos_z):
        with h5py.File(file_path, "a") as h5:
            for axis in ["pos_x", "pos_y", "pos_z"]:
                dataset = h5[axis]
                dataset.resize(dataset.shape[0] + 1, axis=0)

            h5["pos_x"][-1, :] = np.array(pos_x, dtype="f4")
            h5["pos_y"][-1, :] = np.array(pos_y, dtype="f4")
            h5["pos_z"][-1, :] = np.array(pos_z, dtype="f4")
