import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
import os
import contextlib

from torch.utils.data import Dataset
import baccoemu

from DUMP.data.features_engineering import make_features, nonlin_pk, flatten_and_scale_features
from DUMP.data.constants import bacco_train_ranges, bacco_test_ranges, bacco_target_z, solver_dz
from DUMP.utils import find_solver_grid

def sample_cosmologies(
    n_samples: int, 
    random_seed: int, 
    train: bool = False, 
    test: bool = False
):

    assert train != test, "Must specify either train or test"

    if train:
        ranges = bacco_train_ranges
    elif test:
        ranges = bacco_test_ranges
    else:
        raise ValueError("Must specify either train or test")

    sampler = LatinHypercube(d=len(ranges), rng=np.random.default_rng(random_seed))
    lattice = sampler.random((n_samples))

    # Rescale samples to correct ranges
    keys = list(ranges.keys())
    params_dict = {}
    for i, key in enumerate(keys):
        low, high = ranges[key]
        params_dict[key] = low + (high - low) * lattice[:, i]

    return params_dict


class BaccoPk(Dataset):
    def __init__(
        self,
        features_list: list[str],
        target_z: np.ndarray,
        scalers: dict,
        cosmologies_file: str = "./data/cosmologies.csv",
    ):
        super().__init__()
        self.features_list = features_list
        self.solver_z = find_solver_grid(target_z, solver_dz)
        self.target_z = target_z
        self.scalers = scalers
        self.cosmologies = pd.read_csv(cosmologies_file)
        # Don't store emulator directly - use lazy initialization for multiprocessing
        self._bacco_emulator = None

    @property
    def bacco_emulator(self):
        """Lazy initialization of emulator, this is needed to make num_workers > 0 work"""
        if self._bacco_emulator is None:
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                self._bacco_emulator = baccoemu.Matter_powerspectrum()
        return self._bacco_emulator
        
    def __len__(self):
        return len(self.cosmologies)

    def cosmo_to_features(self, cosmo_params: dict[np.float64], z: np.ndarray):
        return make_features(self.bacco_emulator, self.features_list, cosmo_params, z)

    def cosmo_to_target(self, cosmo_params: dict[np.float64], z: np.ndarray):
        # Initial spectrum is 'known', decouple it for convenience
        target = nonlin_pk(self.bacco_emulator, cosmo_params, z) 
        return target[0], target[1:]

    def __getitem__(self, idx):
        cosmo_params = self.cosmologies.iloc[idx].to_dict()
        features_dict = self.cosmo_to_features(cosmo_params, self.solver_z)
        init_cond_raw, target_raw = self.cosmo_to_target(cosmo_params, self.target_z)

        # Scale using shared function
        features = flatten_and_scale_features(features_dict, self.scalers, self.features_list)
        init_cond = (init_cond_raw - self.scalers["target"]["mean"]) / self.scalers["target"]["std"]
        target = (target_raw - self.scalers["target"]["mean"]) / self.scalers["target"]["std"]

        # Convert to float32 numpy arrays for MPS compatibility
        features = features.astype(np.float32)
        init_cond = init_cond.astype(np.float32)
        target = target.astype(np.float32)

        # Convert cosmo_params to float32 (pandas uses float64 by default)
        cosmo_params = {k: np.float32(v) for k, v in cosmo_params.items()}

        return cosmo_params, features, init_cond, target

