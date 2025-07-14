from datetime import datetime
import os
import shutil
import cv2
import optuna
from typing import Dict, Sequence
import numpy as np

COLOUR_MAP = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255)
    }

def setup_artefact_paths(script_path:str, config_name:str=None):
    if config_name is None:
        config_name = os.path.basename(script_path)
        config_name = config_name.replace('.py', '.yaml')
    else:
        if not config_name.endswith(".yaml"):
            config_name = f'{config_name}.yaml'
    script_dir = os.path.dirname(script_path)

    yaml_path = os.path.join(script_dir, 'configs', config_name)
    store_path = os.path.join(script_dir, 'artefacts')
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
    return store_path, yaml_path

def setup_experiment(store_path:str, config_path:str):
    experiment_name = os.path.basename(config_path).split(".")[0]
    experiment_name = f'{experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    precomputed_path = os.path.join(store_path, 'precomputed')
    if not os.path.isdir(precomputed_path):
        os.mkdir(precomputed_path)
    
    store_path = os.path.join(store_path, experiment_name)
    if not os.path.isdir(store_path):
        os.mkdir(store_path)

    shutil.copy2(config_path, store_path)
    return experiment_name, store_path, precomputed_path

def setup_study(store_path:str, config_path:str):
    experiment_name = os.path.basename(config_path).split(".")[0]
    experiment_name = f'opt_{experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    store_path = os.path.join(store_path, experiment_name)
    if not os.path.isdir(store_path):
        os.mkdir(store_path)

    return experiment_name, store_path

def sample_hyperparams(trial:optuna.trial.Trial, hparams: Dict) -> Dict:
    opt_hparams = {}
    float_keys = set(hparams["float_keys"])
    hparams.pop("float_keys")
    log_domain_keys = set(hparams["log_domain_keys"])
    hparams.pop("log_domain_keys")
    sampled_hparams = {}

    for key, val in hparams.items():
        # Fixed hyper-parameter
        if not isinstance(val, list):
            opt_hparams[key] = val
        else:
            # Float hyper-parameter
            if key in float_keys:
                sample_log = True if key in log_domain_keys else False
                sampled_val = trial.suggest_float(key, low=hparams[key][0], high=hparams[key][1], log=sample_log)
            # Categorical hyper-parameter
            else:
                sampled_val = trial.suggest_categorical(key, hparams[key])
            
            opt_hparams[key] = sampled_val
            if isinstance(sampled_val, list):
                str_repr = ""
                for val in sampled_val:
                    str_repr += f"{val}, "
                sampled_val = f"[{str_repr[:-2]}]"
            sampled_hparams[key] = sampled_val
    return opt_hparams, sampled_hparams

def load_and_resize_png(path:str, cell_size:int, keep_alpha:bool=False) -> np.ndarray:
    """
    Load the PNG image from 'path' and resize it to (cell_size, cell_size).
    Returns a 3-channel BGR image (uint8).
    If the PNG has an alpha channel, either discard or handle it as desired.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # read 4 channels if present
    if img is None:
        raise ValueError(f"Failed to load image from {path}")
    
    # If the PNG has 4 channels, drop the alpha channel for simplicity.
    if img.shape[2] == 4 and not keep_alpha:
        img = img[:, :, :3]
    
    # Resize to match cell_size
    resized = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
    return resized

def overlay_with_alpha(background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    """
    Blend a BGRA overlay onto a BGR or BGRA background at (x, y).
    If overlay has an alpha channel, use it to blend with whatever is in 'background'.
    Modifies 'background' in-place.
    """
    # Dimensions of the overlay
    h, w = overlay.shape[:2]

    # Slice the region of interest (ROI) from the background
    roi = background[y:y+h, x:x+w]

    if overlay.shape[2] == 4:
        # Separate the alpha channel from the BGR channels
        alpha = overlay[..., 3] / 255.0
        alpha_inv = 1.0 - alpha

        for c in range(3):
            roi[..., c] = (alpha * overlay[..., c] + alpha_inv * roi[..., c]).astype(np.uint8)
    else:
        # If overlay has no alpha channel, just overwrite
        roi[:] = overlay