from datetime import datetime
import os
import shutil
import cv2
import numpy as np

COLOUR_MAP = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255)
    }

def setup_artefact_paths(script_path:str, script_name:str=None):
    if script_name is None:
        script_name = os.path.basename(script_path)
    script_dir = os.path.dirname(script_path)

    yaml_path = os.path.join(script_dir, 'configs', script_name.replace('.py', '.yaml'))
    store_path = os.path.join(script_dir, 'artefacts')
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
    return store_path, yaml_path

def setup_experiment(script_path:str, base_name:str=None, save_yaml:bool=True):
    store_path, yaml_path = setup_artefact_paths(script_path=script_path)
    if base_name is None:
        base_name = os.path.basename(script_path).replace('.py', '')
    experiment_name = f'{base_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    precomputed_path = os.path.join(store_path, 'precomputed')
    store_path = os.path.join(store_path, experiment_name)
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
    
    if save_yaml is True:
        shutil.copy2(yaml_path, store_path)

    return experiment_name, store_path, yaml_path, precomputed_path

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