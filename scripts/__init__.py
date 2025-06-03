import os
import json
from PIL import Image
import numpy as np

def save_rgb_image(data_dir, filename, rgb_array):
    """Saves an RGB image."""
    img = Image.fromarray(rgb_array)
    img.save(os.path.join(data_dir, filename))

def save_depth_image(data_dir, filename, depth_array):
    """Saves a depth image (normalized to 0-255 for visualization/storage)."""
    # Normalize depth for 8-bit PNG storage (adjust as needed for higher precision)
    depth_normalized = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array) + 1e-6)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    img = Image.fromarray(depth_uint8, 'L') # 'L' for grayscale
    img.save(os.path.join(data_dir, filename))

def save_json_data(data_dir, filename, data):
    """Saves data to a JSON file."""
    with open(os.path.join(data_dir, filename), 'w') as f:
        json.dump(data, f, indent=4)

def create_episode_dir(base_dataset_path, episode_id):
    """Creates a new directory for an episode."""
    episode_dir = os.path.join(base_dataset_path, f"episode_{episode_id:05d}")
    os.makedirs(episode_dir, exist_ok=True)
    return episode_dir