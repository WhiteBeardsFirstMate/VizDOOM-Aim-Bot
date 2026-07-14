import os
import vizdoom as vzd

def ensure_dir(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_CONFIG_DIR)

OUTPUT_ROOT = ensure_dir(os.path.join(PROJECT_ROOT, "outputs_w-o_shaping"))
BASELINE_DIR = ensure_dir(os.path.join(OUTPUT_ROOT, "baseline"))
NOISY_DIR = ensure_dir(os.path.join(OUTPUT_ROOT, "noisy"))
A2C_DIR = ensure_dir(os.path.join(OUTPUT_ROOT, "a2c"))
COMP_DIR = ensure_dir(os.path.join(OUTPUT_ROOT, "comparison"))

CONFIG_FILE_PATH = os.path.join(vzd.scenarios_path, "defend_the_line.cfg")
