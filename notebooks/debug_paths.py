import os
import sys

# Diagnostic script to run from notebooks/ directory
print(f"Current Working Directory: {os.getcwd()}")
print(f"Python Path: {sys.path}")

try:
    sys.path.append(os.path.abspath(".."))
    print(f"Added to path: {os.path.abspath('..')}")

    from src.utils.data_loader import get_project_root, load_config, load_raw

    print("Successfully imported load_raw from src.utils.data_loader")

    root = get_project_root()
    print(f"Project Root resolved to: {root}")

    config = load_config()
    print("Successfully loaded config")

    df = load_raw()
    print(f"Successfully loaded raw data. Shape: {df.shape}")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
