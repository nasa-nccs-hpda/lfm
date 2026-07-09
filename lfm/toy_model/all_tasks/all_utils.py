import os
import shutil
from datetime import datetime
from pathlib import Path

def prepare_output_dir(output_dir, del_previous=False):
    """
    Check if output directory exists and prompt user to clear it.

    Args:
        output_dir: Path to output directory

    Returns:
        bool: True if directory is ready to use, False if user cancelled
    """

    if os.path.exists(output_dir):
        # Check if directory has contents
        contents = os.listdir(output_dir)
        if len(contents) > 0 and del_previous:
            print(f"\n{'='*60}")
            print(f"⚠️  Output directory exists and is not empty:")
            print(f"    {output_dir}")
            print(f"    Contents: {len(contents)} items")
            print(f"{'='*60}")
            print(f"Clearing {output_dir}...")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            print(f"✅ Directory cleared and recreated")
            return True
        else:
            print(f"✅ Using existing output_dir with current contents: {output_dir}")
            return True
    else:
        print(f"✅ Creating new output directory: {output_dir}")
        os.makedirs(output_dir)
        return True

def create_timestamped_output_dir(base_dir):
    """
    Create a timestamped subdirectory under base_dir.

    Args:
        base_dir: Base directory path

    Returns:
        Path to the timestamped directory (e.g., base_dir/20260629_143052)
    """
    timestamp = datetime.now().strftime("date_%Y_%m_%d-time_%H_%M_%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using output subdir: {base_dir}/{timestamp}")
    return output_dir