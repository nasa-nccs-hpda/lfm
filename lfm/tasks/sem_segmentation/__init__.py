# lfm/tasks/segmentation/__init__.py

import sys
import os
import subprocess

def _ensure_termcolor_installed():
    """
    Ensure termcolor is installed (required by DINOv3).
    Only installs if not already available.
    """
    try:
        import termcolor
        # Already installed, nothing to do
        return
    except ImportError:
        pass

    print("Installing termcolor (required by DINOv3)...")

    # Try standard pip install first
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "termcolor"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✓ Termcolor installed successfully")
        return
    except subprocess.CalledProcessError:
        pass

    # Fall back to local installation
    try:
        print("Standard install failed, attempting local installation...")
        home_dir = os.path.expanduser("~")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        target_path = os.path.join(
            home_dir, ".local", "lib", f"python{python_version}", "site-packages"
        )
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            f"--target={target_path}",
            "termcolor", "--force-reinstall", "--no-deps"
        ])
        print(f"✓ Termcolor installed to: {target_path}")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: Failed to install termcolor: {e}")
        print("DINOv3 may not work properly without termcolor.")


# Run installation check on import
_ensure_termcolor_installed()