# lfm/tasks/segmentation/__init__.py

import sys
import os
import subprocess


def _ensure_termcolor_installed():
    """
    Ensure termcolor is installed (required by DINOv3).
    Only installs if not already available.
    """
    # First, ensure the target path is in sys.path
    home_dir = os.path.expanduser("~")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    target_path = os.path.join(
        home_dir, ".local", "lib", f"python{python_version}", "site-packages"
    )

    if target_path not in sys.path:
        sys.path.insert(0, target_path)

    try:
        import termcolor

        return  # Already available
    except ImportError:
        pass

    print("Installing termcolor (required by DINOv3)...")

    # Try standard pip install first
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "termcolor"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("✓ Termcolor installed successfully")

        # Clear import cache for termcolor
        if "termcolor" in sys.modules:
            del sys.modules["termcolor"]

        return
    except subprocess.CalledProcessError:
        pass

    # Fall back to local installation
    try:
        print("Standard install failed, attempting local installation...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"--target={target_path}",
                "termcolor",
                "--force-reinstall",
                "--no-deps",
            ]
        )
        print(f"✓ Termcolor installed to: {target_path}")

        # Clear import cache
        if "termcolor" in sys.modules:
            del sys.modules["termcolor"]

    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: Failed to install termcolor: {e}")
        print("DINOv3 may not work properly without termcolor.")


_ensure_termcolor_installed()
