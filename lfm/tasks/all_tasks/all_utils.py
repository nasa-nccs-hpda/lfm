import os
import shutil

def prepare_output_dir(output_dir):
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
        if len(contents) > 0:
            print(f"\n{'='*60}")
            print(f"⚠️  Output directory exists and is not empty:")
            print(f"    {output_dir}")
            print(f"    Contents: {len(contents)} items")
            print(f"{'='*60}")

            response = input("Clear this directory? (yes/no): ").strip().lower()

            if response in ['yes', 'y']:
                print(f"🗑️  Clearing {output_dir}...")
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                print(f"✅ Directory cleared and recreated")
                return True
            elif response in ['no', 'n']:
                print("❌ Training cancelled. Please backup your files and try again.")
                return False
            else:
                print("Invalid response. Training cancelled.")
                return False
        else:
            print(f"✅ Output directory exists and is empty: {output_dir}")
            return True
    else:
        print(f"✅ Creating new output directory: {output_dir}")
        os.makedirs(output_dir)
        return True
