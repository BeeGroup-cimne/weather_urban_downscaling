#!/usr/bin/env python3
import os
import tarfile
import subprocess
import shutil

# Config
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_FILE = os.path.join(ROOT_DIR, "dist", "weather_urban_downscaling_server_bundle.tar.gz")
ARCHIVE_ROOT_NAME = "weather_urban_downscaling"

# Filter config
EXCLUDE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.DS_Store', '.pyc'}
EXCLUDE_DIRS = {'.git', '.idea', '.vscode', '__pycache__', '.ipynb_checkpoints', 'node_modules', 'wandb'}
EXCLUDE_FILES = {'ablation_results.md', '.env', 'docker-compose.override.yml'}

# Heavy directories to exclude content from but keep structure (via .gitkeep or empty dir)
# We will manually add .gitkeep if they exist in git, but git ls-files handles that.
# We just need to filter out the *content* of these dirs if it's not tracked or if we want to explicitly exclude.
# Actually, the logic "include what git tracks" + "extra files" - "heavies" is best.

def is_excluded(path):
    # normalize path relative to root
    rel_path = os.path.relpath(path, ROOT_DIR)
    
    # Check strict file exclusions
    if os.path.basename(path) in EXCLUDE_FILES:
        return True
    
    # Check extensions
    _, ext = os.path.splitext(path)
    if ext.lower() in EXCLUDE_EXTENSIONS:
        return True
        
    # Check directories in path
    parts = rel_path.split(os.sep)
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
            
    # Check specific heavy paths
    # processed_cache_zarr, archive, reports, future_projects are normally ignored by git anyway.
    # But data/ and experiments/ might have tracked files we want to exclude (like huge outputs).
    
    # Logic: Exclude contents of experiments/ except .gitkeep and models/.gitkeep etc?
    # git ls-files usually is the source of truth.
    # If a file is tracked, we usually want it, UNLESS it's a heavy artifact we committed by mistake.
    # The bash script logic was:
    # exclude experiments/* unless .gitkeep
    # exclude data/* unless .gitkeep
    
    if rel_path.startswith("experiments/"):
        if os.path.basename(rel_path) == ".gitkeep": return False
        if os.path.basename(rel_path) == "README.md": return False
        return True # Exclude everything else in experiments
        
    if rel_path.startswith("data/"):
        if os.path.basename(rel_path) == ".gitkeep": return False
        if os.path.basename(rel_path) == "README.md": return False
        # Specific overrides for config files if any
        return True # Exclude everything else in data

    return False

def get_git_files():
    try:
        # Get list of tracked files
        result = subprocess.run(
            ["git", "ls-files", "-z"], 
            cwd=ROOT_DIR, 
            stdout=subprocess.PIPE, 
            check=True
        )
        files = result.stdout.split(b'\0')
        return [f.decode('utf-8') for f in files if f]
    except Exception as e:
        print(f"⚠️ Git error: {e}")
        return []

def main():
    print(f"📦 Creating server bundle: {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    files_to_add = set()
    
    # 1. Add tracked files
    git_files = get_git_files()
    print(f"   🔍 Found {len(git_files)} tracked files")
    
    for f in git_files:
        full_path = os.path.join(ROOT_DIR, f)
        if not is_excluded(full_path):
            files_to_add.add(f)
            
    # 2. Add extra server files (even if untracked)
    extra_files = [
        "SERVER_README.md",
        "docker-compose.server-fullframe.yml",
        "scripts/run_server_fullframe.sh",
        "scripts/run_ablation_tiles_heatwave_server.sh",
        "scripts/run_stations_eval_ablation.sh",
        "scripts/consolidate_experiment1.py",
        "scripts/run_experiment3_fullframe_replica.sh",
        "scripts/consolidate_experiment3.py",
        "scripts/make_server_bundle.py",
        "config/gpu_server_config.py" # Just in case
    ]
    
    for f in extra_files:
        if os.path.exists(os.path.join(ROOT_DIR, f)):
            files_to_add.add(f)
        else:
            print(f"   ⚠️ Extra file not found: {f}")

    # 3. Create Tarball
    with tarfile.open(OUTPUT_FILE, "w:gz") as tar:
        for f in sorted(list(files_to_add)):
            full_path = os.path.join(ROOT_DIR, f)
            arcname = os.path.join(ARCHIVE_ROOT_NAME, f)
            try:
                tar.add(full_path, arcname=arcname)
                # print(f"   ➕ Added: {f}")
            except Exception as e:
                print(f"   ❌ Failed to add {f}: {e}")

        # 4. Ensure structure directories exist (empty dirs)
        # In tarfile, we can just add directory infos.
        structure_dirs = [
            "data/processed/era5land",
            "data/raw",
            "experiments/models",
            "experiments/logs",
            "experiments/figures"
        ]
        
        for d in structure_dirs:
            # Check if we already added a file that creates this dir
            # If not, add the dir explicitly
            # Simplest is just to add a TarInfo for the dir
            d_arc = os.path.join(ARCHIVE_ROOT_NAME, d)
            t = tarfile.TarInfo(d_arc)
            t.type = tarfile.DIRTYPE
            t.mode = 0o755
            tar.addfile(t)
            
    print(f"✅ Bundle created successfully w/ {len(files_to_add)} files.")
    print(f"   Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
