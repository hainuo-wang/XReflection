import torch
import torch.hub
import os
import sys
sys.path.append("../..")
# Assuming ARCH_REGISTRY is accessible. If not, this import might need adjustment
# or the registry needs to be passed as an argument.
# For a file named model_zoo.py inside xreflection.models, this import should be fine
# if xreflection.utils.registry exists and ARCH_REGISTRY is defined there.
from xreflection.utils.registry import ARCH_REGISTRY

# Base URL for downloading checkpoints
BASE_CHECKPOINT_URL = "https://checkpoints.mingjia.li/"
CACHE_DIR = os.path.join(torch.hub.get_dir(), "xreflection_aux_checkpoints")
# Configuration for models available in the zoo
# Each entry defines how to instantiate the architecture and where to download weights.
# Users need to populate this dictionary with actual model details.
def _ensure_directory_exists(path):
    """Helper function to ensure a directory exists."""
    os.makedirs(path, exist_ok=True)

def _download_cached_file(file_url, cache_dir, filename, progress=True):
    """
    Downloads a file to a cache directory if not already present.
    Returns the local path to the cached file.
    """
    _ensure_directory_exists(cache_dir)
    dst_path = os.path.join(cache_dir, filename)

    if not os.path.exists(dst_path):
        print(f"Downloading {filename} to {dst_path} from {file_url}")
        torch.hub.download_url_to_file(file_url, dst_path, progress=progress)
    else:
        print(f"File {filename} found in cache: {dst_path}")
    return dst_path

def prepare_model_path(suffix_or_path, progress=True):
    """
    Prepares the local path for an auxiliary model file.
    If suffix_or_path is a URL suffix, downloads the file from BASE_CHECKPOINT_URL.
    If it's an existing local path, returns it.
    If None, returns None.
    """
    if suffix_or_path is None:
        return None
    
    if not isinstance(suffix_or_path, str):
        raise ValueError(f"Auxiliary model path/suffix must be a string or None, got {type(suffix_or_path)}")

    # Check if it's already a valid local file path
    if os.path.exists(suffix_or_path):
        print(f"Using local auxiliary file: {suffix_or_path}")
        return suffix_or_path

    # Assume it's a suffix to be downloaded from BASE_CHECKPOINT_URL
    # Ensure suffix_or_path does not start with '/' if it's a relative path on server
    clean_suffix = suffix_or_path.lstrip('/')
    file_url = f"{BASE_CHECKPOINT_URL.rstrip('/')}/{clean_suffix}"
    filename = os.path.basename(clean_suffix) # Takes the last part of the path
    
    return _download_cached_file(file_url, CACHE_DIR, filename, progress=progress)



if __name__ == "__main__":
    print(prepare_model_path("focal.pth"))