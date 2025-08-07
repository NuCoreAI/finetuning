from importlib.resources import files
from pathlib import Path
import os

def get_data_directory(parent:str, subdir:str) -> str:
    """
    Returns the path to a subdirectory within a parent directory.
    
    Args:
        parent (str): The parent directory.
        subdir (str): The subdirectory to access.
        
    Returns:
        str: The path to the specified subdirectory.
    """

    return str(files(parent).joinpath(subdir)) if subdir else str(files(parent))

def get_secrets_dir()-> str:
    """
    Returns the path to the secrets directory.
    
    Returns:
        str: The path to the secrets directory.
    """
    return get_data_directory("secrets", None)
