import torch
import gzip
import pickle
import os

file_path = "streaming_dvc/data/dataset/egovlp_egonce"

print(f"Inspecting {file_path}...")

# Check magic bytes
with open(file_path, 'rb') as f:
    magic = f.read(4)
    print(f"Magic bytes: {magic}")

# Try direct torch load
try:
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    print("Loaded with torch.load (direct).")
    print(f"Type: {type(data)}")
    if isinstance(data, torch.Tensor):
        print(f"Shape: {data.shape}")
    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:5]}")
        first_val = next(iter(data.values()))
        if hasattr(first_val, 'shape'):
            print(f"First value shape: {first_val.shape}")
except Exception as e:
    print(f"Direct torch load failed: {e}")
    
    # Try gzip
    try:
        with gzip.open(file_path, 'rb') as f:
             data = torch.load(f, map_location='cpu')
             print("Loaded with torch.load (gzip).")
             print(f"Type: {type(data)}")
             if isinstance(data, torch.Tensor):
                 print(f"Shape: {data.shape}")
    except Exception as e2:
        print(f"Gzip torch load failed: {e2}")
