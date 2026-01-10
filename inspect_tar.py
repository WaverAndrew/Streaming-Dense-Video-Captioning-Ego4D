import tarfile
import torch
import os

file_path = "streaming_dvc/data/dataset/egovlp_egonce"

print(f"Inspecting tar file: {file_path}")

try:
    with tarfile.open(file_path, "r") as tar:
        print("Tar file opened successfully.")
        members = tar.getmembers()
        print(f"Total members: {len(members)}")
        
        # Find first .pt file
        for member in members:
            if member.name.endswith(".pt"):
                print(f"Extracting {member.name}...")
                f = tar.extractfile(member)
                if f:
                    try:
                        data = torch.load(f, map_location='cpu')
                        print(f"Loaded {member.name}")
                        print(f"Type: {type(data)}")
                        if isinstance(data, torch.Tensor):
                            print(f"Shape: {data.shape}")
                    except Exception as e:
                        print(f"Load failed: {e}")
                break
            
except Exception as e:
    print(f"Tar inspection failed: {e}")
