"""
Training Loop for Streaming Dense Video Captioning.
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

from ..config.default_config import StreamingDVCConfig
from ..models.streaming_model import StreamingDenseVideoCaptioning
from ..data.ego4d_dataset import Ego4DDenseCapDataset


def train(config: StreamingDVCConfig):
    """
    Main training loop.
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Model
    model = StreamingDenseVideoCaptioning(config).to(device)
    model.train()
    
    # 2. Data
    dataset = Ego4DDenseCapDataset(
        config.data, 
        config.training,
        split="train"
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True,
        num_workers=0 # Avoid multiprocessing issues in simple script
    )
    
    # 3. Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # 4. Loop
    print(f"Starting training for {config.training.num_training_steps} steps...")
    
    step = 0
    epoch = 0
    
    while step < config.training.num_training_steps:
        epoch += 1
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            video = batch["video"].to(device)
            input_ids = batch["input_ids"].to(device)
            # attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(
                video=video,
                text_tokens=input_ids
            )
            
            loss = outputs["loss"]
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if step >= config.training.num_training_steps:
                break
                
            if step % config.training.checkpoint_steps == 0:
                save_path = os.path.join(config.output_dir, f"checkpoint_{step}.pt")
                os.makedirs(config.output_dir, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")

    print("Training complete!")


if __name__ == "__main__":
    # Example usage
    from ..config.ego4d_config import get_ego4d_config
    
    config = get_ego4d_config(data_root="ego4d_pilot")
    train(config)
