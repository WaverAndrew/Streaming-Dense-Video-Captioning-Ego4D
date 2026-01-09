
import sys
import os
import torch
from transformers import T5Tokenizer

# Add current directory to path so we can import from vidchapters_test
sys.path.append(os.path.join(os.getcwd(), "vidchapters_test"))

try:
    from model.vid2seq import Vid2Seq, _get_tokenizer
    print("Successfully imported Vid2Seq")
except ImportError as e:
    print(f"Failed to import Vid2Seq: {e}")
    sys.exit(1)

def test_initialization():
    print("Testing Vid2Seq initialization...")
    # The weights require t5-base (d_ff=3072) not t5-v1_1-base (d_ff=2048)
    t5_path = "t5-base"
    
    try:
        # Initialize tokenizer first as it's needed for model resizing
        tokenizer = _get_tokenizer(t5_path, num_bins=100)
        print(f"Tokenizer initialized from {t5_path}")
        
        # Initialize model
        model = Vid2Seq(
            t5_path=t5_path,
            tokenizer=tokenizer,
            num_bins=100
        )
        print("Vid2Seq model initialized successfully")

        # Load weights
        weights_path = "/Users/andre/Desktop/Coding/streaming_dvc/weights/vid2seq_htmchaptersyoucook.pth"
        if os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}...")
            checkpoint = torch.load(weights_path, map_location="cpu")
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            
            # Handle potential prefix issues or strict loading
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
            # print(f"Missing keys: {msg.missing_keys}")
        else:
            print(f"Weights file not found at {weights_path}")
        
        # Basic forward pass check (dummy data)
        # B=1, T=10, D=768 (default embed_dim)
        video = torch.randn(1, 10, 768) 
        # B=1, L=10
        input_ids = torch.randint(0, 100, (1, 10))
        attention_mask = torch.ones(1, 10)
        input_tokenized = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        output_ids = torch.randint(0, 100, (1, 5))
        output_mask = torch.ones(1, 5)
        output_tokenized = {'input_ids': output_ids, 'attention_mask': output_mask}
        
        print("Running dummy forward pass...")
        outputs, _ = model(video, input_tokenized, output_tokenized)
        print(f"Forward pass successful. Loss: {outputs['loss'].item()}")
        
    except Exception as e:
        print(f"Error during initialization or forward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_initialization()
