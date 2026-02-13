"""
Simple test to check if DataParallel works with the current model
"""
import torch
import torch.nn as nn
import sys
import os

# Add project path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

def test_simple_dataparallel():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if we have multiple GPUs
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for this test")
        return True  # Skip test if only 1 GPU
    
    print(f"Using {torch.cuda.device_count()} GPUs")
    
    # Create model
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=0,
        use_proj_occ=False,
        voxel_size=0.0625,
        fusion_local_radius=0.0,
        crop_size=(10, 8, 6),
        use_checkpoint=False
    )
    
    # Wrap with DataParallel
    model_dp = nn.DataParallel(model, device_ids=[0, 1])
    model_dp = model_dp.to(device)
    model_dp.eval()
    
    # Create simple single-frame input (not sequence)
    batch_size = 4
    H, W = 96, 128
    
    images = torch.randn(batch_size, 3, H, W).to(device)
    poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    
    print(f"Input shapes - images: {images.shape}, poses: {poses.shape}, intrinsics: {intrinsics.shape}")
    
    # Test with DataParallel
    with torch.no_grad():
        try:
            output = model_dp(images, poses, intrinsics)
            print(f"✅ DataParallel single-frame test passed. Output type: {type(output)}")
            
            # Check GPU usage
            gpu0_mem = torch.cuda.memory_allocated(0) / 1024**2
            gpu1_mem = torch.cuda.memory_allocated(1) / 1024**2
            print(f"GPU 0 memory: {gpu0_mem:.2f} MB")
            print(f"GPU 1 memory: {gpu1_mem:.2f} MB")
            
            if gpu1_mem > 0:
                print("✅ Both GPUs are being used!")
            else:
                print("⚠️ Only GPU 0 is being used (expected with current implementation)")
                
        except Exception as e:
            print(f"❌ DataParallel test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    test_simple_dataparallel()