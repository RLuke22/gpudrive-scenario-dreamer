#!/usr/bin/env python3
"""
Debug script to test execution path incrementally.
Since imports work, this tests where the crash occurs during actual execution.
"""

import sys
import traceback
import os
from pathlib import Path

def test_step(step_name, func, *args, **kwargs):
    """Test a step and report success/failure."""
    print(f"\n{'='*60}")
    print(f"Testing: {step_name}")
    print(f"{'='*60}")
    try:
        result = func(*args, **kwargs)
        print(f"✓ Successfully completed: {step_name}")
        return True, result
    except Exception as e:
        print(f"✗ Failed at: {step_name}")
        print(f"Error: {e}")
        traceback.print_exc()
        return False, None
    except SystemExit:
        print(f"✗ SystemExit at: {step_name}")
        return False, None

def main():
    print("Starting incremental execution test...")
    print("This will help identify where the core dump occurs during execution.\n")
    
    # Step 1: Import modules
    print("Step 1: Importing modules...")
    try:
        import yaml
        from box import Box
        import torch
        from gpudrive.env.dataset import SceneDataLoader
        from gpudrive.env.env_puffer import PufferGPUDrive
        import pufferlib
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)
    
    # Step 2: Load config
    config_path = "baselines/ppo/config/ppo_base_puffer.yaml"
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)
    
    def load_config():
        with open(config_path, "r") as f:
            config = Box(yaml.safe_load(f))
        return pufferlib.namespace(**config)
    
    success, config = test_step("Load config", load_config)
    if not success:
        sys.exit(1)
    
    # Step 3: Setup device
    def setup_device():
        config["train"]["device"] = config["train"].get("device", "cpu")
        if torch.cuda.is_available():
            config["train"]["device"] = "cuda"
        return config
    
    success, config = test_step("Setup device", setup_device)
    if not success:
        sys.exit(1)
    
    # Step 4: Create dataloader
    def create_dataloader():
        return SceneDataLoader(
            root=config.data_dir,
            batch_size=config.environment.num_worlds,
            dataset_size=config.train.resample_dataset_size
            if config.train.resample_scenes
            else config.environment.k_unique_scenes,
            sample_with_replacement=config.train.sample_with_replacement,
            shuffle=config.train.shuffle_dataset,
            seed=config.train.seed,
        )
    
    success, train_loader = test_step("Create dataloader", create_dataloader)
    if not success:
        sys.exit(1)
    
    # Step 5: Create environment (THIS IS LIKELY WHERE IT CRASHES)
    def create_environment():
        print("\n  Creating PufferGPUDrive environment...")
        print("  This step initializes the C++ simulator.")
        print("  If crash occurs here, check:")
        print("    - SimManager initialization")
        print("    - Route observation tensor access")
        print("    - Component registration")
        
        vecenv = PufferGPUDrive(
            data_loader=train_loader,
            **config.environment,
            **config.train,
        )
        return vecenv
    
    success, vecenv = test_step("Create environment (PufferGPUDrive)", create_environment)
    if not success:
        print(f"\n{'!'*60}")
        print("CRASH DETECTED during environment creation!")
        print("This is likely where your core dump occurs.")
        print(f"{'!'*60}")
        print("\nPossible causes:")
        print("1. Route observation tensor not properly initialized")
        print("2. Route processing system accessing invalid memory")
        print("3. Component registration mismatch")
        print("\nTry commenting out route observation changes in:")
        print("  - src/bindings.cpp (line 150)")
        print("  - src/sim.cpp (routeProcessingSystem)")
        sys.exit(1)
    
    # Step 6: Test reset (if we get here)
    def test_reset():
        print("\n  Testing environment reset...")
        obs = vecenv.env.reset(vecenv.controlled_agent_mask)
        return obs
    
    success, obs = test_step("Environment reset", test_reset)
    if not success:
        print(f"\n{'!'*60}")
        print("CRASH DETECTED during environment reset!")
        print(f"{'!'*60}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("All execution steps successful!")
    print("The core dump might occur later during training.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

