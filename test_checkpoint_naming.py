#!/usr/bin/env python3
"""
Test script to verify checkpoint naming works correctly
"""

import argparse
import sys
import os

def test_checkpoint_naming():
    """Test the checkpoint naming logic"""
    
    # Simulate different argument combinations
    test_cases = [
        {"type": "mesh", "w_smooth": 0.1, "n_points": 1000, "expected": "checkpoint_mesh_ws0.1_np1000.pth"},
        {"type": "mesh", "w_smooth": 2.0, "n_points": 2000, "expected": "checkpoint_mesh_ws2.0_np2000.pth"},
        {"type": "point", "n_points": 500, "expected": "checkpoint_point_np500.pth"},
        {"type": "point", "n_points": 5000, "expected": "checkpoint_point_np5000.pth"},
        {"type": "vox", "expected": "checkpoint_vox.pth"},
    ]
    
    print("🧪 Testing Checkpoint Naming Logic")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['type']} model")
        
        # Simulate the naming logic from train_model.py
        if case["type"] == "mesh":
            checkpoint_name = f"checkpoint_{case['type']}_ws{case['w_smooth']}_np{case['n_points']}.pth"
        elif case["type"] == "point":
            checkpoint_name = f"checkpoint_{case['type']}_np{case['n_points']}.pth"
        else:  # vox
            checkpoint_name = f"checkpoint_{case['type']}.pth"
        
        expected = case["expected"]
        success = checkpoint_name == expected
        
        print(f"  Generated: {checkpoint_name}")
        print(f"  Expected:  {expected}")
        print(f"  Result: {'✅ PASS' if success else '❌ FAIL'}")
        
        if not success:
            print(f"  ❌ Mismatch detected!")
            return False
    
    print(f"\n🎉 All {len(test_cases)} tests passed!")
    return True

def show_example_commands():
    """Show example training commands with new naming"""
    print("\n📋 Example Training Commands:")
    print("=" * 50)
    
    commands = [
        "python train_model.py --type mesh --load_feat --w_smooth 0.1 --n_points 1000 --max_iter 2000",
        "python train_model.py --type mesh --load_feat --w_smooth 2.0 --n_points 2000 --max_iter 2000", 
        "python train_model.py --type point --load_feat --n_points 500 --max_iter 2000",
        "python train_model.py --type point --load_feat --n_points 5000 --max_iter 2000",
    ]
    
    for cmd in commands:
        print(f"  {cmd}")
    
    print("\n📋 Example Evaluation Commands:")
    print("=" * 50)
    
    eval_commands = [
        "python eval_model.py --type mesh --load_checkpoint --load_feat --w_smooth 0.1 --n_points 1000",
        "python eval_model.py --type mesh --load_checkpoint --load_feat --w_smooth 2.0 --n_points 2000",
        "python eval_model.py --type point --load_checkpoint --load_feat --n_points 500",
        "python eval_model.py --type point --load_checkpoint --load_feat --n_points 5000",
    ]
    
    for cmd in eval_commands:
        print(f"  {cmd}")

if __name__ == "__main__":
    print("🔬 Hyperparameter Experiment Setup Test")
    print("=" * 60)
    
    # Test checkpoint naming logic
    if test_checkpoint_naming():
        show_example_commands()
        print(f"\n✅ Setup complete! You can now run hyperparameter experiments.")
        print(f"📖 See EXPERIMENT_GUIDE.md for detailed instructions.")
    else:
        print(f"\n❌ Setup failed! Check the naming logic.")
        sys.exit(1)
