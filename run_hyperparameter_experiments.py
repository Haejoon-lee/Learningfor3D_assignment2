#!/usr/bin/env python3
"""
Hyperparameter Experiment Runner
Runs training and evaluation for different hyperparameter combinations
"""

import subprocess
import os
import time

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        print(f"✅ Success! Duration: {duration:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ Failed after {duration:.1f}s")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🔬 Starting Hyperparameter Experiments")
    print("This will run multiple training and evaluation experiments")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    experiments = []
    
    # 1. w_smooth experiments for mesh model
    print("\n📊 Experiment 1: w_smooth variation (Mesh Model)")
    w_smooth_values = [0.1, 0.5, 1.0, 2.0]
    for w_smooth in w_smooth_values:
        # Training
        train_cmd = f"python train_model.py --type mesh --load_feat --w_smooth {w_smooth} --n_points 1000 --max_iter 2000 --save_freq 500"
        experiments.append((train_cmd, f"Train mesh with w_smooth={w_smooth}"))
        
        # Evaluation
        eval_cmd = f"python eval_model.py --type mesh --load_checkpoint --load_feat --w_smooth {w_smooth} --n_points 1000"
        experiments.append((eval_cmd, f"Eval mesh with w_smooth={w_smooth}"))
    
    # 2. n_points experiments for point model
    print("\n📊 Experiment 2: n_points variation (Point Model)")
    n_points_values = [500, 1000, 2000, 5000]
    for n_points in n_points_values:
        # Training
        train_cmd = f"python train_model.py --type point --load_feat --n_points {n_points} --max_iter 2000 --save_freq 500"
        experiments.append((train_cmd, f"Train point with n_points={n_points}"))
        
        # Evaluation
        eval_cmd = f"python eval_model.py --type point --load_checkpoint --load_feat --n_points {n_points}"
        experiments.append((eval_cmd, f"Eval point with n_points={n_points}"))
    
    # 3. n_points experiments for mesh model (affects sampling)
    print("\n📊 Experiment 3: n_points variation (Mesh Model)")
    mesh_n_points_values = [500, 1000, 2000]
    for n_points in mesh_n_points_values:
        # Training
        train_cmd = f"python train_model.py --type mesh --load_feat --w_smooth 0.1 --n_points {n_points} --max_iter 2000 --save_freq 500"
        experiments.append((train_cmd, f"Train mesh with n_points={n_points}"))
        
        # Evaluation
        eval_cmd = f"python eval_model.py --type mesh --load_checkpoint --load_feat --w_smooth 0.1 --n_points {n_points}"
        experiments.append((eval_cmd, f"Eval mesh with n_points={n_points}"))
    
    # Run experiments
    successful = 0
    failed = 0
    
    for i, (cmd, description) in enumerate(experiments, 1):
        print(f"\n🔄 Progress: {i}/{len(experiments)}")
        success = run_command(cmd, description)
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📈 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(experiments)}")
    
    if failed == 0:
        print(f"\n🎉 All experiments completed successfully!")
    else:
        print(f"\n⚠️  {failed} experiments failed. Check the logs above.")
    
    print(f"\n📁 Check the following for results:")
    print(f"   - Checkpoint files: checkpoint_*.pth")
    print(f"   - Evaluation plots: results/eval_*.png")
    print(f"   - Visualizations: results/eval_*/")

if __name__ == "__main__":
    main()
