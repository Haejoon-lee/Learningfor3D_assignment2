# 🔬 Hyperparameter Experiment Guide

## Modified Training Code
- ✅ **Checkpoint naming**: Now includes hyperparameters in filename
- ✅ **Args saved**: All training arguments saved in checkpoint
- ✅ **Evaluation updated**: Can load hyperparameter-specific checkpoints

## 📊 Experiment Commands

### 1. w_smooth Variation (Mesh Model)
```bash
# Train different smoothness weights
python train_model.py --type mesh --load_feat --w_smooth 0.1 --n_points 1000 --max_iter 2000
python train_model.py --type mesh --load_feat --w_smooth 0.5 --n_points 1000 --max_iter 2000  
python train_model.py --type mesh --load_feat --w_smooth 1.0 --n_points 1000 --max_iter 2000
python train_model.py --type mesh --load_feat --w_smooth 2.0 --n_points 1000 --max_iter 2000

# Evaluate each
python eval_model.py --type mesh --load_checkpoint --load_feat --w_smooth 0.1 --n_points 1000
python eval_model.py --type mesh --load_checkpoint --load_feat --w_smooth 0.5 --n_points 1000
python eval_model.py --type mesh --load_checkpoint --load_feat --w_smooth 1.0 --n_points 1000
python eval_model.py --type mesh --load_checkpoint --load_feat --w_smooth 2.0 --n_points 1000
```

### 2. n_points Variation (Point Model)
```bash
# Train different point densities
python train_model.py --type point --load_feat --n_points 500 --max_iter 2000
python train_model.py --type point --load_feat --n_points 1000 --max_iter 2000
python train_model.py --type point --load_feat --n_points 2000 --max_iter 2000
python train_model.py --type point --load_feat --n_points 5000 --max_iter 2000

# Evaluate each
python eval_model.py --type point --load_checkpoint --load_feat --n_points 500
python eval_model.py --type point --load_checkpoint --load_feat --n_points 1000
python eval_model.py --type point --load_checkpoint --load_feat --n_points 2000
python eval_model.py --type point --load_checkpoint --load_feat --n_points 5000
```

### 3. n_points Variation (Mesh Model - affects sampling)
```bash
# Train different sampling densities for mesh
python train_model.py --type mesh --load_feat --w_smooth 0.1 --n_points 500 --max_iter 2000
python train_model.py --type mesh --load_feat --w_smooth 0.1 --n_points 1000 --max_iter 2000
python train_model.py --type mesh --load_feat --w_smooth 0.1 --n_points 2000 --max_iter 2000

# Evaluate each
python eval_model.py --type mesh --load_checkpoint --load_feat --w_smooth 0.1 --n_points 500
python eval_model.py --type mesh --load_checkpoint --load_feat --w_smooth 0.1 --n_points 1000
python eval_model.py --type mesh --load_checkpoint --load_feat --w_smooth 0.1 --n_points 2000
```

## 📁 Generated Files

### Checkpoint Files
- `checkpoint_mesh_ws0.1_np1000.pth` - Mesh model with w_smooth=0.1, n_points=1000
- `checkpoint_mesh_ws1.0_np1000.pth` - Mesh model with w_smooth=1.0, n_points=1000
- `checkpoint_point_np500.pth` - Point model with n_points=500
- `checkpoint_point_np2000.pth` - Point model with n_points=2000

### Results Files
- `results/eval_mesh.png` - F1-score plot for mesh evaluation
- `results/eval_point.png` - F1-score plot for point evaluation
- `results/eval_mesh/` - Mesh visualizations
- `results/eval_point/` - Point cloud visualizations

## 🎯 Expected Results

### w_smooth Effects (Mesh):
- **0.1**: Spiky meshes, high shape accuracy, poor surface quality
- **1.0**: Balanced smoothness and shape accuracy
- **2.0**: Very smooth meshes, potential shape loss

### n_points Effects:
- **500**: Faster training, less detailed loss computation
- **2000**: More detailed loss, slower training
- **5000**: Very detailed loss, much slower training

## 📊 Analysis
Compare the F1-scores at different thresholds:
- **Strict thresholds (0.01-0.02)**: Surface quality
- **Lenient thresholds (0.04-0.05)**: Overall shape accuracy
