import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
import math
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, out_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title('Evaluation results')
    plt.savefig(out_path, bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        # Ensure voxels have the right shape: (B, H, W, D)
        if len(voxels_src.shape) == 5:  # (B, C, H, W, D)
            voxels_src = voxels_src.squeeze(1)  # Remove channel dimension
        elif len(voxels_src.shape) == 4:  # (B, H, W, D)
            pass  # Already correct shape
        else:
            raise ValueError(f"Unexpected voxel shape: {voxels_src.shape}")
        
        # Apply sigmoid to convert logits to probabilities
        voxels_src = torch.sigmoid(voxels_src)
        
        # Get dimensions from the batch dimension
        H, W, D = voxels_src.shape[1:]
        # Marching cubes on the first item in batch for evaluation
        vox_np = voxels_src.detach().cpu().squeeze(0).numpy()
        vertices_src, faces_src = mcubes.marching_cubes(vox_np, isovalue=0.5)

        # Handle empty meshes (when marching cubes finds no surface)
        if len(vertices_src) == 0 or len(faces_src) == 0:
            print("Warning: Empty mesh from marching cubes, using random points as fallback")
            pred_points = torch.rand(1, args.n_points, 3, device=voxels_src.device) * 32.0
        else:
            vertices_src = torch.tensor(vertices_src, dtype=torch.float32, device=voxels_src.device)
            faces_src = torch.tensor(faces_src.astype(int), dtype=torch.int64, device=voxels_src.device)
            mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
            try:
                pred_points = sample_points_from_meshes(mesh_src, args.n_points)
            except ValueError as e:
                if "empty" in str(e).lower():
                    print(f"Warning: Failed to sample from mesh: {e}, using random points as fallback")
                    pred_points = torch.rand(1, args.n_points, 3, device=voxels_src.device) * 32.0
                else:
                    raise
        # Ensure CPU for utils_vox transforms
        pred_points = pred_points.cpu()
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
        # Apply a rotation transform to align predicted voxels to gt mesh
        angle = -math.pi
        axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
        Rot = axis_angle_to_matrix(axis_angle)
        T_transform = Rotate(Rot)
        pred_points = T_transform.transform_points(pred_points)
        # re-center the predicted points
        pred_points = pred_points - pred_points.mean(1, keepdim=True)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    # Handle ground truth points based on evaluation type
    if args.type == "vox":
        # For voxel evaluation, we need mesh ground truth to sample points for comparison
        if mesh_gt is None:
            # If no mesh GT available, skip evaluation for this sample
            print("Warning: No mesh ground truth available for voxel evaluation")
            return {}
        gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
        gt_points = gt_points - gt_points.mean(1, keepdim=True)
    else:
        gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics



def evaluate_model(args):
    # For voxel evaluation, we need both voxel GT (for visualization) and mesh GT (for comparison)
    # We'll load voxel GT first, then create a separate dataset for mesh GT
    return_voxels = (args.type == "vox")
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=return_voxels, return_feats=args.load_feat)
    
    # For voxel evaluation, also create a mesh dataset for ground truth comparison
    if args.type == "vox":
        mesh_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=False, return_feats=args.load_feat)
        mesh_loader = torch.utils.data.DataLoader(
            mesh_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_batched_R2N2,
            pin_memory=True,
            drop_last=True,
        )
        mesh_loader_iter = iter(mesh_loader)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        # Create checkpoint filename with hyperparameters (same logic as training)
        if args.type == "mesh":
            checkpoint_name = f"checkpoint_{args.type}_ws{args.w_smooth}_np{args.n_points}.pth"
            # checkpoint_name = f"checkpoint_{args.type}.pth"

        elif args.type == "point":
            checkpoint_name = f"checkpoint_{args.type}_np{args.n_points}.pth"
            # checkpoint_name = f"checkpoint_{args.type}.pth"
        else:  # vox
            checkpoint_name = f"checkpoint_{args.type}.pth"

        print(f"Checkpoint name: {checkpoint_name}")
        # Derive experiment name and output directory from checkpoint name
        # checkpoint_mesh_ws0.5_np1000.pth -> eval_mesh_ws0.5_np1000
        exp_name = checkpoint_name.replace('checkpoint_', '').replace('.pth', '')
        results_dir = f"results/eval_{exp_name}"
        import os
        os.makedirs(results_dir, exist_ok=True)

        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded checkpoint: {checkpoint_name}")
        if 'step' in checkpoint:
            print(f"Checkpoint from step: {checkpoint['step']}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        # For voxel evaluation, get mesh ground truth from separate dataset
        if args.type == "vox":
            try:
                mesh_feed_dict = next(mesh_loader_iter)
            except StopIteration:
                mesh_loader_iter = iter(mesh_loader)
                mesh_feed_dict = next(mesh_loader_iter)
            images_gt, mesh_gt = preprocess(mesh_feed_dict, args)
        else:
            images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        # ============ Visualization & saving ============
        # if (step % args.vis_freq) == 0:

        if (step % args.vis_freq) == 0:
            try:
                import os
                from visualizations import (
                    create_voxel_visualization,
                    create_comparison_visualization,
                    create_pointcloud_visualization,
                    create_mesh_visualization,
                    create_rotating_pointcloud_gif,
                    create_rotating_mesh_gif,
                    create_rotating_comparison_gif,
                )

                if args.type == "vox":
                    os.makedirs(results_dir, exist_ok=True)
                    step_tag = f"{step:04d}"

                    # Always save input RGB from feed_dict (even when using precomputed features)
                    try:
                        img_t = feed_dict["images"][0]  # may be (1, H, W, 3) or (H, W, 3)
                        if img_t.dim() == 4:
                            img_t = img_t.squeeze(0)
                        img = img_t.detach().cpu().numpy()
                        if img.max() > 1.0:
                            img = img / 255.0
                        plt.imsave(f"{results_dir}/{step_tag}_input.png", img)
                    except Exception as e:
                        print(f"Warning: failed to save input RGB ({e})")

                    # Predicted voxels (first in batch)
                    # Apply sigmoid to convert logits to probabilities for visualization
                    vox_pred_logits = predictions[0].detach().cpu()
                    vox_pred = torch.sigmoid(vox_pred_logits)
                    
                    # Debug: Print statistics of predicted voxels
                    print(f"[Debug] Predicted voxels stats:")
                    print(f"  Logits - min: {vox_pred_logits.min():.3f}, max: {vox_pred_logits.max():.3f}, mean: {vox_pred_logits.mean():.3f}")
                    print(f"  Probs  - min: {vox_pred.min():.3f}, max: {vox_pred.max():.3f}, mean: {vox_pred.mean():.3f}")
                    print(f"  Voxels > 0.5: {(vox_pred > 0.5).sum().item()} / {vox_pred.numel()}")

                    # If GT voxels are available in feed_dict, use them directly
                    if "voxels" in feed_dict:
                        vox_gt_t = feed_dict["voxels"][0].detach().cpu().float()
                        create_comparison_visualization(vox_pred, vox_gt_t, f"{results_dir}/{step_tag}_vox_compare.png")
                        # Create rotating comparison GIF
                        create_rotating_comparison_gif(vox_pred, vox_gt_t, f"{results_dir}/{step_tag}_vox_compare.gif")
                    else:
                        # Fallback: only save pred visualization
                        create_voxel_visualization(vox_pred, f"{results_dir}/{step_tag}_vox_pred.png")

                elif args.type == "point":
                    os.makedirs(results_dir, exist_ok=True)
                    step_tag = f"{step:04d}"
                    # predictions: B x N x 3, mesh_gt: Meshes
                    try:
                        pred_pts = predictions[0].detach().cpu()
                        gt_pts = sample_points_from_meshes(mesh_gt, args.n_points)[0].detach().cpu()
                        create_pointcloud_visualization(pred_pts, f"{results_dir}/{step_tag}_pred.png", color='red')
                        create_pointcloud_visualization(gt_pts, f"{results_dir}/{step_tag}_gt.png", color='blue')
                        create_rotating_pointcloud_gif(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0), f"{results_dir}/{step_tag}_compare.gif")
                    except Exception as e:
                        print(f"Point vis failed: {e}")

                elif args.type == "mesh":
                    os.makedirs(results_dir, exist_ok=True)
                    step_tag = f"{step:04d}"
                    try:
                        create_mesh_visualization(predictions, f"{results_dir}/{step_tag}_pred.png")
                        create_mesh_visualization(mesh_gt, f"{results_dir}/{step_tag}_gt.png")
                        create_rotating_mesh_gif(predictions, mesh_gt, f"{results_dir}/{step_tag}_compare.gif")
                    except Exception as e:
                        print(f"Mesh vis failed: {e}")
            except Exception as vis_e:
                print(f"Visualization failed: {vis_e}")

        # TODO:
        # if (step % args.vis_freq) == 0:
        #     # visualization block
        #     #  rend = 
        #     plt.imsave(f'vis/{step}_{args.type}.png', rend)
      

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    # Save the F1 curve into the experiment-specific directory
    try:
        save_plot(thresholds, avg_f1_score,  f"{results_dir}/eval_{exp_name}.png")
    except Exception:
        # In case results_dir is not defined (e.g., no checkpoint loaded), fall back
        import os
        os.makedirs("results", exist_ok=True)
        save_plot(thresholds, avg_f1_score,  f"results/eval_{args.type}.png")
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
