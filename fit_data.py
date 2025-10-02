import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch
from visualizations import *



def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    return parser

def fit_mesh(mesh_src, mesh_tgt, args, output_dir=None):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    loss_history = []
    
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()
        loss_history.append(loss_vis)

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    # Save loss curve
    if output_dir:
        save_loss_curve(loss_history, os.path.join(output_dir, f"mesh_loss_curve_{args.max_iter}iter.png"))
    else:
        save_loss_curve(loss_history, f"mesh_loss_curve_{args.max_iter}iter.png")
    print('Done!')
    return loss_history


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args, output_dir=None):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    loss_history = []
    
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()
        loss_history.append(loss_vis)

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    # Save loss curve
    if output_dir:
        save_loss_curve(loss_history, os.path.join(output_dir, f"pointcloud_loss_curve_{args.max_iter}iter.png"))
    else:
        save_loss_curve(loss_history, f"pointcloud_loss_curve_{args.max_iter}iter.png")
    print('Done!')
    return loss_history


def fit_voxel(voxels_src, voxels_tgt, args, output_dir=None):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    loss_history = []
    
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()
        loss_history.append(loss_vis)

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    # Save loss curve
    if output_dir:
        save_loss_curve(loss_history, os.path.join(output_dir, f"voxel_loss_curve_{args.max_iter}iter.png"))
    else:
        save_loss_curve(loss_history, f"voxel_loss_curve_{args.max_iter}iter.png")
    print('Done!')
    return loss_history


def train_model(args):
    # Create output directory structure
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectory based on fitting type
    if args.type == "vox":
        problem_section_dir = os.path.join(output_dir, "1_1_voxel_fitting")
    elif args.type == "point":
        problem_section_dir = os.path.join(output_dir, "1_2_pointcloud_fitting")
    elif args.type == "mesh":
        problem_section_dir = os.path.join(output_dir, "1_3_mesh_fitting")
    else:
        problem_section_dir = os.path.join(output_dir, "misc_fitting")
    
    os.makedirs(problem_section_dir, exist_ok=True)
    print(f"Output directory: {problem_section_dir}")
    
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        loss_history = fit_voxel(voxels_src, voxels_tgt, args, problem_section_dir)
        
        # Visualization
        print("Creating visualizations...")
        create_voxel_visualization(voxels_src, os.path.join(problem_section_dir, "optimized_voxel_grid.png"))
        create_voxel_visualization(voxels_tgt, os.path.join(problem_section_dir, "ground_truth_voxel_grid.png"))
        create_comparison_visualization(voxels_src, voxels_tgt, os.path.join(problem_section_dir, "voxel_comparison.png"))
        create_rotating_comparison_gif(voxels_src, voxels_tgt, os.path.join(problem_section_dir, "voxel_rotating_comparison.gif"))
        print("Voxel visualizations saved!")


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        loss_history = fit_pointcloud(pointclouds_src, pointclouds_tgt, args, problem_section_dir)
        
        # Visualization
        print("Creating visualizations...")
        create_pointcloud_visualization(pointclouds_src, os.path.join(problem_section_dir, "optimized_pointcloud.png"), color='red')
        create_pointcloud_visualization(pointclouds_tgt, os.path.join(problem_section_dir, "ground_truth_pointcloud.png"), color='blue')
        create_rotating_pointcloud_gif(pointclouds_src, pointclouds_tgt, os.path.join(problem_section_dir, "pointcloud_rotating_comparison.gif"))
        print("Point cloud visualizations saved!")        
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        loss_history = fit_mesh(mesh_src, mesh_tgt, args, problem_section_dir)
        
        # Visualization
        print("Creating visualizations...")
        create_mesh_visualization(mesh_src, os.path.join(problem_section_dir, "optimized_mesh.png"), color='lightcoral')
        create_mesh_visualization(mesh_tgt, os.path.join(problem_section_dir, "ground_truth_mesh.png"), color='lightblue')
        create_rotating_mesh_gif(mesh_src, mesh_tgt, os.path.join(problem_section_dir, "mesh_rotating_comparison.gif"))
        print("Mesh visualizations saved!")        


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
