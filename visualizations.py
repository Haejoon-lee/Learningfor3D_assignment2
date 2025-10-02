import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import os
from matplotlib.animation import FuncAnimation

def create_voxel_visualization(voxels, fname="voxel_result.gif", threshold=0.5):
    """
    Create animated visualization of voxel grid
    Args:
        voxels: tensor of shape (H, W, D) or (1, H, W, D)
        fname: output filename
        threshold: threshold for binary voxel visualization
    """
    # Ensure voxels is 3D
    if voxels.dim() == 4:
        voxels = voxels.squeeze(0)
    
    # Convert to numpy and apply threshold
    voxels_np = voxels.detach().cpu().numpy()
    voxels_binary = (voxels_np > threshold).astype(float)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find voxel coordinates
    coords = np.where(voxels_binary > 0)
    
    if len(coords[0]) > 0:
        # Create scatter plot
        ax.scatter(coords[0], coords[1], coords[2], c='red', alpha=0.6, s=20)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Voxel Grid Visualization - {fname}')
    
    # Set equal aspect ratio
    max_range = np.array([voxels_np.shape[0], voxels_np.shape[1], voxels_np.shape[2]]).max() / 2.0
    mid_x = voxels_np.shape[0] / 2.0
    mid_y = voxels_np.shape[1] / 2.0
    mid_z = voxels_np.shape[2] / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save static image
    plt.savefig(fname.replace('.gif', '.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_pointcloud_visualization(points, fname="pointcloud_result.gif", color='blue'):
    """
    Create visualization of point cloud
    Args:
        points: tensor of shape (N, 3) or (1, N, 3)
        fname: output filename
        color: color for points
    """
    # Ensure points is 2D
    if points.dim() == 3:
        points = points.squeeze(0)
    
    points_np = points.detach().cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], 
               c=color, alpha=0.6, s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Point Cloud Visualization - {fname}')
    
    # Set equal aspect ratio
    max_range = np.array([points_np[:, 0].max() - points_np[:, 0].min(),
                         points_np[:, 1].max() - points_np[:, 1].min(),
                         points_np[:, 2].max() - points_np[:, 2].min()]).max() / 2.0
    mid_x = (points_np[:, 0].max() + points_np[:, 0].min()) / 2.0
    mid_y = (points_np[:, 1].max() + points_np[:, 1].min()) / 2.0
    mid_z = (points_np[:, 2].max() + points_np[:, 2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save static image
    plt.savefig(fname.replace('.gif', '.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_mesh_visualization(mesh, fname="mesh_result.gif", color='lightblue'):
    """
    Create visualization of mesh
    Args:
        mesh: PyTorch3D mesh object
        fname: output filename
        color: color for mesh
    """
    # Extract vertices and faces
    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                    triangles=faces, color=color, alpha=0.8, edgecolor='black', linewidth=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Mesh Visualization - {fname}')
    
    # Set equal aspect ratio
    max_range = np.array([verts[:, 0].max() - verts[:, 0].min(),
                         verts[:, 1].max() - verts[:, 1].min(),
                         verts[:, 2].max() - verts[:, 2].min()]).max() / 2.0
    mid_x = (verts[:, 0].max() + verts[:, 0].min()) / 2.0
    mid_y = (verts[:, 1].max() + verts[:, 1].min()) / 2.0
    mid_z = (verts[:, 2].max() + verts[:, 2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save static image
    plt.savefig(fname.replace('.gif', '.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_comparison_visualization(voxels_src, voxels_tgt, fname="comparison_result.png"):
    """
    Create side-by-side comparison of source and target voxels
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': '3d'})
    
    # Process source voxels
    if voxels_src.dim() == 4:
        voxels_src = voxels_src.squeeze(0)
    voxels_src_np = voxels_src.detach().cpu().numpy()
    voxels_src_binary = (voxels_src_np > 0.5).astype(float)
    coords_src = np.where(voxels_src_binary > 0)
    
    # Process target voxels
    if voxels_tgt.dim() == 4:
        voxels_tgt = voxels_tgt.squeeze(0)
    voxels_tgt_np = voxels_tgt.detach().cpu().numpy()
    voxels_tgt_binary = (voxels_tgt_np > 0.5).astype(float)
    coords_tgt = np.where(voxels_tgt_binary > 0)
    
    # Plot source
    if len(coords_src[0]) > 0:
        ax1.scatter(coords_src[0], coords_src[1], coords_src[2], c='red', alpha=0.6, s=20)
    ax1.set_title('Optimized Voxel Grid')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot target
    if len(coords_tgt[0]) > 0:
        ax2.scatter(coords_tgt[0], coords_tgt[1], coords_tgt[2], c='blue', alpha=0.6, s=20)
    ax2.set_title('Ground Truth Voxel Grid')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

def save_loss_curve(losses, fname="loss_curve.png"):
    """
    Save loss curve visualization
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

def create_rotating_comparison_gif(voxels_src, voxels_tgt, fname="rotating_comparison.gif", threshold=0.5, n_frames=36):
    """
    Create rotating GIF comparison of source and target voxels
    """
    # Ensure voxels are 3D
    if voxels_src.dim() == 4:
        voxels_src = voxels_src.squeeze(0)
    if voxels_tgt.dim() == 4:
        voxels_tgt = voxels_tgt.squeeze(0)
    
    # Convert to numpy and apply threshold
    voxels_src_np = voxels_src.detach().cpu().numpy()
    voxels_tgt_np = voxels_tgt.detach().cpu().numpy()
    voxels_src_binary = (voxels_src_np > threshold).astype(float)
    voxels_tgt_binary = (voxels_tgt_np > threshold).astype(float)
    
    # Find voxel coordinates
    coords_src = np.where(voxels_src_binary > 0)
    coords_tgt = np.where(voxels_tgt_binary > 0)
    
    # Create frames
    frames = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        
        fig = plt.figure(figsize=(15, 6))
        
        # Source voxels (left)
        ax1 = fig.add_subplot(121, projection='3d')
        if len(coords_src[0]) > 0:
            ax1.scatter(coords_src[0], coords_src[1], coords_src[2], c='red', alpha=0.6, s=20)
        ax1.set_title('Optimized Voxel Grid')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Target voxels (right)
        ax2 = fig.add_subplot(122, projection='3d')
        if len(coords_tgt[0]) > 0:
            ax2.scatter(coords_tgt[0], coords_tgt[1], coords_tgt[2], c='blue', alpha=0.6, s=20)
        ax2.set_title('Ground Truth Voxel Grid')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Rotate both views
        for ax in [ax1, ax2]:
            ax.view_init(elev=20, azim=angle * 180 / np.pi)
            # Set equal aspect ratio
            max_range = max(voxels_src_np.shape) / 2.0
            mid_x = voxels_src_np.shape[0] / 2.0
            mid_y = voxels_src_np.shape[1] / 2.0
            mid_z = voxels_src_np.shape[2] / 2.0
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(fname, frames, duration=0.1)
    print(f"Rotating comparison GIF saved: {fname}")

def create_rotating_pointcloud_gif(points_src, points_tgt, fname="rotating_pointcloud.gif", n_frames=36):
    """
    Create rotating GIF comparison of source and target point clouds
    """
    # Ensure points are 2D
    if points_src.dim() == 3:
        points_src = points_src.squeeze(0)
    if points_tgt.dim() == 3:
        points_tgt = points_tgt.squeeze(0)
    
    points_src_np = points_src.detach().cpu().numpy()
    points_tgt_np = points_tgt.detach().cpu().numpy()
    
    # Create frames
    frames = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        
        fig = plt.figure(figsize=(15, 6))
        
        # Source points (left)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(points_src_np[:, 0], points_src_np[:, 1], points_src_np[:, 2], 
                   c='red', alpha=0.6, s=1)
        ax1.set_title('Optimized Point Cloud')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Target points (right)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(points_tgt_np[:, 0], points_tgt_np[:, 1], points_tgt_np[:, 2], 
                   c='blue', alpha=0.6, s=1)
        ax2.set_title('Ground Truth Point Cloud')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Rotate both views
        for ax in [ax1, ax2]:
            ax.view_init(elev=20, azim=angle * 180 / np.pi)
            # Set equal aspect ratio
            max_range = np.array([points_src_np[:, 0].max() - points_src_np[:, 0].min(),
                                 points_src_np[:, 1].max() - points_src_np[:, 1].min(),
                                 points_src_np[:, 2].max() - points_src_np[:, 2].min()]).max() / 2.0
            mid_x = (points_src_np[:, 0].max() + points_src_np[:, 0].min()) / 2.0
            mid_y = (points_src_np[:, 1].max() + points_src_np[:, 1].min()) / 2.0
            mid_z = (points_src_np[:, 2].max() + points_src_np[:, 2].min()) / 2.0
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(fname, frames, duration=0.1)
    print(f"Rotating point cloud GIF saved: {fname}")

def create_rotating_mesh_gif(mesh_src, mesh_tgt, fname="rotating_mesh.gif", n_frames=36):
    """
    Create rotating GIF comparison of source and target meshes
    """
    # Extract vertices and faces
    verts_src = mesh_src.verts_packed().detach().cpu().numpy()
    faces_src = mesh_src.faces_packed().detach().cpu().numpy()
    verts_tgt = mesh_tgt.verts_packed().detach().cpu().numpy()
    faces_tgt = mesh_tgt.faces_packed().detach().cpu().numpy()
    
    # Create frames
    frames = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        
        fig = plt.figure(figsize=(15, 6))
        
        # Source mesh (left)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_trisurf(verts_src[:, 0], verts_src[:, 1], verts_src[:, 2], 
                        triangles=faces_src, color='lightcoral', alpha=0.8, 
                        edgecolor='black', linewidth=0.1)
        ax1.set_title('Optimized Mesh')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Target mesh (right)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_trisurf(verts_tgt[:, 0], verts_tgt[:, 1], verts_tgt[:, 2], 
                        triangles=faces_tgt, color='lightblue', alpha=0.8, 
                        edgecolor='black', linewidth=0.1)
        ax2.set_title('Ground Truth Mesh')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Rotate both views
        for ax in [ax1, ax2]:
            ax.view_init(elev=20, azim=angle * 180 / np.pi)
            # Set equal aspect ratio
            max_range = np.array([verts_src[:, 0].max() - verts_src[:, 0].min(),
                                 verts_src[:, 1].max() - verts_src[:, 1].min(),
                                 verts_src[:, 2].max() - verts_src[:, 2].min()]).max() / 2.0
            mid_x = (verts_src[:, 0].max() + verts_src[:, 0].min()) / 2.0
            mid_y = (verts_src[:, 1].max() + verts_src[:, 1].min()) / 2.0
            mid_z = (verts_src[:, 2].max() + verts_src[:, 2].min()) / 2.0
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(fname, frames, duration=0.1)
    print(f"Rotating mesh GIF saved: {fname}")
