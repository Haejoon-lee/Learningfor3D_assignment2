import torch
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt, fit=False):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
	# implement some loss for binary voxel grids
	
	if fit: 
		# For cross entropy, we need to ensure voxel_tgt is long tensor
		voxel_tgt = voxel_tgt.long()
		loss = torch.nn.functional.cross_entropy(voxel_src, voxel_tgt)
	else: 
		# For logits, use numerically stable BCEWithLogits
		voxel_tgt = voxel_tgt.float()
		if voxel_tgt.dim() > voxel_src.dim():
			voxel_tgt = voxel_tgt.squeeze()
		loss = torch.nn.functional.binary_cross_entropy_with_logits(voxel_src, voxel_tgt)

	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	
	# Find nearest neighbors: src -> tgt and tgt -> src
	src_to_tgt = knn_points(point_cloud_src, point_cloud_tgt)
	tgt_to_src = knn_points(point_cloud_tgt, point_cloud_src)
	
	# Chamfer distance: sum of distances from each point to its nearest neighbor
	# src_to_tgt.dists has shape [B, N_src, 1] - distances from each src point to nearest tgt point
	# tgt_to_src.dists has shape [B, N_tgt, 1] - distances from each tgt point to nearest src point
	loss_chamfer = torch.mean(src_to_tgt.dists.sum(1) + tgt_to_src.dists.sum(1))
	
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian