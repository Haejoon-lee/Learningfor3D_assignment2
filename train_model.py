import argparse
import time

import dataset_location
import losses
import torch
from model import SingleViewto3D
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from r2n2_custom import R2N2


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--max_iter", default=100000, type=int)  # 10000 for actual training, 3000 seems to be enough with load_feat
    parser.add_argument("--batch_size", default=32, type=int) # Fix for every models
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh"], type=str
    )
    parser.add_argument("--n_points", default=1000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true')
    return parser


def preprocess(feed_dict, args):
    images = feed_dict["images"].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)


def calculate_loss(predictions, ground_truth, args):
    if args.type == "vox":
        loss = losses.voxel_loss(predictions, ground_truth)
    elif args.type == "point":
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == "mesh":
        # Use a conservative number of points to reduce degeneracy issues
        n_mesh_points = min(args.n_points, 1000)

        def per_mesh_total_face_area(meshes, eps: float = 1e-12) -> torch.Tensor:
            # Compute per-face areas, then sum per-mesh without launching any sampling ops
            if meshes.isempty():
                return torch.zeros((len(meshes),), device=meshes.device())
            verts = meshes.verts_packed()               # (F_verts, 3)
            faces = meshes.faces_packed()               # (F, 3)
            if faces.numel() == 0:
                return torch.zeros((len(meshes),), device=meshes.device())
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            face_areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), ord=2, dim=1)
            faces_per_mesh = meshes.num_faces_per_mesh()
            # Split and sum per mesh
            splits = torch.split(face_areas, faces_per_mesh.tolist())
            sums = torch.stack([s.sum() if s.numel() > 0 else torch.tensor(0.0, device=face_areas.device) for s in splits])
            # Clamp tiny negatives to 0 due to numeric noise
            sums = torch.clamp(sums, min=0.0)
            return sums

        # Pre-check GT and predicted mesh areas; avoid sampling if any zero-area mesh exists
        gt_area_sums = per_mesh_total_face_area(ground_truth)
        if (gt_area_sums <= 1e-12).any():
            print("Warning: GT mesh has zero total face area in batch; using only smoothness loss")
            loss_smooth = losses.smoothness_loss(predictions)
            return args.w_smooth * loss_smooth

        pred_area_sums = per_mesh_total_face_area(predictions)
        if (pred_area_sums <= 1e-12).any():
            print("Warning: Predicted mesh has zero total face area; using only smoothness loss")
            loss_smooth = losses.smoothness_loss(predictions)
            return args.w_smooth * loss_smooth

        # Safe to sample now
        sample_trg = sample_points_from_meshes(ground_truth, n_mesh_points)
        sample_pred = sample_points_from_meshes(predictions, n_mesh_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth
    return loss


def train_model(args):
    # Set return_voxels based on training type
    return_voxels = (args.type == "vox")
    
    r2n2_dataset = R2N2(
        "train",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=return_voxels,
        return_feats=args.load_feat,
    )

    # Disable pin_memory for mesh training to avoid degenerate mesh errors in background threads
    pin_memory = (args.type != "mesh")
    
    # For mesh training, keep batch size small to avoid per-batch degeneracy from any single bad mesh
    effective_batch_size = args.batch_size if args.type != "mesh" else min(args.batch_size, 4)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=effective_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=pin_memory,
        drop_last=True,
        shuffle=True,
    )
    train_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")

    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)
        
        read_start_time = time.time()

        # Handle case where we run out of data due to skipping too many batches
        try:
            feed_dict = next(train_loader)
        except StopIteration:
            print("Data loader exhausted, restarting...")
            train_loader = iter(loader)
            feed_dict = next(train_loader)

        try:
            images_gt, ground_truth_3d = preprocess(feed_dict, args)
            read_time = time.time() - read_start_time

            prediction_3d = model(images_gt, args)

            loss = calculate_loss(prediction_3d, ground_truth_3d, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_time = time.time() - start_time
            iter_time = time.time() - iter_start_time

            loss_vis = loss.cpu().item()

            if (step % args.save_freq) == 0 and step > 0:
                print(f"Saving checkpoint at step {step}")
                # Create checkpoint filename with hyperparameters
                if args.type == "mesh":
                    checkpoint_name = f"checkpoint_{args.type}_ws{args.w_smooth}_np{args.n_points}.pth"
                elif args.type == "point":
                    checkpoint_name = f"checkpoint_{args.type}_np{args.n_points}.pth"
                else:  # vox
                    checkpoint_name = f"checkpoint_{args.type}.pth"
                
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),  # Save all arguments for reference
                    },
                    checkpoint_name,
                )

            print(
                "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f"
                % (step, args.max_iter, total_time, read_time, iter_time, loss_vis)
            )

        except RuntimeError as e:
            if "invalid multinomial distribution" in str(e) or "device-side assert triggered" in str(e):
                print(f"Skipping degenerate batch at step {step}")
                continue  # Skip this batch and move to next iteration
            else:
                raise e  # Re-raise other errors

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
