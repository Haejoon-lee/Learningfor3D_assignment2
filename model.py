from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class VoxModule(nn.Module):
    def __init__(self, device):
        super(VoxModule, self).__init__()
        self.layer0 = torch.nn.Sequential(
                torch.nn.Linear(512, 2048)
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
            # Sigmoid removed; decoder now outputs logits. Use BCEWithLogitsLoss in losses.py
        )

    def forward(self, feats):
        res = self.layer0(feats)
        # Reshape to 3D: 256 channels, 2x2x2 spatial dimensions
        res = res.view((-1, 256, 2, 2, 2))
        res = self.layer1(res)  # 4x4x4
        res = self.layer2(res)  # 8x8x8
        res = self.layer3(res)  # 16x16x16
        res = self.layer4(res)  # 32x32x32
        res = self.layer5(res)  # 32x32x32 with 1 channel
        return res.squeeze(1)  # Remove channel dimension: 32x32x32

class PointModule(nn.Module):
    def __init__(self, device, num_points):
        super(PointModule, self).__init__()
        self.num_points = num_points
        self.layer0 = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, self.num_points),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.num_points, self.num_points*3)
        )

    def forward(self, feats):
        res = self.layer0(feats)
        res = res.view(-1, self.num_points, 3)
        return res

class MeshModule(nn.Module):
    def __init__(self, device, shape):
        super(MeshModule, self).__init__()
        self.shape = shape
        self.layer0 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.shape * 3)
        ) 

    def forward(self, feats):
        res = self.layer0(feats)
        res = res.view(-1, self.shape*3)
        return res

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            self.decoder = VoxModule(device=self.device)
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            self.decoder = PointModule(device=self.device, num_points=self.n_point)
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            self.mesh_pred_template = ico_sphere(4, self.device)
            mesh_shape = self.mesh_pred_template.verts_packed().shape[0]
            self.decoder = MeshModule(device=self.device, shape=mesh_shape)

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            voxels_pred = self.decoder(encoded_feat)
            return voxels_pred

        elif args.type == "point":
            pointclouds_pred = self.decoder(encoded_feat)
            return pointclouds_pred

        elif args.type == "mesh":
            batch_size = encoded_feat.shape[0]
            deform_vertices_pred = self.decoder(encoded_feat)
            # Create batch-specific mesh from template
            mesh_pred = pytorch3d.structures.Meshes(
                self.mesh_pred_template.verts_list() * batch_size, 
                self.mesh_pred_template.faces_list() * batch_size
            )
            mesh_pred = mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return mesh_pred          

