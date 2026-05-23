import torch
from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch3d.ops import knn_points
import nvdiffrast.torch as dr
from utils.geometry import rotation_6d_to_matrix
from gsplat.cuda._wrapper import spherical_harmonics

logger = logging.getLogger()

def _num_sh_bases(degree: int):
    if degree == 0:
        return 1
    if degree == 1:
        return 4
    if degree == 2:
        return 9
    if degree == 3:
        return 16
    return 25

class XYZ_Encoder(nn.Module):
    encoder_type = "XYZ_Encoder"
    """Encode XYZ coordinates or directions to a vector."""

    def __init__(self, n_input_dims):
        super().__init__()
        self.n_input_dims = n_input_dims

    @property
    def n_output_dims(self) -> int:
        raise NotImplementedError

class SinusoidalEncoder(XYZ_Encoder):
    encoder_type = "SinusoidalEncoder"
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(
        self,
        n_input_dims: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        enable_identity: bool = True,
    ):
        super().__init__(n_input_dims)
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.enable_identity = enable_identity
        self.register_buffer(
            "scales", Tensor([2**i for i in range(min_deg, max_deg + 1)])
        )

    @property
    def n_output_dims(self) -> int:
        return (
            int(self.enable_identity) + (self.max_deg - self.min_deg + 1) * 2
        ) * self.n_input_dims

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., n_input_dims]
        Returns:
            encoded: [..., n_output_dims]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[..., None, :] * self.scales[:, None]),
            list(x.shape[:-1])
            + [(self.max_deg - self.min_deg + 1) * self.n_input_dims],
        )
        encoded = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
        if self.enable_identity:
            encoded = torch.cat([x] + [encoded], dim=-1)
        return encoded

class MLP(nn.Module):
    """A simple MLP with skip connections."""

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_layers: int = 3,
        hidden_dims: Optional[int] = 256,
        skip_connections: Optional[Tuple[int]] = [0],
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.n_output_dims = out_dims
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(in_dims, out_dims))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    layers.append(nn.Linear(in_dims, hidden_dims))
                elif i in skip_connections:
                    layers.append(nn.Linear(in_dims + hidden_dims, hidden_dims))
                else:
                    layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.Linear(hidden_dims, out_dims))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        input = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input], -1)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.relu(x)
        return x
    
class SkyModel(nn.Module):
    def __init__(
        self,
        class_name: str,
        n: int, 
        head_mlp_layer_width: int = 64,
        enable_appearance_embedding: bool = True,
        appearance_embedding_dim: int = 16,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.direction_encoding = SinusoidalEncoder(
            n_input_dims=3, min_deg=0, max_deg=6
        )
        self.direction_encoding.requires_grad_(False)
        
        self.enable_appearance_embedding = enable_appearance_embedding
        if self.enable_appearance_embedding:
            self.appearance_embedding_dim = appearance_embedding_dim
            self.appearance_embedding = nn.Embedding(n, appearance_embedding_dim, dtype=torch.float32)
            
        in_dims = self.direction_encoding.n_output_dims + appearance_embedding_dim \
            if self.enable_appearance_embedding else self.direction_encoding.n_output_dims
        self.sky_head = MLP(
            in_dims=in_dims,
            out_dims=3,
            num_layers=3,
            hidden_dims=head_mlp_layer_width,
            skip_connections=[1],
        )
        self.in_test_set = False
    
    def forward(self, image_infos):
        directions = image_infos["viewdirs"]
        self.device = directions.device
        prefix = directions.shape[:-1]
        
        dd = self.direction_encoding(directions.reshape(-1, 3)).to(self.device)
        if self.enable_appearance_embedding:
            # optionally add appearance embedding
            if "img_idx" in image_infos and not self.in_test_set:
                appearance_embedding = self.appearance_embedding(image_infos["img_idx"]).reshape(-1, self.appearance_embedding_dim)
            else:
                # use mean appearance embedding
                appearance_embedding = torch.ones(
                    (*dd.shape[:-1], self.appearance_embedding_dim),
                    device=dd.device,
                ) * self.appearance_embedding.weight.mean(dim=0)
            dd = torch.cat([dd, appearance_embedding], dim=-1)
        rgb_sky = self.sky_head(dd).to(self.device)
        rgb_sky = F.sigmoid(rgb_sky)
        return rgb_sky.reshape(prefix + (3,))
    
    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }

    def freeze_envmap(self, optimizer):
        self.base.requires_grad_(False)
        if optimizer is None:
            return
        for group in optimizer.param_groups:
            if group.get("name") == self.class_prefix + "all":
                envmap_param = group["params"][0]
                if envmap_param in optimizer.state:
                    del optimizer.state[envmap_param]
                break
        
class EnvLight(torch.nn.Module):

    def __init__(
        self,
        class_name: str,
        resolution=1024,
        device: torch.device = torch.device("cuda"),
        activation_name: str = None,
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.resolution = resolution
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )
        
        if activation_name is None:
            self.activation = None
        elif activation_name == 'exp':
            print("Using exp activation for envmap")
            self.activation = torch.exp
            self.base.data[:] = torch.log(self.base.data)
        elif activation_name == 'relu':
            print("Using relu activation for envmap")
            self.activation = F.relu
        
    def forward(self, image_infos):
        l = image_infos["viewdirs"]
        
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])

        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)

        if self.activation is not None:
            light = self.activation(light)

        return light

    def query_light(self, l):
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])
        
        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)
        
        if self.activation is not None:
            light = self.activation(light)
        
        return light

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
    
    def get_activated_base(self):
        base = self.base
        if self.activation is not None:
            base = self.activation(base)
        return base
    
    def export_vis(self):
        """
        Returns a (4H, 3W, 3) tensor that visualizes the cubemap in a cross layout:
               [2]
            [1][4][0][5]
               [3]
        """
        faces = self.base.detach()  # shape: (6, H, W, 3)
        if self.activation is not None:
            faces = self.activation(faces)
        H, W = faces.shape[1], faces.shape[2]

        # Create blank canvas: 3 rows, 4 columns
        canvas = torch.zeros((3 * H, 4 * W, 3), dtype=faces.dtype)

        # Face order mapping to layout
        face_layout = {
            0: (1 * H, 2 * W),  # +X (front)
            1: (1 * H, 0 * W),  # -X (back)
            2: (0 * H, 1 * W),  # +Z (top)
            3: (2 * H, 1 * W),  # -Z (bottom)
            4: (1 * H, 1 * W),  # +Y (left)
            5: (1 * H, 3 * W),  # -Y (right)
        }
        
        for face_id, (row, col) in face_layout.items():
            canvas[row:row + H, col:col + W] = faces[face_id]

        return canvas  # shape: (4H, 3W, 3)

    def meshgrid_lin(self, start: float, end: float, steps: int, 
                 device: torch.device = torch.device('cuda'), 
                 dtype: torch.dtype = torch.float32):
        """
        Creates two [steps x steps] grids X,Y linearly spaced in [start, end],
        with X varying along the horizontal axis and Y along the vertical.
        """
        coords = torch.linspace(start, end, steps, device=device, dtype=dtype)
        # use 'xy' indexing so that X has shape [steps, steps] with varies columns,
        # and Y varies rows
        X, Y = torch.meshgrid(coords, coords, indexing='xy')
        return X, Y

    def sample_light_dir(self, num_sample_rays=64):
        """
        Sample num_sample_rays direction from the cubemap light source.
        Args:
            num_sample_rays: number of rays to sample
        Returns:
            d: sampled directions, shape: [num_sample_rays, 3]
            p_dir: continuous PDF for MIS, shape: [num_sample_rays]
        """
        N = self.resolution
        # 1. Precompute solid angles for each face texel
        dA = (2.0 / N)**2
        X, Y = self.meshgrid_lin(-1+1/N, 1-1/N, N)  # in [-1,1]
        omega = dA / ((X**2 + Y**2 + 1.0)**1.5)    # shape [N,N]

        # 2. Compute raw weights
        L = self.base.max(dim=-1)             # [6, N, N]
        raw_weights = L * omega.unsqueeze(0)   # broadcast to [6,N,N]

        # 3. Normalize
        pdf = raw_weights / raw_weights.sum()  # [6,N,N]
        self._pdf = pdf

        # 4. Sample & map → direction
        idx = torch.multinomial(pdf.reshape(-1), num_sample_rays) # [S]
        f = idx // (N*N)        # face index ∈ [0,5], shape: [S]
        r = idx %  (N*N)        # remainder in [0, N*N-1], shape: [S]
        i = r  // N             # row index, shape: [S]
        j = r  %  N             # col index, shape: [S]
        u = ( j + 0.5 ) / N       # 0 at left, 1 at right, shape: [S]
        v = 1.0 - (i + 0.5) / N   # 0 at bottom, 1 at top, shape: [S]
        d = self.cubemap_uv_to_dir(f, u, v) # [S, 3]

        # 5. Evaluate continuous PDF for MIS:
        p_dir = self._pdf[f,i,j] / omega[i,j] # [S]
        
        return d, p_dir

    def cubemap_uv_to_dir(self,
        face_idx: torch.LongTensor,   # [S]
        u:       torch.Tensor,        # [S]
        v:       torch.Tensor         # [S]
    ) -> torch.Tensor:
        """
        Convert cubemap (face_idx, u, v) → 3D direction in world space.

        face_idx: which of the 6 faces, shape [S]
        u, v: normalized texel centers in [0,1], shape [S]
        returns: [S, 3] direction in world space
        """
        # map [0,1] → [-1,1]
        # note u will be the horizontal axis, v the vertical axis
        # uv origin is at left-bottom
        x = 2.0 * u - 1.0
        y = 2.0 * v - 1.0

        # prepare empty direction components
        shape = u.shape
        device = u.device
        dtype = u.dtype
        dx = torch.zeros(shape, device=device, dtype=dtype)
        dy = torch.zeros(shape, device=device, dtype=dtype)
        dz = torch.zeros(shape, device=device, dtype=dtype)

        # +X face (face 0): 
        m = face_idx == 0
        dx[m], dy[m], dz[m] =  1.0, -x[m], y[m]

        # -X face (face 1):  
        m = face_idx == 1
        dx[m], dy[m], dz[m] = -1.0, x[m], y[m]

        # +Z face (face 2):  
        m = face_idx == 2
        dx[m], dy[m], dz[m] = x[m], -y[m], 1.0

        # -Z face (face 3):  
        m = face_idx == 3
        dx[m], dy[m], dz[m] = x[m], y[m], -1.0

        # +Y face (face 4):  
        m = face_idx == 4
        dx[m], dy[m], dz[m] = x[m], 1.0, y[m]

        # -Y face (face 5):  
        m = face_idx == 5
        dx[m], dy[m], dz[m] = -x[m], -1.0, y[m]

        # stack and normalize
        dirs = torch.stack((dx, dy, dz), dim=-1)            # (...,3)
        dirs = F.normalize(dirs, dim=-1, eps=1e-6)

        return dirs

import imageio
import pyexr

def pixel_grid(width, height, center_x = 0.5, center_y = 0.5):
    y, x = torch.meshgrid(
            (torch.arange(0, height, dtype=torch.float32, device="cuda") + center_y) / height, 
            (torch.arange(0, width, dtype=torch.float32, device="cuda") + center_x) / width)
    return torch.stack((x, y), dim=-1)

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def latlong_to_cubemap(latlong_map, res, device='cuda'):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=device)
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device)
                                )
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return torch.nn.functional.avg_pool2d(cubemap.permute(0, 3, 1, 2), (2, 2)).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device=dout.device)
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=dout.device), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=dout.device)
                                   )
            v = safe_normalize(cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out        

import numpy as np
from . import renderutils as ru

def srgb_to_rgb(img):
    # f is LDR
    if isinstance(img, np.ndarray):
        img = np.where(img <= 0.04045, img / 12.92, np.power((np.maximum(img, 0.04045) + 0.055) / 1.055, 2.4))
        return img
    elif isinstance(img, torch.Tensor):
        img = torch.where(img <= 0.04045, img / 12.92, torch.pow((torch.max(img, torch.tensor(0.04045)) + 0.055) / 1.055, 2.4))
        return img
    else:
        raise TypeError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")

# adapted from IRGS: https://github.com/fudan-zvg/IRGS
class EnvLight_EQ(torch.nn.Module):

    def __init__(
        self,
        class_name: str,
        resolution=1024,
        device: torch.device = torch.device("cuda"),
        activation_name: str = None,
        min_res: int = 64,
        max_res: int = 1024,
        min_roughness: float = 0.08,
        max_roughness: float = 0.5,
        init_value: float = 0.5,
        prior_path: str = None,
        path: str=None,
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.resolution = resolution
        # self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        # self.base = torch.nn.Parameter(
        #     0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        # )
        self.prior_path = prior_path
        
        self.forward_mode = "pure_env"

        self.min_res = min_res # minimum resolution for mip-map
        self.max_res = max_res # maximum resolution for mip-map
        self.min_roughness = min_roughness # minimum roughness for mip-map
        self.max_roughness = max_roughness # maximum roughness for mip-map
        # self.base = torch.nn.Parameter(
        #     torch.full((resolution // 2, resolution, 3), init_value, dtype=torch.float32, device=self.device),
        #     requires_grad=True,
        # )
        if path is not None:
            latlong_img = self.load(path)
            # Resize latlong_img to match the envmap resolution
            target_h = self.resolution // 2
            target_w = self.resolution
            texcoord = pixel_grid(target_w, target_h)
            latlong_img = dr.texture(latlong_img[None, ...], texcoord[None, ...], filter_mode='linear')[0].clamp_min(1e-4)
            self.base = torch.nn.Parameter(latlong_img, requires_grad=True)
        else:
            self.base = torch.nn.Parameter(
                torch.full((resolution // 2, resolution, 3), init_value, dtype=torch.float32, device=self.device),
                requires_grad=True,
            )
        # opencv -> opengl
        self.init_transform = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        self.transform = self.init_transform.clone()
        self.base_mip = None
        
        self.activation_name = activation_name
        if activation_name == 'sigmoid':
            self.activation = torch.sigmoid
            def inverse_sigmoid(x):
                return torch.log(x/(1-x))
            self.inv_activation = inverse_sigmoid
            self.base.data[:] = inverse_sigmoid(self.base.data)
        elif activation_name == 'exp':
            self.activation = torch.exp
            self.inv_activation = torch.log
            self.base.data[:] = torch.log(self.base.data)
        elif activation_name == 'none' or activation_name is None:
            self.activation = lambda x: x
            self.inv_activation = lambda x: x
        else:
            raise NotImplementedError
        
        self.activated_prior_base = None
        if self.prior_path is not None:
            self.load_prior(self.prior_path)
    
    def get_activated_base(self):
        base = self.base
        if self.activation is not None:
            base = self.activation(base)
        
        # # debugging
        # valid_square = [420, 480, 50, 130]
        # # valid_square = [375,470, 50, 120]
        
        # resolution = self.base.shape[1]
        # valid_square = [int(v * (resolution / 512)) for v in valid_square]
        # mask = torch.zeros_like(self.base)
        # mask[valid_square[2]:valid_square[3], valid_square[0]:valid_square[1], :] = 1.0
        # mask = mask.to(torch.bool)
        # # base[mask] *= 10.0
        # base[mask] = 0.01
        
        return base.clamp_min(0.0)
    
    def get_base(self):
        # get non-activated base
        return self.base
    
    def update_pdf(self):
        with torch.no_grad():
            # Compute PDF
            Y = pixel_grid(self.base.shape[1], self.base.shape[0])[..., 1]
            self._pdf = torch.max(self.get_activated_base(), dim=-1)[0] * torch.sin(Y * np.pi) # Scale by sin(theta) for lat-long, https://cs184.eecs.berkeley.edu/sp18/article/25
            self._pdf = self._pdf / torch.sum(self._pdf)
    
    def get_pdf_differentiable(self):
        Y = pixel_grid(self.base.shape[1], self.base.shape[0])[..., 1]
        pdf = torch.max(self.get_activated_base(), dim=-1)[0] * torch.sin(Y * np.pi)
        pdf = pdf / torch.sum(pdf)
        return pdf
    
    def sample_light_directions(self, B, sample_num, training=False, pdf_differentiable=False):
        pdf_flat = self._pdf.reshape(-1)
        light_dir_idx = torch.multinomial(pdf_flat, B*sample_num, replacement=True)
        
        H, W = self._pdf.shape[:2]
        gx = ((light_dir_idx % W  + 0.5) / W) * 2 - 1
        gy = (light_dir_idx // W + 0.5) / H
        if training:
            gx = gx + (torch.rand_like(gx) - 0.5) / W * 2
            gy = gy + (torch.rand_like(gy) - 0.5) / H
        sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
        sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
        direction = torch.stack((
            sintheta*sinphi, 
            costheta, 
            -sintheta*cosphi
        ), dim=-1)
        
        if self.transform is not None:
            # inverse transform
            direction = direction @ self.transform
        direction = direction.reshape(B, sample_num, 3)
        
        probability = self.light_pdf(direction, pdf_differentiable)
        
        return direction, probability
    
    def light_pdf(self, direction, pdf_differentiable=False):
        if pdf_differentiable:
            pdf_flat = self.get_pdf_differentiable().reshape(-1)
        else:
            pdf_flat = self._pdf.reshape(-1)
        direction_flat = direction.reshape(-1, 3)
        if self.transform is not None:
            direction_flat = direction_flat @ self.transform.T
        H, W = self._pdf.shape[:2]
        
        u = (torch.atan2(direction_flat[..., 0], -direction_flat[..., 2]).nan_to_num() / (2.0 * torch.pi) + 0.5)
        v = torch.acos(direction_flat[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi
        
        u_idx = (u * W).clamp(0, W - 1).long()
        v_idx = (v * H).clamp(0, H - 1).long()
        light_dir_idx = u_idx + v_idx * W
        
        pdf_weight = H * W / (2.0 * torch.pi ** 2 * torch.sin(v * torch.pi).clamp_min(1e-6))
        probability = (torch.take_along_dim(pdf_flat, light_dir_idx, dim=0) * pdf_weight).reshape(*direction.shape[:2], 1)
        return probability
    
    # find brightest 0.1% area of the envmap, and return the center direction of that area as the main light direction
    def get_main_dir(self, percentage=0.001):
        with torch.no_grad():
            intensity = torch.max(self.get_activated_base(), dim=-1)[0]
            threshold = torch.quantile(intensity, 1.0 - percentage)
            mask = intensity >= threshold
            if mask.sum() == 0:
                print("Warning: no valid pixel found for main light direction, returning default up direction")
                return torch.tensor([0.0, 1.0, 0.0], device=self.device) # default to up direction if no valid pixel
            # 1. Get indices of the high-intensity pixels
            # mask is (H, W), indices will be (N, 2) where N is number of masked pixels
            indices = torch.nonzero(mask)
            h_idx = indices[:, 0].float()
            w_idx = indices[:, 1].float()
            
            H, W = intensity.shape[:2]

            # 2. Convert indices to UV coordinates (gx, gy)
            # Logic mirrored from sample_light_directions
            gx = ((w_idx + 0.5) / W) * 2 - 1
            gy = (h_idx + 0.5) / H
            
            # 3. Convert UV to Spherical and then Cartesian
            sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
            sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)
            
            directions = torch.stack((
                sintheta * sinphi, 
                costheta, 
                -sintheta * cosphi
            ), dim=-1)
            
            # 4. Apply the inverse transform if it exists
            if self.transform is not None:
                directions = directions @ self.transform

            # 5. Calculate the center direction
            # We take the mean of all direction vectors in the mask and normalize
            main_dir = torch.mean(directions, dim=0)
            main_dir = main_dir / torch.norm(main_dir)

            return main_dir
            

    def capture(self):
        state_dict = super().state_dict()
        return {
            "transform": self.transform,
            "state_dict": state_dict,
            "activation": self.activation_name
        }
    
    def restore(self, model_args):
        activation = model_args['activation']
        self.activation_name = activation
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'exp':
            self.activation = torch.exp
        elif activation == 'none':
            self.activation = lambda x: x
        else:
            raise NotImplementedError
        
        if 'transform' in model_args:
            self.init_transform = model_args['transform']
            self.transform = self.init_transform.clone()
        else:
            print("Warning: no transform found in model_args, using default transform")
        
        self.load_state_dict(model_args['state_dict'])
    
    def calibrate_direction(self, camera_dir):
        """
        We need to make the envmap centralized around the camera direction of the first frame,
        as the Diffusion Light envmap is estimated from the first frame.
        And also make it compatible with traditional envmap representation.
        """
        # first convert the camera_dir to opengl coordinate
        to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device=self.device)
        camera_dir_opengl = camera_dir @ to_opengl.T
        
        # next, compute the rotation matrix to align the camera direction with (0, 0, -1)
        target_dir = torch.tensor([0.0, 0.0, -1.0], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
        camera_dir_opengl = camera_dir_opengl / (camera_dir_opengl.norm() + 1e-8)
        target_dir = target_dir / (target_dir.norm() + 1e-8)

        v = torch.cross(camera_dir_opengl, target_dir)
        s = v.norm()
        c = torch.dot(camera_dir_opengl, target_dir)

        if s < 1e-6:
            if c > 0:
                rot = torch.eye(3, device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
            else:
                # 180 degree rotation around any orthogonal axis
                axis = torch.tensor([1.0, 0.0, 0.0], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
                if torch.allclose(camera_dir_opengl, axis):
                    axis = torch.tensor([0.0, 1.0, 0.0], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
                v = torch.cross(camera_dir_opengl, axis)
                v = v / (v.norm() + 1e-8)
                K = torch.tensor([[0, -v[2], v[1]],
                                  [v[2], 0, -v[0]],
                                  [-v[1], v[0], 0]], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
                rot = torch.eye(3, device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype) + 2 * K @ K
        else:
            K = torch.tensor([[0, -v[2], v[1]],
                              [v[2], 0, -v[0]],
                              [-v[1], v[0], 0]], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
            rot = torch.eye(3, device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype) + K + K @ K * ((1 - c) / (s ** 2 + 1e-8))

        to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        self.init_transform = rot @ to_opengl
        self.transform = self.init_transform.clone()
        
    
    def set_rotate(self, angle_degrees: float = 0, rotate_vertical: bool = False):
        theta = torch.tensor(angle_degrees, dtype=torch.float32, device=self.device) * (np.pi / 180.0)  # Convert degrees to radians
        
        if not rotate_vertical:
            rot_y = torch.tensor([
                [torch.cos(theta), 0, torch.sin(theta)],
                [0, 1, 0],
                [-torch.sin(theta), 0, torch.cos(theta)]
            ], device=self.device)

            self.transform = rot_y @ self.init_transform
        else:
            rot_x = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(theta), -torch.sin(theta)],
                [0, torch.sin(theta), torch.cos(theta)]
            ], device=self.device)

            self.transform = rot_x @ self.init_transform
    
    def build_mips(self, cutoff=0.99):
        """
        Build mip-maps for specular reflection based on cubemap.
        """
        self.base_mip = latlong_to_cubemap(self.get_activated_base(), [self.max_res, self.max_res], self.device)
        
        self.specular = [self.base_mip]
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)
    
    def get_mip(self, roughness):
        """
        Map roughness to mip level.
        """
        return torch.where(
            roughness < self.max_roughness, 
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2), 
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )
    
    # def apply_mask(self):
    #     # for debugging
    #     print("Debugging: applying mask on envmap")
        
    #     mask = torch.zeros_like(self.base)
    #     H, W = mask.shape[:2]
    #     mask[:, :W//3, :] = 1
    #     mask = mask.to(torch.bool)
        
    #     # self.base[~mask] *= 5.0
    #     # self.base[~mask] = 0.01
    #     # self.base[~mask] *= 10.0
        
    #     # self.base *= 10.0
        
    #     # self.base[self.base > 0.9] *= 10.0
        
    #     self.build_mips()
    #     self.update_pdf()
    
    def apply_mask_by_dirs(self, dirs):
        dirs = dirs.reshape(-1, 3)
        # convert dirs to uv coordinates, and mask out the pixels near these uv
        if self.transform is not None:
            dirs = dirs @ self.transform.T
        uv = torch.cat([
            (torch.atan2(dirs[..., :1], -dirs[..., 2:3]).nan_to_num() / (2.0 * torch.pi) + 0.5),
            torch.acos(dirs[..., 1:2].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi,
        ], dim=-1).clamp(0, 1)
        
        # Get the mask for bilinear interpolation pixels
        mask = self.get_bilinear_mask(uv)
        
        # Apply the mask to the base environment map
        with torch.no_grad():
            init_val = 0.5
            if self.activation_name == 'sigmoid':
                init_val = np.log(init_val / (1 - init_val))
            elif self.activation_name == 'exp':
                init_val = np.log(init_val)
            self.base[mask] = init_val
                
    
    def get_bilinear_mask(self, uv):
        """
        Get a mask for the 4 pixels used in bilinear interpolation for given UV coordinates.
        
        Args:
            uv: UV coordinates of shape [..., 2] in range [0, 1]
            
        Returns:
            mask: Boolean mask of shape [H, W] indicating which pixels are used for interpolation
        """
        H, W = self.base.shape[:2]
        
        # Convert UV to pixel coordinates
        u_pixel = uv[..., 0] * (W - 1)  # [0, W-1]
        v_pixel = uv[..., 1] * (H - 1)  # [0, H-1]
        
        # Get the four corner pixels for bilinear interpolation
        u_floor = torch.floor(u_pixel).long()
        u_ceil = torch.ceil(u_pixel).long()
        v_floor = torch.floor(v_pixel).long()
        v_ceil = torch.ceil(v_pixel).long()
        
        # Clamp to valid range
        u_floor = torch.clamp(u_floor, 0, W - 1)
        u_ceil = torch.clamp(u_ceil, 0, W - 1)
        v_floor = torch.clamp(v_floor, 0, H - 1)
        v_ceil = torch.clamp(v_ceil, 0, H - 1)
        
        # Create mask
        mask = torch.zeros(H, W, dtype=torch.bool, device=self.base.device)
        
        # Flatten coordinates for advanced indexing
        u_floor_flat = u_floor.flatten()
        u_ceil_flat = u_ceil.flatten()
        v_floor_flat = v_floor.flatten()
        v_ceil_flat = v_ceil.flatten()
        
        # Mark the 4 corner pixels for each UV coordinate
        # Top-left: (v_floor, u_floor)
        mask[v_floor_flat, u_floor_flat] = True
        # Top-right: (v_floor, u_ceil)
        mask[v_floor_flat, u_ceil_flat] = True
        # Bottom-left: (v_ceil, u_floor)
        mask[v_ceil_flat, u_floor_flat] = True
        # Bottom-right: (v_ceil, u_ceil)
        mask[v_ceil_flat, u_ceil_flat] = True
        
        return mask
    
    def forward(self, image_infos, roughness=None):
        l = image_infos["viewdirs"]
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # Reshape to [B, H, W, -1] if necessary
            l = l.reshape(1, 1, -1, l.shape[-1])
        if self.transform is not None:
            l = l @ self.transform.T
        if roughness is not None:
            roughness = roughness.reshape(1, 1, -1, 1)

        base = self.get_activated_base()
        if self.forward_mode == "diffuse":
            # Diffuse lighting
            light = dr.texture(self.diffuse[None, ...], l.contiguous(), filter_mode='linear', boundary_mode='cube')
        elif self.forward_mode == "pure_env":
            uv = torch.cat([
                (torch.atan2(l[..., :1], -l[..., 2:3]).nan_to_num() / (2.0 * torch.pi) + 0.5),
                torch.acos(l[..., 1:2].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi,
            ], dim=-1).clamp(0, 1)
            light = dr.texture(base[None, ...], uv, filter_mode='linear')
        else:
            # Specular lighting with mip-mapping
            miplevel = self.get_mip(roughness)
            light = dr.texture(
                self.specular[0][None, ...], 
                l,
                mip=list(m[None, ...] for m in self.specular[1:]), 
                mip_level_bias=miplevel[..., 0], 
                filter_mode='linear-mipmap-linear', 
                boundary_mode='cube'
            )
        light = light.view(*prefix, -1)
        
        # return self.activation(light).clamp(0, 3)
        return light

    def query_light(self, l, mode='pure_env', roughness=None):
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # Reshape to [B, H, W, -1] if necessary
            l = l.reshape(1, 1, -1, l.shape[-1])
        if self.transform is not None:
            l = l @ self.transform.T
        if roughness is not None:
            roughness = roughness.reshape(1, 1, -1, 1)

        base = self.get_activated_base()
        if self.forward_mode == "diffuse":
            # Diffuse lighting
            light = dr.texture(self.diffuse[None, ...], l.contiguous(), filter_mode='linear', boundary_mode='cube')
        elif self.forward_mode == "pure_env":
            uv = torch.cat([
                (torch.atan2(l[..., :1], -l[..., 2:3]).nan_to_num() / (2.0 * torch.pi) + 0.5),
                torch.acos(l[..., 1:2].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi,
            ], dim=-1).clamp(0, 1)
            light = dr.texture(base[None, ...], uv, filter_mode='linear')
        else:
            # Specular lighting with mip-mapping
            miplevel = self.get_mip(roughness)
            light = dr.texture(
                self.specular[0][None, ...], 
                l,
                mip=list(m[None, ...] for m in self.specular[1:]), 
                mip_level_bias=miplevel[..., 0], 
                filter_mode='linear-mipmap-linear', 
                boundary_mode='cube'
            )
        light = light.view(*prefix, -1)
        # return self.activation(light).clamp(0, 3)
        # return self.activation(light).clamp_min(0.0)
        return light
        
    def export_vis(self):
        base = self.get_activated_base().detach()
        return base

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
    
    def get_name(self):
        return self.class_prefix + "all"

    def freeze_envmap(self, optimizer):
        self.base.requires_grad_(False)
        if optimizer is None:
            return
        for group in optimizer.param_groups:
            if group.get("name") == self.class_prefix + "all":
                envmap_param = group["params"][0]
                if envmap_param in optimizer.state:
                    del optimizer.state[envmap_param]
                break

    def load(self, path):
        """
        Load an .hdr or .exr environment light map file and convert it to cubemap.
        """
        if path.endswith(".exr"):
            image = pyexr.open(path).get()[:, :, :3]
        elif path.endswith(".hdr"):  # #
            image = imageio.imread(path, format='HDR-FI')[:, :, :3]
        else:
            image = srgb_to_rgb(imageio.imread(path)[:, :, :3] / 255)
        
        # clip max: some exr or hdr have extremely high values
        finite_max_val = np.max(image[np.isfinite(image)])
        image = np.clip(image, 0, finite_max_val)
        
        image = torch.from_numpy(image).to(self.device)
        return image
    
    def load_prior(self, path):
        assert path is not None
        # load a new path and replace the current base data
        latlong_img = self.load(path)
        target_h = self.resolution // 2
        target_w = self.resolution
        texcoord = pixel_grid(target_w, target_h)
        latlong_img = dr.texture(latlong_img[None, ...], texcoord[None, ...], filter_mode='linear')[0].clamp_min(1e-4)
        self.activated_prior_base = latlong_img

    def get_prior(self):
        return self.activated_prior_base

    def load_full(self, path):
        assert path is not None
        # load a new path and replace the current base data
        latlong_img = self.load(path)
        target_h = self.resolution // 2
        target_w = self.resolution
        texcoord = pixel_grid(target_w, target_h)
        latlong_img = dr.texture(latlong_img[None, ...], texcoord[None, ...], filter_mode='linear')[0].clamp_min(1e-4)
        if self.activation_name == 'sigmoid' or self.activation_name == 'exp':
            latlong_img = self.inv_activation(latlong_img)
        with torch.no_grad():
            self.base.data[:] = latlong_img.data[:]

class EnvLight_SG(torch.nn.Module):

    def __init__(
        self,
        class_name: str,
        resolution=1024,
        device: torch.device = torch.device("cuda"),
        activation_name: str = None,
        min_res: int = 64,
        max_res: int = 1024,
        min_roughness: float = 0.08,
        max_roughness: float = 0.5,
        init_value: float = 0.5,
        prior_path: str = None,
        path: str = None,
        num_sgs: int = 32,
        init_sharpness: float = 16.0,
        min_sharpness: float = 1.0e-4,
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.resolution = resolution
        self.num_sgs = num_sgs
        self.prior_path = prior_path
        self.min_res = min_res
        self.max_res = max_res
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.min_sharpness = min_sharpness
        self.init_sharpness = init_sharpness
        self.activation_name = activation_name

        sg_dirs = torch.randn(num_sgs, 3, device=self.device, dtype=torch.float32)
        sg_dirs = F.normalize(sg_dirs, dim=-1)
        self.sg_dirs = torch.nn.Parameter(sg_dirs, requires_grad=True)

        init_amp = torch.full((num_sgs, 3), init_value, device=self.device, dtype=torch.float32)
        init_sharp = torch.full((num_sgs, 1), init_sharpness, device=self.device, dtype=torch.float32)
        self.sg_amplitudes = torch.nn.Parameter(self._softplus_inv(init_amp), requires_grad=True)
        self.sg_sharpness = torch.nn.Parameter(self._softplus_inv(init_sharp), requires_grad=True)

        self.base = self.sg_amplitudes

        self.init_transform = torch.tensor(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            dtype=torch.float32,
            device=self.device,
        )
        self.transform = self.init_transform.clone()
        self.base_mip = None

        self.activated_prior_base = None
        if self.prior_path is not None:
            self.load_prior(self.prior_path)

        if path is not None:
            self.init_from_envmap(path, num_sgs=self.num_sgs, sharpness=self.init_sharpness)

    def _softplus_inv(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=1.0e-8)
        return torch.log(torch.expm1(x))

    def _positive(self, x: torch.Tensor, min_value: float = 0.0) -> torch.Tensor:
        return F.softplus(x) + min_value

    def _get_sg_params(self):
        dirs = F.normalize(self.sg_dirs, dim=-1)
        sharpness = self._positive(self.sg_sharpness, self.min_sharpness).squeeze(-1)
        amplitudes = self._positive(self.sg_amplitudes, 0.0)
        return dirs, sharpness, amplitudes

    def _latlong_grid(self, height: int, width: int):
        u = (torch.arange(width, device=self.device, dtype=torch.float32) + 0.5) / width
        v = (torch.arange(height, device=self.device, dtype=torch.float32) + 0.5) / height
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        phi = (uu * 2.0 - 1.0) * torch.pi
        theta = vv * torch.pi
        sin_theta = torch.sin(theta)
        dirs = torch.stack(
            (
                sin_theta * torch.sin(phi),
                torch.cos(theta),
                -sin_theta * torch.cos(phi),
            ),
            dim=-1,
        )
        texcoord = torch.stack((uu, vv), dim=-1)
        return dirs, vv, texcoord

    def _eval_sg(self, directions: torch.Tensor):
        dirs, sharpness, amplitudes = self._get_sg_params()
        directions = F.normalize(directions, dim=-1)
        dot = torch.sum(directions[..., None, :] * dirs, dim=-1).clamp(-1.0, 1.0)
        weights = torch.exp(sharpness * (dot - 1.0))
        rgb = (weights[..., None] * amplitudes).sum(dim=-2)
        return rgb

    def get_activated_base(self):
        height = self.resolution // 2
        width = self.resolution
        dirs, _, _ = self._latlong_grid(height, width)
        envmap = self._eval_sg(dirs)
        return envmap.clamp_min(0.0)

    def get_base(self):
        return self.get_activated_base()

    def update_pdf(self):
        with torch.no_grad():
            envmap = self.get_activated_base()
            _, vv, _ = self._latlong_grid(envmap.shape[0], envmap.shape[1])
            self._pdf = torch.max(envmap, dim=-1)[0] * torch.sin(vv * torch.pi)
            self._pdf = self._pdf / torch.sum(self._pdf)

    def sample_light_directions(self, B, sample_num, training=False):
        if not hasattr(self, "_pdf"):
            self.update_pdf()
        pdf_flat = self._pdf.reshape(-1)
        light_dir_idx = torch.multinomial(pdf_flat, B * sample_num, replacement=True)

        H, W = self._pdf.shape[:2]
        gx = ((light_dir_idx % W + 0.5) / W) * 2 - 1
        gy = (light_dir_idx // W + 0.5) / H
        if training:
            gx = gx + (torch.rand_like(gx) - 0.5) / W * 2
            gy = gy + (torch.rand_like(gy) - 0.5) / H
        sintheta, costheta = torch.sin(gy * torch.pi), torch.cos(gy * torch.pi)
        sinphi, cosphi = torch.sin(gx * torch.pi), torch.cos(gx * torch.pi)
        direction = torch.stack((
            sintheta * sinphi,
            costheta,
            -sintheta * cosphi
        ), dim=-1)

        if self.transform is not None:
            direction = direction @ self.transform
        direction = direction.reshape(B, sample_num, 3)

        probability = self.light_pdf(direction)
        return direction, probability

    def light_pdf(self, direction):
        if not hasattr(self, "_pdf"):
            self.update_pdf()
        pdf_flat = self._pdf.reshape(-1)
        direction_flat = direction.reshape(-1, 3)
        if self.transform is not None:
            direction_flat = direction_flat @ self.transform.T
        H, W = self._pdf.shape[:2]

        u = (torch.atan2(direction_flat[..., 0], -direction_flat[..., 2]).nan_to_num() / (2.0 * torch.pi) + 0.5)
        v = torch.acos(direction_flat[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi

        u_idx = (u * W).clamp(0, W - 1).long()
        v_idx = (v * H).clamp(0, H - 1).long()
        light_dir_idx = u_idx + v_idx * W

        pdf_weight = H * W / (2.0 * torch.pi ** 2 * torch.sin(v * torch.pi).clamp_min(1e-6))
        probability = (torch.take_along_dim(pdf_flat, light_dir_idx, dim=0) * pdf_weight).reshape(*direction.shape[:2], 1)
        return probability

    def capture(self):
        state_dict = super().state_dict()
        return {
            "transform": self.transform,
            "state_dict": state_dict,
            "activation": self.activation_name
        }

    def restore(self, model_args):
        activation = model_args.get('activation', None)
        self.activation_name = activation
        if 'transform' in model_args:
            self.init_transform = model_args['transform']
            self.transform = self.init_transform.clone()
        else:
            print("Warning: no transform found in model_args, using default transform")
        self.load_state_dict(model_args['state_dict'])

    def calibrate_direction(self, camera_dir):
        to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device=self.device)
        camera_dir_opengl = camera_dir @ to_opengl.T

        target_dir = torch.tensor([0.0, 0.0, -1.0], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
        camera_dir_opengl = camera_dir_opengl / (camera_dir_opengl.norm() + 1e-8)
        target_dir = target_dir / (target_dir.norm() + 1e-8)

        v = torch.cross(camera_dir_opengl, target_dir)
        s = v.norm()
        c = torch.dot(camera_dir_opengl, target_dir)

        if s < 1e-6:
            if c > 0:
                rot = torch.eye(3, device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
            else:
                axis = torch.tensor([1.0, 0.0, 0.0], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
                if torch.allclose(camera_dir_opengl, axis):
                    axis = torch.tensor([0.0, 1.0, 0.0], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
                v = torch.cross(camera_dir_opengl, axis)
                v = v / (v.norm() + 1e-8)
                K = torch.tensor([[0, -v[2], v[1]],
                                  [v[2], 0, -v[0]],
                                  [-v[1], v[0], 0]], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
                rot = torch.eye(3, device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype) + 2 * K @ K
        else:
            K = torch.tensor([[0, -v[2], v[1]],
                              [v[2], 0, -v[0]],
                              [-v[1], v[0], 0]], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
            rot = torch.eye(3, device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype) + K + K @ K * ((1 - c) / (s ** 2 + 1e-8))

        to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device=self.device)
        self.init_transform = rot @ to_opengl
        self.transform = self.init_transform.clone()

    def set_rotate(self, angle_degrees: float = 0, rotate_vertical: bool = False):
        theta = torch.tensor(angle_degrees, dtype=torch.float32, device=self.device) * (torch.pi / 180.0)

        if not rotate_vertical:
            rot_y = torch.tensor([
                [torch.cos(theta), 0, torch.sin(theta)],
                [0, 1, 0],
                [-torch.sin(theta), 0, torch.cos(theta)]
            ], device=self.device)
            self.transform = rot_y @ self.init_transform
        else:
            rot_x = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(theta), -torch.sin(theta)],
                [0, torch.sin(theta), torch.cos(theta)]
            ], device=self.device)
            self.transform = rot_x @ self.init_transform

    def build_mips(self, cutoff=0.99):
        self.base_mip = latlong_to_cubemap(self.get_activated_base(), [self.max_res, self.max_res], self.device)
        self.specular = [self.base_mip]
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff)

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def get_mip(self, roughness):
        return torch.where(
            roughness < self.max_roughness,
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2),
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )

    def apply_mask_by_dirs(self, dirs, threshold: float = 0.99):
        dirs = F.normalize(dirs.reshape(-1, 3), dim=-1)
        sg_dirs = F.normalize(self.sg_dirs, dim=-1)
        dots = sg_dirs @ dirs.T
        mask = dots.max(dim=1)[0] > threshold
        if not torch.any(mask):
            return
        with torch.no_grad():
            self.sg_amplitudes[mask] = self._softplus_inv(torch.full_like(self.sg_amplitudes[mask], 1.0e-4))

    def forward(self, image_infos, mode='pure_env', roughness=None):
        l = image_infos["viewdirs"]
        prefix = l.shape[:-1]
        if len(prefix) != 3:
            l = l.reshape(1, 1, -1, l.shape[-1])
        if self.transform is not None:
            l = l @ self.transform.T
        light = self._eval_sg(l).view(*prefix, -1)
        return light

    def query_light(self, l, mode='pure_env', roughness=None):
        prefix = l.shape[:-1]
        if len(prefix) != 3:
            l = l.reshape(1, 1, -1, l.shape[-1])
        if self.transform is not None:
            l = l @ self.transform.T
        light = self._eval_sg(l).view(*prefix, -1)
        return light

    def export_vis(self):
        return self.get_activated_base().detach()

    def get_param_groups(self):
        return {
            self.class_prefix + "all": self.parameters(),
        }

    def get_name(self):
        return self.class_prefix + "all"

    def freeze_envmap(self, optimizer):
        self.sg_dirs.requires_grad_(False)
        self.sg_amplitudes.requires_grad_(False)
        self.sg_sharpness.requires_grad_(False)
        if optimizer is None:
            return
        for group in optimizer.param_groups:
            if group.get("name") == self.class_prefix + "all":
                for param in group["params"]:
                    if param in optimizer.state:
                        del optimizer.state[param]
                break

    def load(self, path):
        if path.endswith(".exr"):
            image = pyexr.open(path).get()[:, :, :3]
        elif path.endswith(".hdr"):
            image = imageio.imread(path, format='HDR-FI')[:, :, :3]
        else:
            image = srgb_to_rgb(imageio.imread(path)[:, :, :3] / 255)
        image = torch.from_numpy(image).to(self.device)
        return image

    def load_prior(self, path):
        assert path is not None
        latlong_img = self.load(path)
        target_h = self.resolution // 2
        target_w = self.resolution
        _, _, texcoord = self._latlong_grid(target_h, target_w)
        latlong_img = dr.texture(latlong_img[None, ...], texcoord[None, ...], filter_mode='linear')[0].clamp_min(1e-4)
        self.activated_prior_base = latlong_img

    def get_prior(self):
        return self.activated_prior_base

    def init_from_envmap(self, envmap, num_sgs: int = None, sharpness: float = None):
        if isinstance(envmap, str):
            envmap = self.load(envmap)
        envmap = envmap.to(self.device)
        target_h = self.resolution // 2
        target_w = self.resolution
        dirs, _, texcoord = self._latlong_grid(target_h, target_w)
        latlong_img = dr.texture(envmap[None, ...], texcoord[None, ...], filter_mode='linear')[0].clamp_min(0.0)
        residual = latlong_img.clone()
        num_sgs = num_sgs or self.num_sgs
        num_sgs = min(num_sgs, self.num_sgs)
        sharpness = sharpness if sharpness is not None else self.init_sharpness

        with torch.no_grad():
            for idx in range(num_sgs):
                luminance = residual.max(dim=-1)[0]
                max_idx = torch.argmax(luminance)
                y = (max_idx // target_w).long()
                x = (max_idx % target_w).long()
                peak_dir = F.normalize(dirs[y, x], dim=-1)
                peak_color = residual[y, x].clamp_min(0.0)

                self.sg_dirs[idx].copy_(peak_dir)
                self.sg_sharpness[idx].copy_(self._softplus_inv(torch.tensor([sharpness], device=self.device)))
                self.sg_amplitudes[idx].copy_(self._softplus_inv(peak_color))

                dot = (dirs * peak_dir).sum(dim=-1).clamp(-1.0, 1.0)
                weights = torch.exp(sharpness * (dot - 1.0))
                sg_map = weights[..., None] * peak_color
                residual = (residual - sg_map).clamp_min(0.0)

            if num_sgs < self.num_sgs:
                self.sg_amplitudes[num_sgs:].copy_(self._softplus_inv(torch.zeros_like(self.sg_amplitudes[num_sgs:])))
                self.sg_sharpness[num_sgs:].copy_(self._softplus_inv(torch.full_like(self.sg_sharpness[num_sgs:], self.init_sharpness)))

    def load_full(self, path):
        assert path is not None
        self.init_from_envmap(path)


class EnvLight_SH_SG(torch.nn.Module):

    def __init__(
        self,
        class_name: str,
        resolution=1024,
        device: torch.device = torch.device("cuda"),
        activation_name: str = None,
        min_res: int = 64,
        max_res: int = 1024,
        min_roughness: float = 0.08,
        max_roughness: float = 0.5,
        init_value: float = 0.5,
        prior_path: str = None,
        path: str = None,
        sh_degree: int = 3,
        num_sgs: int = None,
        init_sharpness: float = 16.0,
        min_sharpness: float = 1.0e-4,
        sharpness_schedule: bool = False,
        sharpness_start_step: int = 0,
        sharpness_end_step: int = 0,
        sharpness_start: float = None,
        sharpness_end: float = None,
        peak_threshold: float = 0.75,
        peak_kernel_size: int = 3,
        peak_min_distance: int = 0,
        sg_amplitude_scale: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.resolution = resolution
        self.prior_path = prior_path
        self.min_res = min_res
        self.max_res = max_res
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.init_sharpness = init_sharpness
        self.min_sharpness = min_sharpness
        self.peak_threshold = peak_threshold
        self.peak_kernel_size = peak_kernel_size
        self.peak_min_distance = peak_min_distance
        self.sg_amplitude_scale = sg_amplitude_scale
        self.sh_degree = sh_degree
        self.activation_name = activation_name
        self.sharpness_schedule = sharpness_schedule
        self.sharpness_start_step = int(sharpness_start_step)
        self.sharpness_end_step = int(sharpness_end_step)
        self.sharpness_start = self.init_sharpness if sharpness_start is None else float(sharpness_start)
        self.sharpness_end = self.sharpness_start if sharpness_end is None else float(sharpness_end)

        self.init_transform = torch.tensor(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            dtype=torch.float32,
            device=self.device,
        )
        self.transform = self.init_transform.clone()
        self.base_mip = None
        self.register_buffer("init_sg_dirs", torch.empty(0, 3, device=self.device))

        self._init_sh_coeffs(init_value)
        self._init_sgs(num_sgs=num_sgs)

        if path is not None:
            self.init_from_envmap(path)

        self.activated_prior_base = None
        if self.prior_path is not None:
            self.load_prior(self.prior_path)

    def _init_sh_coeffs(self, init_value: float):
        num_coeffs = _num_sh_bases(self.sh_degree)
        self.sh_coeffs = torch.nn.Parameter(
            torch.zeros(num_coeffs, 3, device=self.device, dtype=torch.float32),
            requires_grad=True,
        )
        self.base = self.sh_coeffs

    def _init_sgs(self, num_sgs: int = None):
        if num_sgs is None:
            num_sgs = 1
        self.num_sgs = int(num_sgs)
        sg_dirs = torch.randn(self.num_sgs, 3, device=self.device, dtype=torch.float32)
        sg_dirs = F.normalize(sg_dirs, dim=-1)
        self.sg_dirs = torch.nn.Parameter(sg_dirs, requires_grad=True)

        init_amp = torch.full((self.num_sgs, 3), 0.0, device=self.device, dtype=torch.float32)
        init_sharp = torch.full((self.num_sgs, 1), self.init_sharpness, device=self.device, dtype=torch.float32)
        self.sg_amplitudes = torch.nn.Parameter(self._softplus_inv(init_amp), requires_grad=True)
        self.sg_sharpness = torch.nn.Parameter(
            self._softplus_inv(init_sharp),
            requires_grad=not self.sharpness_schedule,
        )
        if self.sharpness_schedule:
            self._set_sg_sharpness(self.sharpness_start)

    # def _softplus_inv(self, x: torch.Tensor) -> torch.Tensor:
    #     x = torch.clamp(x, min=1.0e-8)
    #     return torch.log(torch.expm1(x))
    
    def _softplus_inv(self, y: torch.Tensor, min_value: float = 1e-6) -> torch.Tensor:
        # 1. 避免 y 太小導致 log(exp(y)-1) 出錯 (負數或零)
        y_clamped = torch.clamp(y, min=min_value)
        
        # 2. 準備輸出的 tensor
        out = torch.zeros_like(y_clamped)
        
        # 3. 設定一個安全閾值 (Threshold)
        # 當 y > 20.0 時，exp(y) 已經很大了，softplus(y) ≈ y
        # 所以 inverse softplus 也直接回傳 y 即可，完全不需要過 exp，避免 overflow
        safe_mask = y_clamped > 20.0
        
        # 情況 A: 大數值 (直接 Linear Mapping)
        out[safe_mask] = y_clamped[safe_mask]
        
        # 情況 B: 小數值 (走原本的公式: log(exp(y) - 1))
        # 使用 expm1 (exp(x)-1) 可以增加接近 0 時的精度
        out[~safe_mask] = torch.log(torch.expm1(y_clamped[~safe_mask]))
        
        return out

    def _positive(self, x: torch.Tensor, min_value: float = 0.0) -> torch.Tensor:
        return F.softplus(x) + min_value

    def _get_sg_params(self):
        dirs = F.normalize(self.sg_dirs, dim=-1)
        sharpness = self._positive(self.sg_sharpness, self.min_sharpness).squeeze(-1)
        amplitudes = self._positive(self.sg_amplitudes, 0.0)
        return dirs, sharpness, amplitudes

    def _latlong_grid(self, height: int, width: int):
        u = (torch.arange(width, device=self.device, dtype=torch.float32) + 0.5) / width
        v = (torch.arange(height, device=self.device, dtype=torch.float32) + 0.5) / height
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        phi = (uu * 2.0 - 1.0) * torch.pi
        theta = vv * torch.pi
        sin_theta = torch.sin(theta)
        dirs = torch.stack(
            (
                sin_theta * torch.sin(phi),
                torch.cos(theta),
                -sin_theta * torch.cos(phi),
            ),
            dim=-1,
        )
        texcoord = torch.stack((uu, vv), dim=-1)
        return dirs, vv, texcoord

    def _eval_sh(self, directions: torch.Tensor):
        directions = F.normalize(directions, dim=-1)
        prefix = directions.shape[:-1]
        dirs_flat = directions.reshape(-1, 3)
        coeffs = self.sh_coeffs.unsqueeze(0).expand(dirs_flat.shape[0], -1, -1)
        sh_rgb = spherical_harmonics(self.sh_degree, dirs_flat, coeffs)
        return sh_rgb.reshape(*prefix, 3)

    def _eval_sg(self, directions: torch.Tensor):
        dirs, sharpness, amplitudes = self._get_sg_params()
        directions = F.normalize(directions, dim=-1)
        dot = torch.sum(directions[..., None, :] * dirs, dim=-1).clamp(-1.0, 1.0)
        weights = torch.exp(sharpness * (dot - 1.0))
        rgb = (weights[..., None] * amplitudes).sum(dim=-2)
        return rgb

    def _luminance(self, envmap: torch.Tensor):
        return 0.2126 * envmap[..., 0] + 0.7152 * envmap[..., 1] + 0.0722 * envmap[..., 2]

    def _find_envmap_peaks(self, envmap: torch.Tensor):
        luminance = self._luminance(envmap)
        log_luminance = torch.log1p(luminance)

        kernel = max(3, min(int(self.peak_kernel_size), 5))
        if kernel % 2 == 0:
            kernel += 1
        pooled = F.max_pool2d(log_luminance[None, None, ...], kernel, stride=1, padding=kernel // 2)[0, 0]

        max_val = log_luminance.max()
        if self.peak_threshold <= 1.0:
            thresh = self.peak_threshold * max_val
        else:
            thresh = torch.tensor(self.peak_threshold, device=self.device, dtype=log_luminance.dtype)

        is_peak = (log_luminance >= pooled) & (log_luminance >= thresh)
        ys, xs = torch.nonzero(is_peak, as_tuple=True)
        if ys.numel() == 0:
            max_idx = torch.argmax(log_luminance)
            ys = (max_idx // log_luminance.shape[1]).reshape(1)
            xs = (max_idx % log_luminance.shape[1]).reshape(1)
        values = log_luminance[ys, xs]
        order = torch.argsort(values, descending=True)
        ys = ys[order]
        xs = xs[order]

        if self.peak_min_distance > 0 and ys.numel() > 1:
            dirs, _, _ = self._latlong_grid(envmap.shape[0], envmap.shape[1])
            peak_dirs = dirs[ys, xs]
            angle_rad = float(self.peak_min_distance) * torch.pi / 180.0
            dot_thresh = torch.cos(torch.tensor(angle_rad, device=envmap.device, dtype=peak_dirs.dtype))
            keep_dirs = []
            keep_indices = []
            for idx, d in enumerate(peak_dirs):
                if not keep_dirs:
                    keep_dirs.append(d)
                    keep_indices.append(idx)
                    continue
                dots = torch.stack([torch.dot(d, kd) for kd in keep_dirs])
                if torch.all(dots <= dot_thresh):
                    keep_dirs.append(d)
                    keep_indices.append(idx)
            ys = ys[keep_indices]
            xs = xs[keep_indices]

        return ys, xs

    def _init_sh_from_envmap(self, envmap: torch.Tensor):
        mean_rgb = envmap.mean(dim=(0, 1))
        c0 = 0.28209479177387814
        with torch.no_grad():
            self.sh_coeffs.zero_()
            self.sh_coeffs[0] = mean_rgb / c0

    def _init_sgs_from_envmap(self, envmap: torch.Tensor, sharpness: float = None):
        sharpness = sharpness if sharpness is not None else self.init_sharpness
        ys, xs = self._find_envmap_peaks(envmap)
        dirs, _, _ = self._latlong_grid(envmap.shape[0], envmap.shape[1])
        peak_dirs = dirs[ys, xs]
        peak_colors = envmap[ys, xs].clamp_min(0.0) * float(self.sg_amplitude_scale)

        num_peaks = peak_dirs.shape[0]
        self.num_sgs = int(num_peaks)
        self.sg_dirs = torch.nn.Parameter(peak_dirs.clone(), requires_grad=True)
        self.sg_amplitudes = torch.nn.Parameter(self._softplus_inv(peak_colors), requires_grad=True)
        self.sg_sharpness = torch.nn.Parameter(
            self._softplus_inv(torch.full((num_peaks, 1), sharpness, device=self.device)),
            requires_grad=not self.sharpness_schedule,
        )
        self.init_sg_dirs = F.normalize(peak_dirs.detach().clone(), dim=-1)
        if self.sharpness_schedule:
            self._set_sg_sharpness(self.sharpness_start)
        
        print(f"Initialized {num_peaks} SGs from environment map.")

    def init_from_envmap(self, envmap):
        if isinstance(envmap, str):
            envmap = self.load(envmap)
        envmap = envmap.to(self.device)
        target_h = self.resolution // 2
        target_w = self.resolution
        _, _, texcoord = self._latlong_grid(target_h, target_w)
        latlong_img = dr.texture(envmap[None, ...], texcoord[None, ...], filter_mode='linear')[0].clamp_min(0.0)
        self._init_sh_from_envmap(latlong_img)
        self._init_sgs_from_envmap(latlong_img, sharpness=self.init_sharpness)
        if self.sharpness_schedule:
            self._set_sg_sharpness(self.sharpness_start)

    def get_activated_base(self):
        height = self.resolution // 2
        width = self.resolution
        dirs, _, _ = self._latlong_grid(height, width)
        envmap = self._eval_sh(dirs) + self._eval_sg(dirs)
        return envmap.clamp_min(0.0)

    def get_base(self):
        return self.get_activated_base()

    def update_pdf(self):
        with torch.no_grad():
            envmap = self.get_activated_base()
            _, vv, _ = self._latlong_grid(envmap.shape[0], envmap.shape[1])
            self._pdf = torch.max(envmap, dim=-1)[0] * torch.sin(vv * torch.pi)
            self._pdf = self._pdf / torch.sum(self._pdf)

    def sample_light_directions(self, B, sample_num, training=False):
        if not hasattr(self, "_pdf"):
            self.update_pdf()
        pdf_flat = self._pdf.reshape(-1)
        light_dir_idx = torch.multinomial(pdf_flat, B * sample_num, replacement=True)

        H, W = self._pdf.shape[:2]
        gx = ((light_dir_idx % W + 0.5) / W) * 2 - 1
        gy = (light_dir_idx // W + 0.5) / H
        if training:
            gx = gx + (torch.rand_like(gx) - 0.5) / W * 2
            gy = gy + (torch.rand_like(gy) - 0.5) / H
        sintheta, costheta = torch.sin(gy * torch.pi), torch.cos(gy * torch.pi)
        sinphi, cosphi = torch.sin(gx * torch.pi), torch.cos(gx * torch.pi)
        direction = torch.stack((
            sintheta * sinphi,
            costheta,
            -sintheta * cosphi
        ), dim=-1)

        if self.transform is not None:
            direction = direction @ self.transform
        direction = direction.reshape(B, sample_num, 3)

        probability = self.light_pdf(direction)
        return direction, probability

    def light_pdf(self, direction):
        if not hasattr(self, "_pdf"):
            self.update_pdf()
        pdf_flat = self._pdf.reshape(-1)
        direction_flat = direction.reshape(-1, 3)
        if self.transform is not None:
            direction_flat = direction_flat @ self.transform.T
        H, W = self._pdf.shape[:2]

        u = (torch.atan2(direction_flat[..., 0], -direction_flat[..., 2]).nan_to_num() / (2.0 * torch.pi) + 0.5)
        v = torch.acos(direction_flat[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi

        u_idx = (u * W).clamp(0, W - 1).long()
        v_idx = (v * H).clamp(0, H - 1).long()
        light_dir_idx = u_idx + v_idx * W

        pdf_weight = H * W / (2.0 * torch.pi ** 2 * torch.sin(v * torch.pi).clamp_min(1e-6))
        probability = (torch.take_along_dim(pdf_flat, light_dir_idx, dim=0) * pdf_weight).reshape(*direction.shape[:2], 1)
        return probability

    def capture(self):
        state_dict = super().state_dict()
        return {
            "transform": self.transform,
            "state_dict": state_dict,
            "activation": self.activation_name
        }

    def restore(self, model_args):
        activation = model_args.get('activation', None)
        self.activation_name = activation
        if 'transform' in model_args:
            self.init_transform = model_args['transform']
            self.transform = self.init_transform.clone()
        else:
            print("Warning: no transform found in model_args, using default transform")
        self.load_state_dict(model_args['state_dict'])

    def calibrate_direction(self, camera_dir):
        to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device=self.device)
        camera_dir_opengl = camera_dir @ to_opengl.T

        target_dir = torch.tensor([0.0, 0.0, -1.0], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
        camera_dir_opengl = camera_dir_opengl / (camera_dir_opengl.norm() + 1e-8)
        target_dir = target_dir / (target_dir.norm() + 1e-8)

        v = torch.cross(camera_dir_opengl, target_dir)
        s = v.norm()
        c = torch.dot(camera_dir_opengl, target_dir)

        if s < 1e-6:
            if c > 0:
                rot = torch.eye(3, device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
            else:
                axis = torch.tensor([1.0, 0.0, 0.0], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
                if torch.allclose(camera_dir_opengl, axis):
                    axis = torch.tensor([0.0, 1.0, 0.0], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
                v = torch.cross(camera_dir_opengl, axis)
                v = v / (v.norm() + 1e-8)
                K = torch.tensor([[0, -v[2], v[1]],
                                  [v[2], 0, -v[0]],
                                  [-v[1], v[0], 0]], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
                rot = torch.eye(3, device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype) + 2 * K @ K
        else:
            K = torch.tensor([[0, -v[2], v[1]],
                              [v[2], 0, -v[0]],
                              [-v[1], v[0], 0]], device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype)
            rot = torch.eye(3, device=camera_dir_opengl.device, dtype=camera_dir_opengl.dtype) + K + K @ K * ((1 - c) / (s ** 2 + 1e-8))

        to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device=self.device)
        self.init_transform = rot @ to_opengl
        self.transform = self.init_transform.clone()

    def set_rotate(self, angle_degrees: float = 0, rotate_vertical: bool = False):
        theta = torch.tensor(angle_degrees, dtype=torch.float32, device=self.device) * (torch.pi / 180.0)

        if not rotate_vertical:
            rot_y = torch.tensor([
                [torch.cos(theta), 0, torch.sin(theta)],
                [0, 1, 0],
                [-torch.sin(theta), 0, torch.cos(theta)]
            ], device=self.device)
            self.transform = rot_y @ self.init_transform
        else:
            rot_x = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(theta), -torch.sin(theta)],
                [0, torch.sin(theta), torch.cos(theta)]
            ], device=self.device)
            self.transform = rot_x @ self.init_transform

    def build_mips(self, cutoff=0.99):
        self.base_mip = latlong_to_cubemap(self.get_activated_base(), [self.max_res, self.max_res], self.device)
        self.specular = [self.base_mip]
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff)

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def get_mip(self, roughness):
        return torch.where(
            roughness < self.max_roughness,
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2),
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )

    def apply_mask_by_dirs(self, dirs, threshold: float = 0.99):
        if self.num_sgs == 0:
            return
        dirs = F.normalize(dirs.reshape(-1, 3), dim=-1)
        sg_dirs = F.normalize(self.sg_dirs, dim=-1)
        dots = sg_dirs @ dirs.T
        mask = dots.max(dim=1)[0] > threshold
        if not torch.any(mask):
            return
        with torch.no_grad():
            self.sg_amplitudes[mask] = self._softplus_inv(torch.full_like(self.sg_amplitudes[mask], 1.0e-4))

    def forward(self, image_infos, mode='pure_env', roughness=None):
        l = image_infos["viewdirs"]
        prefix = l.shape[:-1]
        if len(prefix) != 3:
            l = l.reshape(1, 1, -1, l.shape[-1])
        if self.transform is not None:
            l = l @ self.transform.T
        light = (self._eval_sh(l) + self._eval_sg(l)).view(*prefix, -1)
        return light

    def query_light(self, l, mode='pure_env', roughness=None):
        prefix = l.shape[:-1]
        if len(prefix) != 3:
            l = l.reshape(1, 1, -1, l.shape[-1])
        if self.transform is not None:
            l = l @ self.transform.T
        light = (self._eval_sh(l) + self._eval_sg(l)).view(*prefix, -1)
        return light

    def export_vis(self):
        return self.get_activated_base().detach()

    def get_param_groups(self):
        return {
            self.class_prefix + "all": self.parameters(),
        }

    def get_name(self):
        return self.class_prefix + "all"

    def compute_sg_reg(self):
        if self.init_sg_dirs.numel() == 0:
            return torch.tensor(0.0, device=self.sg_dirs.device)
        cur_dirs = F.normalize(self.sg_dirs, dim=-1)
        target_dirs = self.init_sg_dirs
        if cur_dirs.shape != target_dirs.shape:
            min_count = min(cur_dirs.shape[0], target_dirs.shape[0])
            if min_count == 0:
                return torch.tensor(0.0, device=self.sg_dirs.device)
            cur_dirs = cur_dirs[:min_count]
            target_dirs = target_dirs[:min_count]
        cos_sim = F.cosine_similarity(cur_dirs, target_dirs, dim=-1)
        return (1.0 - cos_sim).mean()

    def compute_SH_reg(self):
        return (self.sh_coeffs ** 2).mean()

    def _set_sg_sharpness(self, sharpness_value: float):
        sharpness_value = max(float(sharpness_value), self.min_sharpness)
        with torch.no_grad():
            value = torch.full(
                (self.sg_sharpness.shape[0], 1),
                sharpness_value,
                device=self.sg_sharpness.device,
                dtype=self.sg_sharpness.dtype,
            )
            self.sg_sharpness.copy_(self._softplus_inv(value))

    def update_sharpness(self, step: int):
        if not self.sharpness_schedule:
            return None
        if self.sharpness_end_step <= self.sharpness_start_step:
            sharpness_value = self.sharpness_end
        elif step <= self.sharpness_start_step:
            sharpness_value = self.sharpness_start
        elif step >= self.sharpness_end_step:
            sharpness_value = self.sharpness_end
        else:
            t = (step - self.sharpness_start_step) / (self.sharpness_end_step - self.sharpness_start_step)
            sharpness_value = self.sharpness_start + t * (self.sharpness_end - self.sharpness_start)
        self._set_sg_sharpness(sharpness_value)
        return sharpness_value

    def freeze_envmap(self, optimizer):
        self.sh_coeffs.requires_grad_(False)
        self.sg_dirs.requires_grad_(False)
        self.sg_amplitudes.requires_grad_(False)
        self.sg_sharpness.requires_grad_(False)
        if optimizer is None:
            return
        for group in optimizer.param_groups:
            if group.get("name") == self.class_prefix + "all":
                for param in group["params"]:
                    if param in optimizer.state:
                        del optimizer.state[param]
                break

    def load(self, path):
        if path.endswith(".exr"):
            image = pyexr.open(path).get()[:, :, :3]
        elif path.endswith(".hdr"):
            image = imageio.imread(path, format='HDR-FI')[:, :, :3]
        else:
            image = srgb_to_rgb(imageio.imread(path)[:, :, :3] / 255)
        image = torch.from_numpy(image).to(self.device)
        return image

    def load_prior(self, path):
        assert path is not None
        latlong_img = self.load(path)
        target_h = self.resolution // 2
        target_w = self.resolution
        _, _, texcoord = self._latlong_grid(target_h, target_w)
        latlong_img = dr.texture(latlong_img[None, ...], texcoord[None, ...], filter_mode='linear')[0].clamp_min(1e-4)
        self.activated_prior_base = latlong_img

    def get_prior(self):
        return self.activated_prior_base

    def load_full(self, path):
        assert path is not None
        self.init_from_envmap(path)


class AffineTransform(nn.Module):
    def __init__(
        self,
        class_name: str,
        n: int, 
        embedding_dim: int = 4,
        pixel_affine: bool = False,
        base_mlp_layer_width: int = 64,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.embedding_dim = embedding_dim
        self.pixel_affine = pixel_affine
        self.embedding = nn.Embedding(n, embedding_dim, dtype=torch.float32)
        
        input_dim = (embedding_dim + 2)if self.pixel_affine else embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, base_mlp_layer_width),
            nn.ReLU(),
            nn.Linear(base_mlp_layer_width, 12),
        )
        self.in_test_set = False
        
        self.zero_init()
        
    def zero_init(self):
        torch.nn.init.zeros_(self.embedding.weight)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, image_infos):
        if "img_idx" in image_infos and not self.in_test_set:
            embedding = self.embedding(image_infos["img_idx"])
        else:
            # use mean appearance embedding
            embedding = torch.ones(
                (*image_infos["viewdirs"].shape[:-1], self.embedding_dim),
                device=image_infos["viewdirs"].device,
            ) * self.embedding.weight.mean(dim=0)
        if self.pixel_affine:
            embedding = torch.cat([embedding, image_infos["pixel_coords"]], dim=-1)
        affine = self.decoder(embedding)
        affine = affine.reshape(*embedding.shape[:-1], 3, 4)
        
        affine[..., :3, :3] = affine[..., :3, :3] + torch.eye(3, device=affine.device).reshape(1, 3, 3)
        return affine

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
    
class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(
        self,
        class_name: str,
        n: int,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        
        self.zero_init() # important for initialization !!

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, x_multires=10, t_multires=10):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.x_multires = x_multires
        self.t_multires = t_multires
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(self.x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling
    
    
class ConditionalDeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, embed_dim=10,
                 x_multires=10, t_multires=10, 
                 deform_quat=True, deform_scale=True):
        super(ConditionalDeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.embed_dim = embed_dim
        self.deform_quat = deform_quat
        self.deform_scale = deform_scale
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch + embed_dim

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        if self.deform_quat:
            self.gaussian_rotation = nn.Linear(W, 4)
        if self.deform_scale:
            self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t, condition):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb, condition], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, condition, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling, rotation = None, None
        if self.deform_scale: 
            scaling = self.gaussian_scaling(h)
        if self.deform_quat:
            rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling

class VoxelDeformer(nn.Module):
    def __init__(
        self,
        vtx,
        vtx_features,
        resolution_dhw=[8, 32, 32],
        short_dim_dhw=0,  # 0 is d, corresponding to z
        long_dim_dhw=1,
        is_resume=False
    ) -> None:
        super().__init__()
        # vtx B,N,3, vtx_features: B,N,J
        # d-z h-y w-x; human is facing z; dog is facing x, z is upward, should compress on y
        B = vtx.shape[0]
        assert vtx.shape[0] == vtx_features.shape[0], "Batch size mismatch"

        # * Prepare Grid
        self.resolution_dhw = resolution_dhw
        device = vtx.device
        d, h, w = self.resolution_dhw

        self.register_buffer(
            "ratio",
            torch.Tensor(
                [self.resolution_dhw[long_dim_dhw] / self.resolution_dhw[short_dim_dhw]]
            ).squeeze(),
        )
        self.ratio_dim = -1 - short_dim_dhw
        x_range = (
            (torch.linspace(-1, 1, steps=w, device=device))
            .view(1, 1, 1, w)
            .expand(1, d, h, w)
        )
        y_range = (
            (torch.linspace(-1, 1, steps=h, device=device))
            .view(1, 1, h, 1)
            .expand(1, d, h, w)
        )
        z_range = (
            (torch.linspace(-1, 1, steps=d, device=device))
            .view(1, d, 1, 1)
            .expand(1, d, h, w)
        )
        grid = (
            torch.cat((x_range, y_range, z_range), dim=0)
            .reshape(1, 3, -1)
            .permute(0, 2, 1)
        )
        grid = grid.expand(B, -1, -1)

        gt_bbox_min = (vtx.min(dim=1).values).to(device)
        gt_bbox_max = (vtx.max(dim=1).values).to(device)
        offset = (gt_bbox_min + gt_bbox_max) * 0.5
        self.register_buffer(
            "global_scale", torch.Tensor([1.2]).squeeze()
        )  # from Fast-SNARF
        scale = (
            (gt_bbox_max - gt_bbox_min).max(dim=-1).values / 2 * self.global_scale
        ).unsqueeze(-1)

        corner = torch.ones_like(offset) * scale
        corner[:, self.ratio_dim] /= self.ratio
        min_vert = (offset - corner).reshape(-1, 1, 3)
        max_vert = (offset + corner).reshape(-1, 1, 3)
        self.bbox = torch.cat([min_vert, max_vert], dim=1)

        self.register_buffer("scale", scale.unsqueeze(1)) # [B, 1, 1]
        self.register_buffer("offset", offset.unsqueeze(1)) # [B, 1, 3]

        grid_denorm = self.denormalize(
            grid
        )  # grid_denorm is in the same scale as the canonical body

        if not is_resume:
            weights = (
                self._query_weights_smpl(
                    grid_denorm,
                    smpl_verts=vtx.detach().clone(),
                    smpl_weights=vtx_features.detach().clone(),
                )
                .detach()
                .clone()
            )
        else:
            # random initialization
            weights = torch.randn(
                B, vtx_features.shape[-1], *resolution_dhw
            ).to(device)

        self.register_buffer("lbs_voxel_base", weights.detach())
        self.register_buffer("grid_denorm", grid_denorm)

        self.num_bones = vtx_features.shape[-1]

        # # debug
        # import numpy as np
        # np.savetxt("./debug/dbg.xyz", grid_denorm[0].detach().cpu())
        # np.savetxt("./debug/vtx.xyz", vtx[0].detach().cpu())
        return

    def enable_voxel_correction(self):
        voxel_w_correction = torch.zeros_like(self.lbs_voxel_base)
        self.voxel_w_correction = nn.Parameter(voxel_w_correction)

    def enable_additional_correction(self, additional_channels, std=1e-4):
        additional_correction = (
            torch.ones(
                self.lbs_voxel_base.shape[0],
                additional_channels,
                *self.lbs_voxel_base.shape[2:]
            )
            * std
        )
        self.additional_correction = nn.Parameter(additional_correction)

    @property
    def get_voxel_weight(self):
        w = self.lbs_voxel_base
        if hasattr(self, "voxel_w_correction"):
            w = w + self.voxel_w_correction
        if hasattr(self, "additional_correction"):
            w = torch.cat([w, self.additional_correction], dim=1)
        return w

    def get_tv(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        tv_x = torch.abs(d[:, :, 1:, :, :] - d[:, :, :-1, :, :]).mean()
        tv_y = torch.abs(d[:, :, :, 1:, :] - d[:, :, :, :-1, :]).mean()
        tv_z = torch.abs(d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).mean()
        return (tv_x + tv_y + tv_z) / 3.0
        # tv_x = torch.abs(d[:, :, 1:, :, :] - d[:, :, :-1, :, :]).sum()
        # tv_y = torch.abs(d[:, :, :, 1:, :] - d[:, :, :, :-1, :]).sum()
        # tv_z = torch.abs(d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).sum()
        # return tv_x + tv_y + tv_z

    def get_mag(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        return torch.norm(d, dim=1).mean()

    def forward(self, xc, mode="bilinear"):
        shape = xc.shape  # ..., 3
        # xc = xc.reshape(1, -1, 3)
        w = F.grid_sample(
            self.get_voxel_weight,
            self.normalize(xc)[:, :, None, None],
            align_corners=True,
            mode=mode,
            padding_mode="border",
        )
        w = w.squeeze(3, 4).permute(0, 2, 1)
        w = w.reshape(*shape[:-1], -1)
        # * the w may have more channels
        return w

    def normalize(self, x):
        x_normalized = x.clone()
        x_normalized -= self.offset
        x_normalized /= self.scale
        x_normalized[..., self.ratio_dim] *= self.ratio
        return x_normalized

    def denormalize(self, x):
        x_denormalized = x.clone()
        x_denormalized[..., self.ratio_dim] /= self.ratio
        x_denormalized *= self.scale
        x_denormalized += self.offset
        return x_denormalized

    def _query_weights_smpl(self, x, smpl_verts, smpl_weights):
        # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py
        dist, idx, _ = knn_points(x, smpl_verts.detach(), K=30) # [B, N, 30]
        dist = dist.sqrt().clamp_(0.0001, 1.0)
        expanded_smpl_weights = smpl_weights.unsqueeze(2).expand(-1, -1, idx.shape[2], -1) # [B, N, 30, J]
        weights = expanded_smpl_weights.gather(1, idx.unsqueeze(-1).expand(-1, -1, -1, expanded_smpl_weights.shape[-1])) # [B, N, 30, J]

        ws = 1.0 / dist
        ws = ws / ws.sum(-1, keepdim=True)
        weights = (ws[..., None] * weights).sum(-2)

        b = x.shape[0]
        c = smpl_weights.shape[-1]
        d, h, w = self.resolution_dhw
        weights = weights.permute(0, 2, 1).reshape(b, c, d, h, w)
        for _ in range(30):
            mean = (
                weights[:, :, 2:, 1:-1, 1:-1]
                + weights[:, :, :-2, 1:-1, 1:-1]
                + weights[:, :, 1:-1, 2:, 1:-1]
                + weights[:, :, 1:-1, :-2, 1:-1]
                + weights[:, :, 1:-1, 1:-1, 2:]
                + weights[:, :, 1:-1, 1:-1, :-2]
            ) / 6.0
            weights[:, :, 1:-1, 1:-1, 1:-1] = (
                weights[:, :, 1:-1, 1:-1, 1:-1] - mean
            ) * 0.7 + mean
            sums = weights.sum(1, keepdim=True)
            weights = weights / sums
        return weights.detach()
