import trimesh
import torch

try:
    from surfel_tracer import GaussianTracer
except ImportError:
    print("Fail to import surfel_tracer. Make sure Gaussian Splatting is installed correctly.")
    # use dummy class to avoid errors
    class GaussianTracer:
        def __init__(self, *args, **kwargs):
            pass

# Using 2D gaussian tracer from IRGS: https://github.com/fudan-zvg/IRGS

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def flip_align_view(normal, viewdir):
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = torch.sum(
        normal * -viewdir, dim=-1, keepdims=True) # (N, 1)
    non_flip = dotprod>=0 # (N, 1)
    normal_flipped = normal*torch.where(non_flip, 1, -1) # (N, 3)
    return normal_flipped, non_flip

def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return x / norm

def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
    RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
    trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
    trans[:,:3,:3] = RS
    trans[:, 3,:3] = center
    trans[:, 3, 3] = 1
    return trans

class CustomSurfelTracer(GaussianTracer):
    
    def __init__(self, transmittance_min=0.001):
        super().__init__(transmittance_min)
    
        icosahedron = trimesh.creation.icosahedron()
        
        # change to inner sphere radius equal to 1.0
        # the central point of each face must be on the unit sphere
        self.unit_icosahedron_vertices = torch.from_numpy(icosahedron.vertices).float().cuda() * 1.2584 
        self.unit_icosahedron_faces = torch.from_numpy(icosahedron.faces).long().cuda()
        self.alpha_min = 1 / 255
    
    def get_boundings(self, gaussians, alpha_min=0.01):
        mu = gaussians.means
        opacity = gaussians.opacities
        scale = gaussians.scales
        rotation = gaussians.quats
        # gaussians z-scale is 0, but not sure what will happen, so set it to be 1e-6, as in the original code
        scale[..., 2] = 1e-6
        
        L = build_scaling_rotation(scale, rotation)
        
        vertices_b = (2 * (opacity/alpha_min).log()).sqrt()[:, None] * (self.unit_icosahedron_vertices[None] @ L.transpose(-1, -2)) + mu[:, None]
        faces_b = self.unit_icosahedron_faces[None] + torch.arange(mu.shape[0], device="cuda")[:, None, None] * 12
        gs_id = torch.arange(mu.shape[0], device="cuda")[:, None].expand(-1, faces_b.shape[1])
        return vertices_b.reshape(-1, 3), faces_b.reshape(-1, 3), gs_id.reshape(-1)
    
    
    def build_acc_dataclass(self, gaussians):
        vertices_b, faces_b, gs_id = self.get_boundings(gaussians=gaussians, alpha_min=self.alpha_min)
        self.build_bvh(vertices_b, faces_b, gs_id)
    
    def update_acc_dataclass(self, gaussians):
        vertices_b, faces_b, gs_id = self.get_boundings(gaussians=gaussians, alpha_min=self.alpha_min)
        self.update_bvh(vertices_b, faces_b, gs_id)
    
    def render_dataclass(self, gaussians, rays_o, rays_d, features=None, camera_center=None, back_culling=False, opacity_mask=None):
        means3D = gaussians.means
        num_gaussians = means3D.shape[0]
        shs_dc = gaussians.features_dc
        shs_rest = gaussians.features_rest.reshape(num_gaussians, -1)
        shs = torch.cat([shs_dc, shs_rest], dim=1)
        opacity = gaussians.opacities*opacity_mask if opacity_mask is not None else gaussians.opacities
        scales = gaussians.scales
        rotations = gaussians.quats
        
        s = 1 / scales[..., :2]
        R = build_rotation(rotations)
        ru = R[:, :, 0] * s[:,0:1]
        rv = R[:, :, 1] * s[:,1:2]
        
        active_sh_degree = 3 # hardcoded for now
        
        # splat2world = self.get_covariance()
        splat2world = build_covariance_from_scaling_rotation(means3D, scales[..., :2], 1.0, rotations)
        normals_raw = splat2world[: ,2, :3] 
        if camera_center is not None:
            normals_raw, positive = flip_align_view(normals_raw, means3D - camera_center)
        normals = safe_normalize(normals_raw)
        
        color, normal, feature, depth, alpha = self.trace(rays_o, rays_d, means3D, opacity, ru, rv, normals, features, shs, alpha_min=self.alpha_min, deg=active_sh_degree, back_culling=back_culling)
        
        alpha_ = alpha[..., None]
        color = torch.where(alpha_ < 1 - self.transmittance_min, color, color / alpha_)
        normal = torch.where(alpha_ < 1 - self.transmittance_min, normal, normal / alpha_)
        feature = torch.where(alpha_ < 1 - self.transmittance_min, feature, feature / alpha_)
        depth = torch.where(alpha < 1 - self.transmittance_min, depth, depth / alpha)
        alpha = torch.where(alpha < 1 - self.transmittance_min, alpha, torch.ones_like(alpha))
        
        return {
            "color": color,
            "normal": normal,
            "feature": feature,
            "depth": depth,
            "alpha" : alpha,
            "normals": normals,
        }

if __name__ == "__main__":
    tracer = CustomSurfelTracer()
    print("CustomSurfelTracer initialized successfully.")