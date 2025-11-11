
import torch
from torch import nn

from .core import Callback

# Cell
class PatchCB(Callback):
    def __init__(self, patch_len, stride,
                 behavior_adaptive: bool = False,
                 angle_threshold: float = 15.0):
        """
        Args:
            behavior_adaptive: Use behavior-adaptive patching
            angle_threshold: Angle threshold for behavior detection
        """
        self.patch_len = patch_len
        self.stride = stride
        self.behavior_adaptive = behavior_adaptive
        self.angle_threshold = angle_threshold

    def before_forward(self):
        self.set_patch()

    def set_patch(self):
        """Apply behavior-adaptive or standard patching"""
        if self.behavior_adaptive:
            xb_patch, num_patch = behavior_adaptive_patching(
                self.xb, self.patch_len, self.angle_threshold)
        else:
            xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)

        self.learner.xb = xb_patch


class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio,
                 mask_when_pred: bool = False,
                 behavior_adaptive: bool = True,
                 angle_threshold: float = 15.0):
        """
        Args:
            behavior_adaptive: Use behavior-adaptive patching instead of uniform
            angle_threshold: Angle threshold for behavior detection (degrees)
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.behavior_adaptive = behavior_adaptive
        self.angle_threshold = angle_threshold

    def before_fit(self):
        self.learner.loss_func = self._loss
        device = self.learner.device

    def before_forward(self):
        self.patch_masking()

    def patch_masking(self):
        """Apply behavior-adaptive or standard patching"""
        if self.behavior_adaptive:
            # Use behavior-adaptive patching
            xb_patch, num_patch = behavior_adaptive_patching(
                self.xb, self.patch_len, self.angle_threshold)
        else:
            # Use standard uniform patching
            xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)

        # Apply masking
        xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)
        self.mask = self.mask.bool()
        self.learner.xb = xb_mask
        self.learner.yb = xb_patch

    def _loss(self, preds, target):
        """Handle Flight2Vec multi-output case"""
        if isinstance(preds, tuple):
            reconstruction_pred = preds[0]
            if isinstance(target, tuple):
                reconstruction_target = target[0]
            else:
                reconstruction_target = target
        else:
            reconstruction_pred = preds
            reconstruction_target = target

        loss = (reconstruction_pred - reconstruction_target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask).sum() / self.mask.sum()
        return loss


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


class Patch(nn.Module):
    def __init__(self,seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len  + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)                 # xb: [bs x num_patch x n_vars x patch_len]
        return x


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, nvars,device=xb.device)  # noise in [0, 1], bs x L x nvars
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]                                              # ids_keep: [bs x len_keep x nvars]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]                                                 # ids_keep: [bs x len_keep]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))        # x_kept: [bs x len_keep x dim]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)                        # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))    # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)                                          # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore


# Add to the end of patch_mask.py

def compute_direction_labels(data, patch_len=32):
    """
    Compute 26-class direction labels for Flight2Vec
    data: [bs, seq_len, n_vars] where first 3 vars are [lon, lat, alt]
    returns: [bs, num_patches] direction labels
    """
    bs, seq_len, n_vars = data.shape

    # Extract position info (lon, lat, alt)
    positions = data[:, :, :3]  # [bs, seq_len, 3]

    # Calculate direction vectors: next_point - current_point
    direction_vectors = positions[:, 1:, :] - positions[:, :-1, :]  # [bs, seq_len-1, 3]

    # Get signs (-1, 0, 1)
    direction_signs = torch.sign(direction_vectors)  # [bs, seq_len-1, 3]

    # Convert to 26 classes
    direction_labels = []
    for i in range(bs):
        batch_labels = []
        for j in range(0, direction_signs.shape[1], patch_len):
            patch_dirs = direction_signs[i, j:j + patch_len, :]
            # Use middle position of patch as label
            mid_idx = min(patch_len // 2, patch_dirs.shape[0] - 1)
            d_lon, d_lat, d_alt = patch_dirs[mid_idx]

            # Map (-1,0,1) to (0,1,2) and calculate class
            d_lon_idx = int(d_lon.item() + 1)  # -1->0, 0->1, 1->2
            d_lat_idx = int(d_lat.item() + 1)
            d_alt_idx = int(d_alt.item() + 1)

            # Calculate 26-class label (exclude all-zero case)
            label = d_lon_idx * 9 + d_lat_idx * 3 + d_alt_idx
            if label == 13:  # (1,1,1) -> original (0,0,0) case
                label = 0
            elif label > 13:
                label -= 1

            batch_labels.append(label)
        direction_labels.append(batch_labels)

    return torch.tensor(direction_labels, dtype=torch.long)


def calculate_flight_angles(coordinates, window_len=5, scale_factor=1000):
    """Calculate angle changes for behavior detection"""
    angles = []
    num_points = len(coordinates)

    # Calculate scale factors for normalization
    lon_range = coordinates[:, 0].max() - coordinates[:, 0].min()
    lat_range = coordinates[:, 1].max() - coordinates[:, 1].min()
    alt_range = coordinates[:, 2].max() - coordinates[:, 2].min()

    lon_scale = (1.0 / lon_range * scale_factor) if lon_range != 0 else 1.0
    lat_scale = (1.0 / lat_range * scale_factor) if lat_range != 0 else 1.0
    alt_scale = (1.0 / alt_range * scale_factor) if alt_range != 0 else 1.0

    for i in range(window_len, num_points - window_len):
        prev_point = coordinates[i - window_len]
        curr_point = coordinates[i]
        next_point = coordinates[i + window_len]

        # Apply scaling
        prev_scaled = torch.tensor([prev_point[0] * lon_scale, prev_point[1] * lat_scale, prev_point[2] * alt_scale])
        curr_scaled = torch.tensor([curr_point[0] * lon_scale, curr_point[1] * lat_scale, curr_point[2] * alt_scale])
        next_scaled = torch.tensor([next_point[0] * lon_scale, next_point[1] * lat_scale, next_point[2] * alt_scale])

        vector1 = curr_scaled - prev_scaled
        vector2 = next_scaled - curr_scaled

        if torch.norm(vector1) == 0 or torch.norm(vector2) == 0:
            angles.append(0)
            continue

        dot_product = torch.dot(vector1, vector2)
        norm_product = torch.norm(vector1) * torch.norm(vector2)
        if norm_product == 0:
            angle = 0
        else:
            cos_theta = torch.clamp(dot_product / norm_product, -1.0, 1.0)
            angle = torch.acos(cos_theta) * 180.0 / torch.pi  # Convert to degrees

        angles.append(angle.item())

    return angles


def behavior_adaptive_patching(xb, patch_len, angle_threshold=15, window_len=5):
    """
    Create patches using behavior-adaptive strategy
    xb: [bs, seq_len, n_vars]
    """
    bs, seq_len, n_vars = xb.shape
    device = xb.device

    all_patches = []
    all_num_patches = []

    for b in range(bs):
        # Extract coordinates for this batch item
        coordinates = xb[b, :, :3].cpu().numpy()  # lon, lat, alt

        # Calculate angles
        angles = calculate_flight_angles(coordinates, window_len)

        # Identify activity points
        activity_points = []
        for i, angle in enumerate(angles):
            if angle > angle_threshold:
                activity_points.append(i + window_len)  # Adjust for window offset

        if not activity_points:
            # No activity points, use uniform patching
            patches, num_patch = create_patch(xb[b:b + 1], patch_len, patch_len)
            all_patches.append(patches[0])
            all_num_patches.append(num_patch)
            continue

        # Group consecutive activity points
        groups = []
        activity_points = sorted(set(activity_points[::2]))  # Reduce density

        if activity_points:
            current_group = [activity_points[0]]
            for point in activity_points[1:]:
                if point - current_group[-1] <= 20:
                    current_group.append(point)
                else:
                    groups.append(current_group)
                    current_group = [point]
            groups.append(current_group)

        # Get center points of each group
        center_points = []
        for group in groups:
            mid_idx = len(group) // 2
            center_points.append(group[mid_idx])

        # Create behavior patches around center points
        behavior_indices = set()
        half_patch = patch_len // 2

        for point in center_points:
            start = max(0, point - half_patch)
            end = min(seq_len, point + half_patch)
            for i in range(start, end):
                behavior_indices.add(i)

        # Calculate remaining indices for non-behavior patches
        all_indices = set(range(seq_len))
        remaining_indices = sorted(all_indices - behavior_indices)

        # Create patches from behavior indices
        patches_data = []
        behavior_indices = sorted(behavior_indices)

        # Group behavior indices into patches
        i = 0
        while i < len(behavior_indices) - patch_len + 1:
            patch_indices = behavior_indices[i:i + patch_len]
            if len(patch_indices) == patch_len:
                patch_data = xb[b, patch_indices, :]
                patches_data.append(patch_data)
            i += patch_len

        # Add some non-behavior patches if we have remaining space
        if len(remaining_indices) >= patch_len:
            step = max(1, len(remaining_indices) // max(1, (seq_len // patch_len - len(patches_data))))
            for i in range(0, len(remaining_indices) - patch_len + 1, step * patch_len):
                if len(patches_data) >= seq_len // patch_len:
                    break
                patch_indices = remaining_indices[i:i + patch_len]
                if len(patch_indices) == patch_len:
                    patch_data = xb[b, patch_indices, :]
                    patches_data.append(patch_data)

        # If we don't have enough patches, fill with uniform sampling
        while len(patches_data) < seq_len // patch_len:
            start = len(patches_data) * patch_len
            if start + patch_len <= seq_len:
                patch_data = xb[b, start:start + patch_len, :]
                patches_data.append(patch_data)
            else:
                break

        if patches_data:
            batch_patches = torch.stack(patches_data, dim=0)  # [num_patches, patch_len, n_vars]
            batch_patches = batch_patches.permute(1, 0, 2)  # [patch_len, num_patches, n_vars]
            # Actually we need [num_patches, n_vars, patch_len]
            batch_patches = batch_patches.permute(1, 2, 0)  # [num_patches, n_vars, patch_len]
        else:
            # Fallback to uniform patching
            patches, num_patch = create_patch(xb[b:b + 1], patch_len, patch_len)
            batch_patches = patches[0]

        all_patches.append(batch_patches)
        all_num_patches.append(batch_patches.shape[0])

    # Stack all patches
    max_patches = max(all_num_patches)

    # Pad patches to same length
    padded_patches = []
    for patches in all_patches:
        if patches.shape[0] < max_patches:
            # Pad with zeros or repeat last patch
            padding_needed = max_patches - patches.shape[0]
            last_patch = patches[-1:].repeat(padding_needed, 1, 1)
            patches = torch.cat([patches, last_patch], dim=0)
        padded_patches.append(patches)

    result = torch.stack(padded_patches, dim=0)  # [bs, num_patches, n_vars, patch_len]

    return result, max_patches

# if __name__ == "__main__":
#     bs, L, nvars, D = 2,20,4,5
#     xb = torch.randn(bs, L, nvars, D)
#     xb_mask, mask, ids_restore = create_mask(xb, mask_ratio=0.5)
#     breakpoint()


