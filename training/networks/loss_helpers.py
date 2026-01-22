
import math
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

def visualize_boundary(mask, boundary, b=0, t=0):
    m = mask[b, t].detach().cpu()
    bd = boundary[b, t].detach().cpu()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(m, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("Mask")
    axs[0].axis("off")

    axs[1].imshow(bd, cmap="hot", vmin=0, vmax=bd.max())
    axs[1].set_title("Boundary (scaled)")
    axs[1].axis("off")

    axs[2].imshow(m, cmap="gray", vmin=0, vmax=1)
    axs[2].imshow(bd, cmap="hot", alpha=0.9)
    axs[2].set_title("Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig("boundary_visualization.png")

def Patch_frame_CE_loss(
    patch_scores, # [B, T, N] logits
    labels, # [B,] (0=real, 1=fake)
    cfg=None,
    loss_function=F.binary_cross_entropy_with_logits
    ):
    """
    patch_scores: [B, T, N] logits
    labels: [B,] (0=real, 1=fake)
    returns: scalar loss
    """
    topk_percent = cfg.get('topk_percent', 0.2)
    B, T, N = patch_scores.shape
    device = patch_scores.device
    K = max(1, int(N * topk_percent))
    # focus on top-k patches per frame
    topk_scores, _ = torch.topk(patch_scores, k=K, dim=2)  # [B,T,K]
    patch_scores = topk_scores.reshape(B, T * K)  # [B,T*K]
    labels_exp = labels.unsqueeze(-1).float().expand(-1, T * K)  # [B,T*K]
    loss = loss_function(
        patch_scores,
        labels_exp,
        reduction='mean'
    )
    loss=cfg['alpha']*loss
    return loss

def patch_corr(x, y, eps=1e-6):
    """
    x, y: [B, N]
    returns: mean correlation over batch
    """
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)

    x_norm = torch.norm(x, dim=1)
    y_norm = torch.norm(y, dim=1)

    corr = (x * y).sum(dim=1) / (x_norm * y_norm + eps)
    return corr.mean()

def scores_to_patch_map(patch_scores):
    """
    patch_scores: [B, T, N]
    returns: [B, T, H_p, W_p]
    """
    B, T, N = patch_scores.shape
    H = W = int(N ** 0.5)
    return patch_scores.view(B, T, H, W)

def downsample_flow(flow, target_h, target_w):
    """
    flow: [B, T-1, 2, H, W]
    returns: [B, T-1, 2, target_h, target_w]
    """
    B, Tm1, _, H, W = flow.shape
    flow_ds = F.interpolate(
        flow.view(B * Tm1, 2, H, W),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )
    return flow_ds.view(B, Tm1, 2, target_h, target_w)

def warp_patch_map(patch_map, flow):
    """
    patch_map: [B, H, W]
    flow: [B, 2, H, W]  (dx, dy)
    """
    B, H, W = patch_map.shape
    device = patch_map.device

    # normalized grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid = torch.stack((x, y), dim=-1)  # [H,W,2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    # normalize flow
    flow_norm = torch.zeros_like(flow)
    flow_norm[:, 0] = flow[:, 0] / (W / 2)
    flow_norm[:, 1] = flow[:, 1] / (H / 2)

    warped = F.grid_sample(
        patch_map.unsqueeze(1),
        grid + flow_norm.permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return warped.squeeze(1)

def patch_continuity_flow_loss(
    patch_scores,    # [B,T,N]
    flow,            # [B,T-1,2,H,W]
    alpha_fake=1.0,
    conf_thresh=0.5
):
    B, T, N = patch_scores.shape
    device = patch_scores.device

    H_p = W_p = int(N ** 0.5)

    # patch probability maps
    patch_maps = scores_to_patch_map(torch.sigmoid(patch_scores))

    # downsample flow
    flow_ds = downsample_flow(flow, H_p, W_p)

    loss = 0.0
    count = 0

    for t in range(1, T):
        prev = patch_maps[:, t - 1]      # [B,H_p,W_p]
        curr = patch_maps[:, t]

        warped_prev = warp_patch_map(prev, flow_ds[:, t - 1])

        # focus on confident fake regions
        mask = curr > conf_thresh

        diff = torch.abs(curr - warped_prev)
        diff = diff * mask.float()

        loss += diff.mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)

    return loss / count

def patch_continuity_loss(
    patch_scores,          # [B, T, N] logits
    window=1,              # temporal look-back (1 = t vs t-1)
    topk=5,                # focus on strongest fake patches
    spatial_radius=1       # neighborhood size
):
    """
    Enforces smooth spatial transport of fake evidence across frames.
    """

    B, T, N = patch_scores.shape
    device = patch_scores.device

    # infer patch grid
    H = W = int(math.sqrt(N))
    if H * W != N:
        raise ValueError(f"N={N} is not a square number")

    probs = torch.sigmoid(patch_scores)  # [B,T,N]

    loss_accum = 0.0
    count = 0

    for t in range(window, T):
        prev = probs[:, t - window]   # [B,N]
        curr = probs[:, t]            # [B,N]

        # focus on strong fake evidence at current frame
        topk_vals, topk_idx = torch.topk(curr, k=min(topk, N), dim=1)

        for b in range(B):
            for k_idx in topk_idx[b]:
                idx = k_idx.item()
                y = idx // W
                x = idx % W

                # neighborhood in previous frame
                y0 = max(0, y - spatial_radius)
                y1 = min(H, y + spatial_radius + 1)
                x0 = max(0, x - spatial_radius)
                x1 = min(W, x + spatial_radius + 1)

                nbr_vals = []
                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        nbr_idx = yy * W + xx
                        nbr_vals.append(prev[b, nbr_idx])

                nbr_vals = torch.stack(nbr_vals)

                # continuity penalty
                # current fake evidence should be explainable by nearby past evidence
                loss_accum += F.relu(curr[b, idx] - nbr_vals.max())

                count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)

    return loss_accum / count

def extract_boundary(mask, kernel_size=3):
    """
    mask: [B,H,W], [B,T,H,W], or [B,T,H,W,1]
    returns: boundary map with same shape
    """
    # Remove channel dim if present
    if mask.dim() == 5 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)  # [B,T,H,W]

    if mask.dim() == 3:
        B, H, W = mask.shape
        mask_ = mask.unsqueeze(1)   # [B,1,H,W]
        is_temporal = False
    elif mask.dim() == 4:
        B, T, H, W = mask.shape
        mask_ = mask.view(B*T, 1, H, W)
        is_temporal = True
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    pad = kernel_size // 2
    dilated = F.max_pool2d(mask_, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask_, kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(min=0)

    if is_temporal:
        boundary = boundary.view(B, T, H, W)
    else:
        boundary = boundary.squeeze(1)

    return boundary

def boundary_to_patch_weights(boundary, num_patches):
    """
    boundary: [B, T, H, W]
    returns: patch_weights [B, T, N]
    """
    B, T, H, W = boundary.shape
    P = int(num_patches ** 0.5)
    ph, pw = H // P, W // P

    boundary = boundary.view(B, T, P, ph, P, pw)
    patch_weights = boundary.mean(dim=(3, 5))  # [B,T,P,P]
    return patch_weights.flatten(2)            # [B,T,N]

# def patch_xray_loss(
#     patch_scores,   # [B, T, N] logits
#     label,          # [B] (0=real, 1=fake)
#     mask,           # [B, T, H, W] or [B, T, H, W, 1]
#     cfg,
#     extract_boundary_fn,
#     boundary_to_patch_weights_fn,
# ):
#     """
#     Patch_Xray loss (non-temporal)

#     Encourages strong, sparse patch responses near face boundaries
#     ONLY for fake samples.

#     Returns:
#         scalar loss tensor
#     """

#     alpha = cfg.get("alpha", 1.0)
#     k = cfg.get("k", 5)

#     device = patch_scores.device
#     B, T, N = patch_scores.shape

#     # --------------------------------------------------
#     # Boundary → patch weights
#     # --------------------------------------------------
#     boundary = extract_boundary_fn(mask)          # [B,T,H,W]
#     patch_weights = boundary_to_patch_weights_fn(
#         boundary, N
#     )                                             # [B,T,N]

#     patch_weights = patch_weights / (
#         patch_weights.amax(dim=(1, 2), keepdim=True) + 1e-6
#     )
#     patch_weights = (patch_weights > 0).float() 
#     # breakpoint()
#     # --------------------------------------------------
#     # Fake-only selection
#     # --------------------------------------------------
#     fake_mask = (label == 1)
#     if not fake_mask.any():
#         return torch.zeros((), device=device)

#     ps = patch_scores[fake_mask]      # [Bf,T,N]
#     pw = patch_weights[fake_mask]     # [Bf,T,N]

#     # --------------------------------------------------
#     # Patch saliency (no temporal)
#     # --------------------------------------------------
#     # Option 1: absolute activation
#     patch_energy = torch.relu(ps)     # [Bf,T,N]

#     # Option 2 (alternative): deviation from frame mean
#     # frame_mean = patch_energy.mean(dim=-1, keepdim=True)
#     # patch_energy = torch.abs(patch_energy - frame_mean)

#     # --------------------------------------------------
#     # Boundary-weighted saliency
#     # --------------------------------------------------
#     weighted_energy = patch_energy * pw   # [Bf,T,N]

#     # --------------------------------------------------
#     # MIL Top-K over patches & frames
#     # --------------------------------------------------
#     flat_energy = weighted_energy.view(weighted_energy.size(0), -1)

#     topk_vals, _ = torch.topk(
#         flat_energy,
#         k=min(k, flat_energy.shape[1]),
#         dim=1
#     )

#     loss = topk_vals.mean()

#     return alpha * loss

def patch_xray_loss(
    patch_scores,   # [B, T, N] logits
    label,          # [B] (0=real, 1=fake)
    mask,           # [B, T, H, W] or [B, T, H, W, 1]
    cfg,
    extract_boundary_fn,
    boundary_to_patch_weights_fn,
):
    """
    Patch_Xray loss (per-frame patch MIL)

    Enforces: for a fake video,
    EACH frame must have at least K boundary patches
    with strong fake evidence.
    """

    alpha = cfg.get("alpha", 1.0)
    k = cfg.get("k", 5)   # patches per frame

    device = patch_scores.device
    B, T, N = patch_scores.shape

    # --------------------------------------------------
    # Boundary → binary patch mask
    # --------------------------------------------------
    boundary = extract_boundary_fn(mask)          # [B,T,H,W]
    patch_mask = boundary_to_patch_weights_fn(
        boundary, N
    )                                             # [B,T,N]

    patch_mask = (patch_mask > 0).float()         # binary

    # --------------------------------------------------
    # Fake-only selection
    # --------------------------------------------------
    fake_mask = (label == 1)
    if not fake_mask.any():
        return torch.zeros((), device=device)

    ps = patch_scores[fake_mask]   # [Bf,T,N]
    pm = patch_mask[fake_mask]     # [Bf,T,N]

    # --------------------------------------------------
    # Patch saliency
    # --------------------------------------------------
    patch_energy = torch.relu(ps)          # [Bf,T,N]
    boundary_energy = patch_energy * pm    # [Bf,T,N]

    # --------------------------------------------------
    # Per-frame Top-K boundary patches
    # --------------------------------------------------
    # shape: [Bf, T, k]
    topk_vals, _ = torch.topk(
        boundary_energy,
        k=min(k, boundary_energy.shape[-1]),
        dim=-1
    )
    target = torch.ones_like(topk_vals)
    loss = F.binary_cross_entropy_with_logits(topk_vals, target)

    return alpha * loss



def Patch_frame_MIL_ranking_loss(patch_scores, label, cfg=None,margin=1.0):
    """
    Patch-frame MIL ranking loss with random real-fake frame pairs.
    
    Args:
        patch_scores: Tensor of shape [B, T, N], logits for each patch per frame
        label: Tensor of shape [B], 0=real, 1=fake
        cfg: dict of configurations:
            - use_sigmoid: bool, apply sigmoid to logits (default True)
            - margin: float, ranking margin (default 1.0)
            - random_pairs: int, number of random real-fake frame pairs to sample (default 10)
    
    Returns:
        Scalar loss tensor
    """
    B, T, N = patch_scores.shape
    device = patch_scores.device
    
    # Default configs
    if cfg is None:
        cfg = {}
    use_sigmoid = cfg.get('use_sigmoid', True)
    num_pairs = cfg.get('random_pairs', 500)
    
    # Apply sigmoid if needed
    if use_sigmoid:
        patch_scores = torch.sigmoid(patch_scores)
    
    # Compute max patch score per frame (MIL assumption)
    frame_scores, _ = patch_scores.max(dim=2)  # [B, T]

    # Separate real and fake videos
    real_mask = (label == 0)
    fake_mask = (label == 1)
    
    if real_mask.sum() == 0 or fake_mask.sum() == 0:
        # No real or fake videos in batch, return 0 loss
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Extract real and fake frames
    real_frames = frame_scores[real_mask]  # [B_real, T]
    fake_frames = frame_scores[fake_mask]  # [B_fake, T]
    
    # Flatten frames for random selection
    real_frames = real_frames.view(-1)  # [B_real * T]
    fake_frames = fake_frames.view(-1)  # [B_fake * T]
    
    # Sample random pairs
    num_real = real_frames.size(0)
    num_fake = fake_frames.size(0)
    num_samples = min(num_pairs, num_real, num_fake)
    
    real_indices = torch.randperm(num_real, device=device)[:num_samples]
    fake_indices = torch.randperm(num_fake, device=device)[:num_samples]
    
    real_samples = real_frames[real_indices]  # [num_samples]
    fake_samples = fake_frames[fake_indices]  # [num_samples]
    
    # Compute variances
    real_var = real_samples.var(unbiased=False)
    fake_var = fake_samples.var(unbiased=False)
    
    # Compute ranking loss for this set of pairs
    ranking_loss = F.relu(margin - (fake_samples.mean() - real_samples.mean())) ** 2
    
    # Total loss
    loss = real_var + fake_var + ranking_loss
    
    return loss*cfg.get('alpha', 1.0)


def Patch_frame_MIL_TopK_ranking_loss(
    patch_scores,          # [B, T, N]  (LOGITS, not probabilities)
    labels,                # [B]  (0=real, 1=fake)
    cfg=None
):
    """
    Patch-level MIL ranking loss with Top-K evidence selection.

    This loss enforces:
        fake patches > real patches
    using multiple patch evidences instead of hard max.

    Returns:
        Scalar loss
    """

    # -----------------------------
    # Config
    # -----------------------------
    if cfg is None:
        cfg = {}

    topk = cfg.get("topk", 10)                 # number of patch evidences
    margin = cfg.get("margin", 0.5)            # ranking margin
    lambda_var = cfg.get("lambda_var", 0.1)    # variance regularization weight
    num_pairs = cfg.get("num_pairs", 512)      # random ranking pairs

    device = patch_scores.device
    B, T, N = patch_scores.shape

    # -----------------------------
    # Flatten patches across time
    # -----------------------------
    # [B, T, N] → [B, T*N]
    patch_scores = patch_scores.view(B, -1)

    # -----------------------------
    # Top-K patch evidence per video
    # -----------------------------
    K = min(topk, patch_scores.size(1))
    topk_scores, _ = torch.topk(patch_scores, K, dim=1)  # [B, K]

    # -----------------------------
    # Separate real / fake videos
    # -----------------------------
    real_mask = labels == 0
    fake_mask = labels == 1

    if real_mask.sum() == 0 or fake_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    real_scores = topk_scores[real_mask]  # [Br, K]
    fake_scores = topk_scores[fake_mask]  # [Bf, K]

    # -----------------------------
    # Flatten for ranking
    # -----------------------------
    real_scores = real_scores.reshape(-1)   # [Br*K]
    fake_scores = fake_scores.reshape(-1)   # [Bf*K]

    # -----------------------------
    # Random pair sampling
    # -----------------------------
    num_samples = min(num_pairs, real_scores.numel(), fake_scores.numel())

    real_idx = torch.randperm(real_scores.numel(), device=device)[:num_samples]
    fake_idx = torch.randperm(fake_scores.numel(), device=device)[:num_samples]

    real_samples = real_scores[real_idx]
    fake_samples = fake_scores[fake_idx]

    # -----------------------------
    # Ranking loss (patch-level)
    # -----------------------------
    ranking_loss = F.relu(
        margin - (fake_samples - real_samples)
    ).mean()

    # -----------------------------
    # Variance regularization
    # -----------------------------
    var_loss = real_scores.var(unbiased=False) + fake_scores.var(unbiased=False)

    # -----------------------------
    # Final loss
    # -----------------------------
    loss = ranking_loss + lambda_var * var_loss

    return loss

def Patch_frame_MIL_ranking_all_loss(patch_scores, label, cfg=None, margin=1.0):
    """
    Fully MIL ranking: all fake frames > all real frames.
    """
    if cfg is None:
        cfg = {}
    topk_percent=cfg.get('topk_percent',1.0)
    
    margin = margin
    alpha = cfg.get('alpha', 1.0)



    B, T, N = patch_scores.shape
    device = patch_scores.device
    topk = max(1, int(N * topk_percent))

    # Top-K per frame
    topk_scores, _ = torch.topk(patch_scores, k=topk, dim=2)  # [B, T, K]
    frame_scores = topk_scores.mean(dim=2)  # [B, T]

    # Separate real/fake frames
    real_mask = (label == 0)
    fake_mask = (label == 1)
    if real_mask.sum() == 0 or fake_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    real_frames = frame_scores[real_mask].reshape(-1, 1)  # [R,1]
    fake_frames = frame_scores[fake_mask].reshape(1, -1)  # [1,F]

    # Pairwise difference: fake - real
    diff = fake_frames - real_frames  # [1,F] - [R,1] = [R,F]
    ranking_loss = F.relu(margin - diff).pow(2).mean()

    # Variance regularization
    real_var = real_frames.var(unbiased=False)
    fake_var = fake_frames.var(unbiased=False)

    loss = ranking_loss + real_var + fake_var
    return loss * alpha