import torch
from torchvision import transforms
import torch.nn.functional as F
import PIL

def resize_feature_map(feat, target_res):
    """
    Topology-preserving resizing for MTG features.
    feat: [B, C, H, W]
    """
    B, C, H, W = feat.shape

    if target_res == H:
        return feat

    # ----- STEP 1: integer pooling when possible -----
    if H % target_res == 0:
        k = H // target_res
        return F.avg_pool2d(feat, kernel_size=k)

    # ----- STEP 2: safe two-stage resize (recommended for 32x32) -----
    # first remove high-freq instability
    if H == 48:
        feat = F.avg_pool2d(feat, kernel_size=2)  # 48 → 24
        H = 24

    # then resize smoothly
    feat = F.interpolate(
        feat,
        size=(target_res, target_res),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    return feat


def resize_mask(mask, target_res):
    """
    Stable mask resizing.
    Avoid nearest-neighbor aliasing.
    """
    mask = F.interpolate(
        mask,
        size=(target_res, target_res),
        mode="bilinear",
        align_corners=False,
    )

    # binarize again
    mask = (mask > 0.5).float()
    return mask


def compute_masked_nssm(
    feat_gen,
    feat_ref,
    mask_ref_list,
    target_res=32,
    diag_band=0,
):
    """
    feat_gen/ref: [B, N, C]
    """

    device = feat_gen.device
    B, N, C = feat_gen.shape
    H = W = int(N ** 0.5)

    # ---------------- Mask preparation ----------------
    transform_mask = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
    ])

    if isinstance(mask_ref_list, list):
        masks = torch.stack(
            [transform_mask(m).to(device) for m in mask_ref_list]
        )
    else:
        masks = transform_mask(mask_ref_list).to(device)
        masks = masks.unsqueeze(0).repeat(B, 1, 1, 1)

    # ---------------- Normalize features ----------------
    f_g = F.normalize(feat_gen, dim=-1)
    f_r = F.normalize(feat_ref, dim=-1)

    # reshape to spatial maps
    f_g = f_g.view(B, H, W, C).permute(0, 3, 1, 2)
    f_r = f_r.view(B, H, W, C).permute(0, 3, 1, 2)
    masks = masks.view(B, 1, H, W)

    # ---------------- Topology-preserving resize ----------------
    f_g = resize_feature_map(f_g, target_res)
    f_r = resize_feature_map(f_r, target_res)
    masks = resize_mask(masks, target_res)

    # flatten again
    f_g = f_g.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
    f_r = f_r.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
    masks = masks.view(B, -1)

    # ---------------- NSSM computation ----------------
    nssm_g = torch.bmm(f_g, f_g.transpose(1, 2))
    nssm_r = torch.bmm(f_r, f_r.transpose(1, 2))

    N = nssm_g.shape[1]

    idx = torch.arange(N, device=device)
    diag_mask = torch.abs(idx[:, None] - idx[None, :]) > diag_band

    nssm_scores = []

    for b in range(B):
        m_r = masks[b]

        combined_mask = (m_r[:, None] * m_r[None, :]) > 0
        combined_mask = combined_mask & diag_mask

        if combined_mask.sum() < 2:
            nssm_scores.append(0.0)
            continue

        vec_g = nssm_g[b][combined_mask]
        vec_r = nssm_r[b][combined_mask]

        if vec_g.std() == 0 or vec_r.std() == 0:
            nssm_scores.append(0.0)
            continue

        x = vec_g - vec_g.mean()
        y = vec_r - vec_r.mean()

        score = (x * y).mean() / (x.std() * y.std() + 1e-8)
        nssm_scores.append(1 - score.item())

    return nssm_scores
