import torch
import torch.nn.functional as F

def bootstrapped_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, ratio: float = 0.25, min_k: int = 1024) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    B = loss.shape[0]
    loss = loss.view(B, -1)
    N = loss.shape[1]
    k = int(N * float(ratio))
    if k < int(min_k):
        k = int(min_k)
    if k > N:
        k = N
    topk = torch.topk(loss, k, dim=1, largest=True, sorted=False).values
    return topk.mean()

def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    num = 2.0 * (prob * target).sum(dim=(2, 3)) + eps
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1.0 - (num / den)).mean()

def boot_bce_plus_dice(logits: torch.Tensor, target: torch.Tensor, ratio: float = 0.25, min_k: int = 1024) -> torch.Tensor:
    bce = bootstrapped_bce_with_logits(logits, target, ratio=ratio, min_k=min_k)
    dsc = dice_loss_from_logits(logits, target)
    return 0.5 * bce + 0.5 * dsc
