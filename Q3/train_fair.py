import torch
import torch.nn.functional as F

def fairness_loss(outputs, targets, gender_labels, lambda_fair=0.5):

    ce_loss = F.cross_entropy(outputs, targets, reduction='none')

    mask_m = (gender_labels == 0)
    mask_f = (gender_labels == 1)

    loss_m = ce_loss[mask_m].mean() if mask_m.sum() > 0 else torch.tensor(0.0, device=outputs.device)
    loss_f = ce_loss[mask_f].mean() if mask_f.sum() > 0 else torch.tensor(0.0, device=outputs.device)

    gap = torch.abs(loss_m - loss_f)

    return ce_loss.mean() + lambda_fair * gap