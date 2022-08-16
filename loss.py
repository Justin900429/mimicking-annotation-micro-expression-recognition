import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSupCoL(nn.Module):
    """
    Credit to: Guillaume Erhard
    Adapted from: https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
    """

    def __init__(self, temperature=0.1):
        super(WeightedSupCoL, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        if len(targets) == 1:
            return None
        device = targets.device

        mask_anchor_out = (1 - torch.eye(projections.shape[0])).to(device)
        cardinality_per_samples = targets.sum(dim=-1)

        logits = torch.div(torch.matmul(projections, projections.T), self.temperature)

        exp_logits = torch.exp(logits) + 1e-5
        denominator = (exp_logits * mask_anchor_out).sum(dim=1, keepdim=True)
        log_prob = -torch.log(exp_logits / denominator)
        supervised_contrastive_loss_per_sample = (log_prob * targets).sum(
            dim=1
        ) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss


class InfoNCE(nn.Module):
    def __init__(self, device, temperature=0.1):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, x_1, x_2):
        similarity = torch.div(torch.matmul(x_1, x_2.T), self.temperature)
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        mask_anchor_out = (1 - torch.eye(logits.shape[0])).to(self.device)
        logits_mask = logits * mask_anchor_out
        positive = torch.diag(logits)

        loss = -torch.log(torch.exp(positive) / torch.exp(logits_mask).sum(dim=-1))
        return loss.mean()


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing_rate: float = 0.1, mode: str = "mean"):
        super(LabelSmoothing, self).__init__()
        assert mode in ["mean", "sum", "none"], (
            f"Not supported mode for {mode}." f" Should be in ['mean', 'sum', 'none']"
        )
        self.mode = mode
        self.smoothing_rate = smoothing_rate

    def forward(self, predict, target):
        # Shape of predict: (batch_size, num_classes)
        log_probs = F.log_softmax(predict, dim=-1)

        # Vanilla cross entropy loss
        NLL_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1))
        NLL_loss = NLL_loss.squeeze(1)

        # Add the smoothing loss to it
        smoothing_loss = -log_probs.mean(dim=-1)
        total_loss = (
            1 - self.smoothing_rate
        ) * NLL_loss + self.smoothing_rate * smoothing_loss

        if self.mode == "mean":
            return total_loss.mean()
        elif self.mode == "sum":
            return total_loss.sum()
        else:
            return total_loss


def adjust_learning_rate(optimizer, epoch, args):
    """
    :param optimizer: torch.optim
    :param epoch: int
    :param mode: str
    :param args: argparse.Namespace
    :return: None
    """
    lr = args.con_lr
    n_epochs = args.con_epochs

    eta_min = lr * (args.con_lr_decay**3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    import torch.nn.functional as F

    loss = InfoNCE("cpu")
    test = F.normalize(torch.randn(5, 3), dim=1, p=2)
    com = loss(test, test)
