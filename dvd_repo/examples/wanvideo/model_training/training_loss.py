import torch
import torch.nn as nn


class GradientLoss3DSeparate(nn.Module):

    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def forward(self, x, target):
        _, _, T, H, W = x.shape
        assert x.shape == target.shape
        if T > 1:
            dt_x = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
            dt_y = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
            grad_t = dt_x - dt_y

        dh_x = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dh_y = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        grad_h = dh_x - dh_y

        dw_x = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        dw_y = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        grad_w = dw_x - dw_y

        if self.p == 1:
            if T > 1:
                loss_t = torch.mean(torch.abs(grad_t))
            else:
                # loss_t = torch.tensor(0, dtype=x.dtype, device=x.device)
                loss_t = 0.0
            loss_h = torch.mean(torch.abs(grad_h))
            loss_w = torch.mean(torch.abs(grad_w))

        elif self.p == 2:
            if T > 1:
                loss_t = torch.mean(grad_t ** 2)
            else:
                # loss_t = torch.tensor(0, dtype=x.dtype, device=x.device)
                loss_t = 0.0

            loss_h = torch.mean(grad_h ** 2)
            loss_w = torch.mean(grad_w ** 2)

        else:
            raise ValueError("p should be 1 or 2")

        return loss_t, loss_h, loss_w
