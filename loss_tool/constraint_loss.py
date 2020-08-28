import contextlib
from typing import Tuple

import torch


def cons_binary_value(x: torch.Tensor) -> torch.Tensor:
    d_reshaped = x.view(x.shape[0], -1, *(1 for _ in range(x.dim() - 2)))
    x = 0
    return x


class cons_Loss:
    def __init__(
        self,  cons_list
    ):
        super(cons_Loss, self).__init__()
        self.cons_list = cons_list

    def forward(self, model, x: torch.Tensor):

        cons_value = cons_binary_value(x)
        cons_loss = 0
        return cons_loss

