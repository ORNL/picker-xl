import importlib.resources

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


#
def arrival_probabilities_to_indices(
    probabilities: torch.Tensor,
    min_itp_probability: float = 0.1,
    min_its_probability: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert picker network outputs (probabilities) into pred indices for
    wave arrival prediction.
    """
    max_values, max_indices = torch.max(probabilities, dim=-1)

    itp_indices = max_indices[:, 1]
    itp_indices[max_values[:, 1] < min_itp_probability] = -9999
    its_indices = max_indices[:, 2]
    its_indices[max_values[:, 2] < min_its_probability] = -9999
    return itp_indices, its_indices


# PNet model was modified from https://github.com/seisbench/seisbench/blob/main/seisbench/models/phasenet.py, May 2024
class PNet(L.LightningModule):
    def __init__(
        self,
        in_dim: int = 5700,
        in_chans: int = 3,
        out_chans: int = 3,
        filters_root=32,
        kernel_size=7,
        depth=5,
        stride=4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.out_dim = in_dim
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = stride

        self.activation = torch.relu
        #
        self.inc = nn.Conv1d(
            self.in_chans, self.filters_root, self.kernel_size, padding="same"
        )
        self.in_bn = nn.BatchNorm1d(self.filters_root, eps=1e-5)

        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root
        for i in range(self.depth):
            filters = int(2**i * self.filters_root)
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-5)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                if i in range(1, self.depth - 1):
                    padding = 0
                else:
                    padding = self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    self.kernel_size,
                    self.stride,
                    padding=padding,
                    bias=False,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-5)
            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        for i in range(self.depth - 1):
            filters = int(2 ** (self.depth - 2 - i) * self.filters_root)
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-5)
            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-5)

            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        self.out = nn.Conv1d(last_filters, self.out_chans, 1, padding="same")
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.in_bn(self.inc(x)))

        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)

                x = self.activation(bn2(conv_down(x)))

        for i, ((conv_up, bn1, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))
            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        x = self.out(x)
        x = self.softmax(x)

        return x

    @staticmethod
    def _merge_skip(skip, x):
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]

        return torch.cat([skip, x_resize], dim=1)


#
class Picker:
    def __init__(self):
        self.model = PNet()
        with importlib.resources.files("pickerxl").joinpath("model").joinpath(
            "large.ckpt"
        ) as fid:
            checkpoint_data = torch.load(fid)
        self.model.load_state_dict(checkpoint_data)
        self.model.eval()

    def predict_probability(self, data, eps=1e-10):
        data = np.array(data).astype(np.float32)
        data = data / (np.max(np.abs(data), axis=(1,2), keepdims=True) + eps)
        data = torch.Tensor(data)
        preds = self.model(data)
        return preds

    def predict_arrivals(self, data):
        preds = self.predict_probability(data)
        p_index, s_index = arrival_probabilities_to_indices(preds)
        p_index = p_index.detach().cpu().numpy().astype(float)
        s_index = s_index.detach().cpu().numpy().astype(float)
        p_index[p_index < 0] = np.nan
        s_index[s_index < 0] = np.nan
        return p_index, s_index
