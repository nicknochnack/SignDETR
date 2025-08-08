import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import sys 
from colorama import Fore 
from torchinfo import summary
import sys 
import math


def pos_encoding(seq_length, dim_size):
    p = torch.zeros((seq_length, dim_size))
    for k in range(seq_length):
        for i in range(int(dim_size / 2)):
            p[k, 2 * i] = torch.sin(torch.tensor(k / (10000 ** (2 * i / dim_size))))
            p[k, 2 * i + 1] = torch.cos(torch.tensor(k / (10000 ** (2 * i / dim_size))))
    return p


def _get_1d_sincos_pos_embed(length: int, dim: int, temperature: float = 10000.0, device=None):
    assert dim % 2 == 0
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # (L,1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(temperature) / dim)
    )  # (dim/2)
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L, dim)


def build_2d_sincos_position_embedding(height: int, width: int, dim: int, device=None):
    """Create 2D sine-cos positional encoding of shape (1, H*W, dim).
    Half dims for Y, half for X.
    """
    assert dim % 2 == 0, "positional dim must be even"
    dim_half = dim // 2
    pe_y = _get_1d_sincos_pos_embed(height, dim_half, device=device)  # (H, dim/2)
    pe_x = _get_1d_sincos_pos_embed(width, dim_half, device=device)   # (W, dim/2)
    # Combine to (H, W, dim)
    pos = torch.zeros(height, width, dim, device=device, dtype=torch.float32)
    pos[:, :, :dim_half] = pe_y[:, None, :].expand(-1, width, -1)
    pos[:, :, dim_half:] = pe_x[None, :, :].expand(height, -1, -1)
    pos = pos.view(1, height * width, dim)  # (1, H*W, dim)
    return pos


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=1, num_decoder_layers=1, num_queries=25):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True, dropout=0.1)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # number of object queries
        self.num_queries = num_queries
        # learned query positional encodings
        self.query_pos = nn.Parameter(torch.randn(self.num_queries, hidden_dim))

        # normalizations
        self.norm_src = nn.LayerNorm(hidden_dim)
        self.norm_tgt = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        b, c, H_in, W_in = inputs.shape
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to hidden_dim feature planes for the transformer
        feat = self.conv(x)  # (b, d, Hf, Wf)
        bsz, d_model, Hf, Wf = feat.shape
        src = feat.flatten(2).permute(0, 2, 1)  # (b, Hf*Wf, d)

        # dynamic 2D sine-cos positional encoding
        pos = build_2d_sincos_position_embedding(Hf, Wf, d_model, device=feat.device)  # (1, Hf*Wf, d)
        src = self.norm_src(src + pos)

        # decoder target: zero content + learned query positional encodings
        tgt = torch.zeros(bsz, self.num_queries, d_model, device=feat.device)
        query_pos = self.query_pos.unsqueeze(0).expand(bsz, -1, -1)
        tgt = self.norm_tgt(tgt + query_pos)

        # propagate through the transformer
        hs = self.transformer(src=src, tgt=tgt)  # (b, num_queries, d)

        # finally project transformer outputs to class labels and bounding boxes
        return {
            'pred_logits': self.linear_class(hs),
            'pred_boxes': self.linear_bbox(hs).sigmoid()
        }


if __name__ == '__main__': 
    model = DETR(num_classes=3)
    summary(model, (5,3,224,224))