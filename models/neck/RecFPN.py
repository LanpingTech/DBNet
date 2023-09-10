import torch
import torch.nn as nn
import torch.nn.functional as F

class RecFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_recursions=3):
        super(RecFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels * len(in_channels)
        self.num_recursions = num_recursions

        # Lateral connections
        self.lateral_convs = nn.ModuleList([nn.Conv2d(self.in_channels[i], out_channels, 1)
                                            for i in range(len(self.in_channels))])

        # Top-down convolution
        self.top_down_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # x is a list of feature maps from the backbone network, from low-level to high-level

        # First pass: standard FPN
        out = []
        prev_top_down = None
        for l in reversed(range(len(x))):
            lateral = self.lateral_convs[l](x[l])
            if prev_top_down is not None:
                top_down = F.interpolate(prev_top_down, size=lateral.shape[2:], mode="nearest") + lateral
            else:
                top_down = lateral
            out.insert(0, top_down)
            prev_top_down = self.top_down_conv(top_down)

        # Recursions
        for r in range(self.num_recursions - 1):
            new_out = []
            prev_top_down = None
            for l in reversed(range(len(x))):
                lateral = self.lateral_convs[l](x[l])
                if prev_top_down is not None:
                    top_down = F.interpolate(prev_top_down, size=lateral.shape[2:], mode="nearest") + lateral
                else:
                    top_down = lateral
                new_out.insert(0, top_down)
                prev_top_down = self.top_down_conv(top_down)

            out = [out[i] + new_out[i] for i in range(len(out))]

        out = self._upsample_cat(*out)
        return out
    
    def _upsample_cat(self, p3, p4, p5, p6):
        h, w = p3.size()[2:]
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        p6 = F.interpolate(p6, size=(h, w))
        return torch.cat([p3, p4, p5, p6], dim=1)

