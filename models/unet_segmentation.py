import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Two consecutive convolutional layers with ReLU activation.
    Used in both encoder and decoder blocks.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    Input shape: (batch_size, in_channels, H, W)
    Output shape: (batch_size, out_channels, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net model for image segmentation.
    Input shape: (batch_size, 1, H, W) - grayscale MRI slice
    Output shape: (batch_size, 1, H, W) - segmentation mask (values in [0, 1])
    Encoder: Downsampling path extracts features.
    Decoder: Upsampling path reconstructs segmentation mask.
    """
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder (downsampling)
        # Input: (batch_size, 1, H, W)
        self.down1 = DoubleConv(1, 64)    # Output: (batch_size, 64, H, W)
        self.down2 = DoubleConv(64, 128)  # Output: (batch_size, 128, H, W)
        self.down3 = DoubleConv(128, 256) # Output: (batch_size, 256, H, W)

        # Decoder (upsampling)
        # Input: (batch_size, 256, H, W)
        self.up1 = DoubleConv(256, 128)   # Output: (batch_size, 128, H, W)
        self.up2 = DoubleConv(128, 64)    # Output: (batch_size, 64, H, W)

        # Final segmentation layer
        self.final = nn.Conv2d(64, 1, 1)  # Output: (batch_size, 1, H, W)

    def forward(self, x):
        """
        Forward pass through U-Net.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, H, W)
        Returns:
            torch.Tensor: Segmentation mask of shape (batch_size, 1, H, W)
        """
        # Encoder
        d1 = self.down1(x)   # (batch_size, 64, H, W)
        d2 = self.down2(d1)  # (batch_size, 128, H, W)
        d3 = self.down3(d2)  # (batch_size, 256, H, W)

        # Decoder
        u1 = self.up1(d3)    # (batch_size, 128, H, W)
        u2 = self.up2(u1)    # (batch_size, 64, H, W)

        # Final segmentation
        out = self.final(u2) # (batch_size, 1, H, W)

        # Sigmoid for mask output
        return torch.sigmoid(out)