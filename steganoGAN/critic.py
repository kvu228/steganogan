import torch
import torch.nn as nn

class SteganographyDiscriminator(nn.Module):
    """Neural network module that discriminates between real and steganographic images.

    Args:
        hidden_channels (int): Number of channels in hidden convolutional layers

    Input:
        image (Tensor): (batch_size, 3, height, width) input image
    Output:
        authenticity_score (Tensor): (batch_size, 1) probability of being real
    """

    def __init__(self, hidden_channels: int=32):
        super().__init__()
        self.version = '1'
        self.hidden_channels = hidden_channels
        self.discriminator_layers = self._build_discriminator()

    def _create_conv_layer(self, in_channels: int, out_channels: int) -> nn.Conv2d:
        """Constructs a convolutional layer with fixed kernel size and padding.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1  # Added to maintain spatial dimensions
        )

    def _build_discriminator(self) -> nn.Sequential:
        """Constructs the sequential layers for the discriminator."""
        return nn.Sequential(
            self._create_conv_layer(3, self.hidden_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(self.hidden_channels),

            self._create_conv_layer(self.hidden_channels, self.hidden_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(self.hidden_channels),

            self._create_conv_layer(self.hidden_channels, self.hidden_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(self.hidden_channels),

            self._create_conv_layer(self.hidden_channels, 1)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Processes an image to produce an authenticity score.

        Args:
            image: Input image tensor (cover or stego)

        Returns:
            Tensor: Scalar score per image in batch (higher = more real)
        """
        features = self.discriminator_layers(image)

        # Global average pooling
        batch_size = features.shape[0]
        return features.view(batch_size, -1).mean(dim=1, keepdim=True)

    def upgrade_legacy(self) -> None:
        """Maintains compatibility with older model versions."""
        if not hasattr(self, 'version'):
            self.discriminator_layers = self._build_discriminator()
            self.version = '1'
