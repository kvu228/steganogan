import torch
import torch.nn as nn

class SteganographyDecoder(nn.Module):
    """Neural network module that decodes hidden data from steganographic images.

    Args:
        secret_data_channels (int): Number of channels in the hidden data tensor
        hidden_channels (int): Number of channels in hidden convolutional layers

    Input:
        stego_image (Tensor): (batch_size, 3, height, width) steganographic image
    Output:
        decoded_data (Tensor): (batch_size, secret_data_channels, height, width) reconstructed data
    """

    def __init__(self, secret_data_channels: int = 1, hidden_channels: int = 32):
        super().__init__()
        self.version = '1'
        self.secret_data_channels = secret_data_channels
        self.hidden_channels = hidden_channels
        self.decoder_layers = self._build_decoder()

    def _create_conv_layer(self, in_channels: int, out_channels: int) -> nn.Conv2d:
        """Helper to create a convolutional layer with fixed parameters.

        Args:
            in_channels: Input channels for the convolution
            out_channels: Output channels for the convolution

        Returns:
            nn.Conv2d: Configured convolutional layer
        """
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def _build_decoder(self) -> nn.Sequential:
        """Constructs the decoder architecture.

        Returns:
            nn.Sequential: Sequence of layers forming the decoder
        """
        layers = nn.ModuleList()
        # First layer: 3 -> hidden_channels
        layers.append(nn.Sequential(
            self._create_conv_layer(3, self.hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_channels)
        ))

        # Second layer: hidden_channels -> hidden_channels
        layers.append(nn.Sequential(
            self._create_conv_layer(self.hidden_channels, self.hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_channels)
        ))
        # Third layer: hidden_channels*2 -> hidden_channels
        layers.append(nn.Sequential(
            self._create_conv_layer(self.hidden_channels*2, self.hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_channels)
        ))
        # Final layer: hidden_channels*3 -> secret_data_channels
        layers.append(
            self._create_conv_layer(self.hidden_channels*3, self.secret_data_channels)
        )

        return layers

    def forward(self, stego_image: torch.Tensor) -> torch.Tensor:
        """Processes a steganographic image to recover hidden data.

        Args:
            stego_image: Image containing hidden data

        Returns:
            torch.Tensor: Reconstructed data tensor
        """
        # Process through first layer
        features = self.decoder_layers[0](stego_image)
        feature_maps = [features]

        # Process through second layer
        features = self.decoder_layers[1](features)
        feature_maps.append(features)

        # Process through third layer with concatenated features
        combined = torch.cat(feature_maps, dim=1)  # Concatenate along channel dimension
        features = self.decoder_layers[2](combined)
        feature_maps.append(features)

        # Process through final layer with all features concatenated
        combined = torch.cat(feature_maps, dim=1)
        return self.decoder_layers[3](combined)

    def upgrade_legacy(self) -> None:
        """Converts legacy model versions to work with current implementation."""
        if not hasattr(self, 'version'):
            # Handle version conversion logic here
            self.version = '1'
