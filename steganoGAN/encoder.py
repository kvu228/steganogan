import torch
import torch.nn as nn

class SteganographyEncoder(nn.Module):
    """
    A neural network module that encodes secret data into a cover image.
    Input:
    - cover_image: (batch_size, 3, height, width) tensor
    - secret_data: (batch_size, data_channels, height, width) tensor
    Output:
    - steganographic_image: (batch_size, 3, height, width) tensor
    """

    def __init__(self, data_channels:int=1, hidden_channels:int=32):
        """
        Args:
            data_channels: Number of channels in the secret data tensor
            hidden_channels: Number of channels in hidden convolutional layers
        """
        super().__init__()
        self.version = '1'
        self.data_channels = data_channels
        self.hidden_channels = hidden_channels
        self.add_cover_image = False  # Whether to add original image to output

        # Initialize model components
        self.feature_extractor, self.processing_layers = self._build_model_components()

    def _create_conv_layer(self, input_channels, output_channels):
        """Helper to create a convolutional layer with standard parameters."""
        return nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=1
        )

    def _build_model_components(self):
        """Constructs the neural network components."""

        # Feature extraction from cover image
        feature_extractor = nn.Sequential(
            self._create_conv_layer(3, self.hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_channels),
        )

        # Processing stages that combine features and secret data
        processing_layers = nn.Sequential(
            self._create_conv_layer(self.hidden_channels + self.data_channels, self.hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_channels),
            self._create_conv_layer(self.hidden_channels, self.hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_channels),
            self._create_conv_layer(self.hidden_channels, 3),
            nn.Tanh(),
        )

        return feature_extractor, processing_layers

    def forward(self, cover_image, secret_data):
        """
        Embeds secret data into a cover image.

        Args:
            cover_image: Original image to hide data in
            secret_data: Data to be hidden in the image

        Returns:
            steganographic_image: Image with hidden data
        """
        # Extract initial features from cover image
        features = self.feature_extractor(cover_image)

        # Concatenate features with secret data once
        combined_input = torch.cat([features, secret_data], dim=1)

        # Process through all layers at once
        output = self.processing_layers(combined_input)

        # Optional: Add original image to output
        if self.add_cover_image:
            return cover_image + output

        return output

    def upgrade_legacy(self):
        """Maintains compatibility with older model versions."""
        if not hasattr(self, 'version'):
            self.version = '1'
