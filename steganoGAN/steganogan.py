import gc
import inspect
import json
import os
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import torch
from imageio.v2 import imread, imwrite
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from tqdm import tqdm

from steganoGAN.utils import (
    bits_to_bytearray,
    recover_and_decompress_text,
    structural_similarity_index as ssim,
    encode_text_to_bits
)

# Default paths for logs and saved models
DEFAULT_LOG_DIRECTORY = Path(__file__).parent / 'train'

# Metrics tracked during training
TRAINING_METRICS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',
    'val.cover_score',
    'val.generated_score',
    'val.ssim',
    'val.psnr',
    'val.bpp',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.generated_score',
]


class SteganoGAN:
    """
    A GAN-based steganography model for hiding and extracting messages in images.

    This class implements the entire steganography pipeline including:
    - Message encoding in cover images
    - Message decoding from steganographic images
    - Model training with adversarial techniques
    - Model saving and loading
    """

    def __init__(self, data_depth, encoder, decoder, critic,
                 use_cuda=False, verbose=False, log_dir=None, **kwargs):
        """
        Initialize the SteganoGAN model with neural network components.

        Args:
            data_depth: Bit depth for the hidden message
            encoder: Network for embedding messages in images
            decoder: Network for extracting messages from images
            critic: Discriminator network for adversarial training
            use_cuda: Whether to use GPU acceleration when available
            verbose: Whether to print detailed progress information
            log_dir: Directory for saving training logs and samples
            **kwargs: Additional parameters passed to component constructors
        """
        self.verbose = verbose
        self.data_depth = data_depth

        # Initialize neural network components
        kwargs['data_depth'] = data_depth
        self.encoder = self._initialize_component(encoder, kwargs)
        self.decoder = self._initialize_component(decoder, kwargs)
        self.critic = self._initialize_component(critic, kwargs)

        # Set computation device (CPU/GPU)
        self.set_device(use_cuda)

        # Initialize optimizers (will be created during training)
        self.critic_optimizer = None
        self.decoder_optimizer = None

        # Training metrics and history
        self.current_metrics = None
        self.training_history = []
        self.completed_epochs = 0 if kwargs.get('completed_epochs') is None else kwargs['completed_epochs']

        # Set up logging directory for training artifacts
        self.log_dir = log_dir
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(exist_ok=True)
            self.samples_dir = log_path / 'samples'
            self.samples_dir.mkdir(exist_ok=True)

    def _initialize_component(self, class_or_instance, kwargs):
        """
        Initialize a neural network component if it's a class, or return the instance.

        Args:
            class_or_instance: Neural network class or instance
            kwargs: Constructor arguments for class initialization

        Returns:
            Initialized neural network component
        """
        if not inspect.isclass(class_or_instance):
            return class_or_instance

        # Extract relevant arguments for this component
        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec if arg in kwargs}

        return class_or_instance(**init_args)

    def set_device(self, use_cuda=True):
        """
        Configure the computation device (CPU or CUDA) for all model components.

        Args:
            use_cuda: Whether to attempt using CUDA if available
        """
        if use_cuda and torch.cuda.is_available():
            self.cuda_enabled = True
            self.device = torch.device('cuda')
        else:
            self.cuda_enabled = False
            self.device = torch.device('cpu')

        if self.verbose:
            if not self.cuda_enabled:
                print('CUDA is not available. Defaulting to CPU device')
            else:
                print(f'Using {self.device} device')

        # Move all components to the selected device
        print(f'Moving components to {self.device}...')
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)

    def _generate_random_payload(self, cover_image):
        """
        Generate random binary data to hide inside the cover image.

        Args:
            cover_image: Cover image tensor

        Returns:
            Random binary payload matching cover image dimensions
        """
        batch_size, _, height, width = cover_image.size()
        return torch.zeros(
            (batch_size, self.data_depth, height, width),
            device=self.device
        ).random_(0, 2)

    def _encode_decode(self, cover_image, quantize=False):
        """
        Encode random data into a cover image and then decode it.

        Args:
            cover_image: Image tensor to use as cover
            quantize: Whether to quantize the generated image (simulates image saving)

        Returns:
            stego_image: Generated steganographic image
            message_payload: Random data encoded in the image
            decoded_payload: Data decoded from the generated image
        """
        message_payload = self._generate_random_payload(cover_image)
        stego_image = self.encoder(cover_image, message_payload)

        # Simulate image saving/loading through quantization
        if quantize:
            stego_image = (255.0 * (stego_image + 1.0) / 2.0).long()
            stego_image = 2.0 * stego_image.float() / 255.0 - 1.0

        decoded_payload = self.decoder(stego_image)

        return stego_image, message_payload, decoded_payload

    def _evaluate_image(self, image):
        """
        Evaluate image authenticity using the critic network.

        Args:
            image: Image tensor to evaluate

        Returns:
            Authenticity score (higher = more authentic)
        """
        return torch.mean(self.critic(image))

    def _initialize_optimizers(self):
        """
        Create optimizers for the critic and encoder-decoder networks.

        Returns:
            critic_optimizer: Optimizer for the critic network
            encoder_decoder_optimizer: Optimizer for the encoder and decoder networks
        """
        encoder_decoder_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        critic_optimizer = Adam(self.critic.parameters(), lr=1e-4)
        encoder_decoder_optimizer = Adam(encoder_decoder_params, lr=1e-4)
        return critic_optimizer, encoder_decoder_optimizer

    def _train_discriminator(self, train_loader, metrics):
        """
        Train the critic network to distinguish between real and steganographic images.

        Args:
            train_loader: DataLoader with training images
            metrics: Dictionary to collect training metrics
        """

        for batch in tqdm(train_loader, disable=not self.verbose):
            gc.collect()
            # Handle different batch structures
            if isinstance(batch, tuple):
                if len(batch) >= 2:
                    # If batch has at least 2 elements as (images, _)
                    cover_images = batch[0]
                else:
                    # If batch is a single-element tuple
                    cover_images = batch[0]
            else:
                # If batch is not a tuple (direct tensor)
                cover_images = batch
            cover_images = cover_images.to(self.device)
            # Generate steganographic images
            message_payload = self._generate_random_payload(cover_images)
            stego_images = self.encoder(cover_images, message_payload)

            # Calculate authenticity scores
            real_image_score = self._evaluate_image(cover_images)
            fake_image_score = self._evaluate_image(stego_images)

            # Train critic to maximize difference between real and fake
            self.critic_optimizer.zero_grad()
            (real_image_score - fake_image_score).backward(retain_graph=False)
            self.critic_optimizer.step()

            # Weight clipping for WGAN-style training
            for param in self.critic.parameters():
                param.data.clamp_(-0.1, 0.1)

            # Record metrics
            metrics['train.cover_score'].append(real_image_score.item())
            metrics['train.generated_score'].append(fake_image_score.item())

    def _train_encoder_decoder(self, train_loader, metrics):
        """
        Train the encoder and decoder networks for message hiding and recovery.

        Args:
            train_loader: DataLoader with training images
            metrics: Dictionary to collect training metrics
        """
        for batch in tqdm(train_loader, disable=not self.verbose):
            gc.collect()
            # Handle different batch structures
            if isinstance(batch, tuple):
                if len(batch) >= 2:
                    # If batch has at least 2 elements as (images, _)
                    cover_images = batch[0]
                else:
                    # If batch is a single-element tuple
                    cover_images = batch[0]
            else:
                # If batch is not a tuple (direct tensor)
                cover_images = batch
            cover_images = cover_images.to(self.device)

            # Encode and decode
            stego_images, message_payload, decoded_payload = self._encode_decode(cover_images)

            # Calculate losses
            image_quality_loss, message_decoding_loss, decoding_accuracy = self._calculate_losses(
                cover_images, stego_images, message_payload, decoded_payload
            )

            # Calculate adversarial component
            adversarial_score = self._evaluate_image(stego_images)

            # Update encoder and decoder
            self.decoder_optimizer.zero_grad()
            combined_loss = 100.0 * image_quality_loss + message_decoding_loss + adversarial_score
            combined_loss.backward()
            self.decoder_optimizer.step()

            # Record metrics
            metrics['train.encoder_mse'].append(image_quality_loss.item())
            metrics['train.decoder_loss'].append(message_decoding_loss.item())
            metrics['train.decoder_acc'].append(decoding_accuracy.item())

    def _calculate_losses(self, cover_image, stego_image, original_payload, decoded_payload):
        """
        Calculate various loss components for the steganography system.

        Args:
            cover_image: Original cover image
            stego_image: Generated steganographic image
            original_payload: Original message payload
            decoded_payload: Decoded message payload

        Returns:
            image_quality_loss: MSE loss between cover and stego images
            message_decoding_loss: Binary cross-entropy for message decoding
            decoding_accuracy: Bit accuracy of decoded message
        """
        image_quality_loss = mse_loss(stego_image, cover_image)
        message_decoding_loss = binary_cross_entropy_with_logits(decoded_payload, original_payload)

        # Calculate bit accuracy
        correct_bits = (decoded_payload >= 0.0).eq(original_payload >= 0.5).sum().float()
        total_bits = original_payload.numel()
        decoding_accuracy = correct_bits / total_bits

        return image_quality_loss, message_decoding_loss, decoding_accuracy

    def _validate(self, validation_loader, metrics):
        """
        Validate model performance on the validation dataset.

        Args:
            validation_loader: DataLoader with validation images
            metrics: Dictionary to collect validation metrics
        """
        for batch in tqdm(validation_loader, disable=not self.verbose):
            gc.collect()
            # Handle different batch structures
            if isinstance(batch, tuple):
                if len(batch) >= 2:
                    # If batch has at least 2 elements as (images, _)
                    cover_images = batch[0]
                else:
                    # If batch is a single-element tuple
                    cover_images = batch[0]
            else:
                # If batch is not a tuple (direct tensor)
                cover_images = batch
            cover_images = cover_images.to(self.device)

            # Encode and decode with quantization (simulating real usage)
            stego_images, message_payload, decoded_payload = self._encode_decode(
                cover_images, quantize=True
            )

            # Calculate losses and scores
            image_quality_loss, message_decoding_loss, decoding_accuracy = self._calculate_losses(
                cover_images, stego_images, message_payload, decoded_payload
            )
            fake_image_score = self._evaluate_image(stego_images)
            real_image_score = self._evaluate_image(cover_images)

            # Record comprehensive metrics
            metrics['val.encoder_mse'].append(image_quality_loss.item())
            metrics['val.decoder_loss'].append(message_decoding_loss.item())
            metrics['val.decoder_acc'].append(decoding_accuracy.item())
            metrics['val.cover_score'].append(real_image_score.item())
            metrics['val.generated_score'].append(fake_image_score.item())
            metrics['val.ssim'].append(ssim(cover_images, stego_images).item())

            # Calculate PSNR and bits-per-pixel
            psnr = 10 * torch.log10(4 / image_quality_loss).item()
            bpp = self.data_depth * (2 * decoding_accuracy.item() - 1)

            metrics['val.psnr'].append(psnr)
            metrics['val.bpp'].append(bpp)

    def _save_sample_images(self, samples_dir, cover_images, epoch):
        """
        Save sample cover and steganographic images during training.

        Args:
            samples_dir: Directory to save samples
            cover_images: Batch of cover images
            epoch: Current training epoch
        """
        cover_images = cover_images.to(self.device)
        stego_images, _, _ = self._encode_decode(cover_images)

        num_samples = stego_images.size(0)
        for sample_idx in range(num_samples):
            # Save original cover image
            cover_path = samples_dir / f'{sample_idx}.cover.png'
            cover_image = (cover_images[sample_idx].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
            imageio.imwrite(str(cover_path), (255.0 * cover_image).astype('uint8'))

            # Save steganographic image
            stego_path = samples_dir / f'{sample_idx}.generated-{epoch:02d}.png'
            stego_image = stego_images[sample_idx].clamp(-1.0, 1.0).permute(1, 2, 0)
            stego_image = (stego_image.detach().cpu().numpy() + 1.0) / 2.0
            imageio.imwrite(str(stego_path), (255.0 * stego_image).astype('uint8'))

    def fit(self, train_loader, validation_loader, epochs=5):
        """
        Train the SteganoGAN model.

        Args:
            train_loader: DataLoader with training images
            validation_loader: DataLoader with validation images
            epochs: Number of training epochs
        """
        # Initialize optimizers if not already done
        if self.critic_optimizer is None:
            self.critic_optimizer, self.decoder_optimizer = self._initialize_optimizers()

        # Get a sample batch for generating examples
        if self.log_dir:
            sample_batch = next(iter(validation_loader))[0]

        # Training loop
        total_epochs = self.completed_epochs + epochs
        for epoch in range(1, epochs + 1):
            # Update epoch counter
            self.completed_epochs += 1
            current_epoch = self.completed_epochs

            if self.verbose:
                print(f'Epoch {current_epoch}/{total_epochs}')

            # Initialize metrics collection
            epoch_metrics = {field: [] for field in TRAINING_METRICS}

            # Training phase
            self._train_discriminator(train_loader, epoch_metrics)
            self._train_encoder_decoder(train_loader, epoch_metrics)

            # Validation phase
            self._validate(validation_loader, epoch_metrics)

            # Calculate average metrics for the epoch
            self.current_metrics = {
                k: sum(v) / len(v) for k, v in epoch_metrics.items() if v
            }
            self.current_metrics['epoch'] = current_epoch

            # Save artifacts if logging is enabled
            if self.log_dir:
                log_dir = Path(self.log_dir)

                # Save metrics history
                self.training_history.append(self.current_metrics)
                metrics_path = log_dir / 'metrics.log'
                with open(metrics_path, 'w') as metrics_file:
                    json.dump(self.training_history, metrics_file, indent=4)

                # Save model checkpoint
                bpp_value = self.current_metrics['val.bpp']
                checkpoint_name = f'{current_epoch}.bpp-{bpp_value:.3f}.p'
                self.save(log_dir / checkpoint_name)

                # Generate and save sample images
                if self.log_dir and 'sample_batch' in locals():
                    self._save_sample_images(self.samples_dir, sample_batch, epoch)

            # Clean up memory
            if self.cuda_enabled:
                torch.cuda.empty_cache()
            gc.collect()

    def _prepare_message_payload(self, width, height, message_text):
        """
        Prepare a message payload for encoding in an image.

        Args:
            width: Width of the target image
            height: Height of the target image
            message_text: Text message to encode

        Returns:
            Payload tensor ready for encoding
        """
        # Convert text to bits and add padding
        message_bits = encode_text_to_bits(message_text) + [0] * 32

        # Repeat message to fill the entire payload area
        payload_bits = message_bits.copy()
        required_bits = width * height * self.data_depth

        while len(payload_bits) < required_bits:
            payload_bits += message_bits

        # Truncate to exact size needed
        payload_bits = payload_bits[:required_bits]

        # Convert to tensor of proper dimensions
        return torch.FloatTensor(payload_bits).view(1, self.data_depth, height, width)

    def encode(self, cover_path, output_path, message_text):
        """
        Encode a secret message into an image.

        Args:
            cover_path: Path to the cover image
            output_path: Path to save the steganographic image
            message_text: Secret message to hide
        """
        # Load and preprocess cover image
        cover_image = imread(cover_path, pilmode='RGB') / 127.5 - 1.0
        cover_tensor = torch.FloatTensor(cover_image).permute(2, 1, 0).unsqueeze(0)

        # Prepare message payload
        _, _, height, width = cover_tensor.size()
        message_payload = self._prepare_message_payload(width, height, message_text)

        # Move to device and encode message
        cover_tensor = cover_tensor.to(self.device)
        message_payload = message_payload.to(self.device)
        stego_tensor = self.encoder(cover_tensor, message_payload)[0].clamp(-1.0, 1.0)

        # Convert to image and save
        stego_image = (stego_tensor.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output_path, stego_image.astype('uint8'))

        if self.verbose:
            print('Message encoded successfully.')

    def decode(self, image_path):
        """
        Decode a hidden message from a steganographic image.

        Args:
            image_path: Path to the steganographic image

        Returns:
            Decoded secret message

        Raises:
            ValueError: If image doesn't exist or no message is found
        """
        # Verify image exists
        if not os.path.exists(image_path):
            raise ValueError(f'Unable to read image: {image_path}')

        # Load and preprocess image
        stego_image = imread(image_path, pilmode='RGB') / 255.0
        stego_tensor = torch.FloatTensor(stego_image).permute(2, 1, 0).unsqueeze(0)
        stego_tensor = stego_tensor.to(self.device)

        # Decode message bits
        decoded_bits = (self.decoder(stego_tensor).view(-1) > 0).int().cpu().numpy().tolist()
        decoded_bytes = bits_to_bytearray(decoded_bits)

        # Extract candidate messages (separated by null sequences)
        message_candidates = Counter()

        # 1. Try without splitting (whole message)
        whole_message = recover_and_decompress_text(bytearray(decoded_bytes))
        if whole_message:
            message_candidates[whole_message] += 10  # Give higher weight

        # 2. Try with splitting as fallback
        for candidate in decoded_bytes.split(b'\x00\x00\x00\x00'):
            decoded_text = recover_and_decompress_text(bytearray(candidate))
            if decoded_text:
                message_candidates[decoded_text] += 1

        # Return most common message or raise error
        if not message_candidates:
            raise ValueError('No valid message found in the image.')

        most_common_message, _ = message_candidates.most_common(1)[0]
        return most_common_message

    def save(self, path):
        """
        Save the trained model to disk.

        Args:
            path: Path where the model will be saved
        """
        torch.save(self, path)

    @classmethod
    def load(cls, path=None, use_cuda=True, verbose=False):
        """
        Load a pretrained SteganoGAN model.

        Args:
            architecture: Name of a pretrained model architecture
            path: Path to a custom pretrained model file
            use_cuda: Whether to use GPU acceleration if available
            verbose: Whether to enable verbose output

        Returns:
            Loaded SteganoGAN model

        Raises:
            ValueError: If neither architecture nor path is provided, or if both are provided
        """
        # Determine model path
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found.')

        # Load model
        steganogan = torch.load(path, map_location='cpu', weights_only=False)
        steganogan.verbose = verbose

        # Handle backward compatibility
        steganogan.encoder.upgrade_legacy()
        steganogan.decoder.upgrade_legacy()
        steganogan.critic.upgrade_legacy()

        # Configure device
        steganogan.set_device(use_cuda)
        return steganogan
