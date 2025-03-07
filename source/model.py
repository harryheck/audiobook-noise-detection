import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, input_shape, output_length, dilation=1):
        """Builds a CNN model for audio classification using mel spectrograms.

        Args:
            input_shape (tuple): Shape of the input spectrogram (height, width, channels).
            output_length (int): Number of output classes.
            dilation (int): Dilation rate for convolutional layers.
        """
        super(AudioCNN, self).__init__()
        
        self.batch_norm1 = nn.BatchNorm2d(1)
        
        # First Conv Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=2, padding_mode="replicate", dilation=dilation, bias=False)
        self.leaky_relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second Conv Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=2, padding_mode="replicate", dilation=dilation, bias=False)
        self.leaky_relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        # Third Conv Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=2, padding_mode="replicate", dilation=dilation, bias=False)
        self.leaky_relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.4)
        
        # Fourth Conv Block
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=2, padding_mode="replicate", dilation=dilation, bias=False)
        self.leaky_relu4 = nn.LeakyReLU(0.1)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(128, 128)
        self.dropout4 = nn.Dropout(0.5)
        
        # Output Layer
        self.fc2 = nn.Linear(128, output_length)

    def forward(self, x):
        x = self.batch_norm1(x)
        
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.pool1(x)
        x = self.batch_norm2(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.pool2(x)
        x = self.batch_norm3(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.pool3(x)
        x = self.batch_norm4(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.leaky_relu4(x)
        x = self.global_pool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.silu(x)  # Swish activation
        x = self.dropout4(x)
        
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
