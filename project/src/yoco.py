import torch
import torch.nn as nn
import torch.nn.functional as F

class YOCO(nn.Module):
    def __init__(self, num_classes=13, count_range=6):
        super(YOCO, self).__init__()
        self.num_classes = num_classes
        self.count_range = count_range
        self.output_dim = num_classes * count_range

        self.features = nn.Sequential(
            # Input: [B, 3, 1200, 800]
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 1200, 800]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 32, 600, 400]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 600, 400]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 64, 300, 200]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, 300, 200]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 128, 150, 100]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [B, 256, 150, 100]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 256, 75, 50]

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # [B, 512, 75, 50]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 512, 37, 25]

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # [B, 512, 37, 25]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 512, 18, 12]
        )

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # [B, 256, 18, 12]
            nn.LeakyReLU(0.3),
            nn.Conv2d(256, self.output_dim, kernel_size=1),  # [B, output_dim (13x6=78), 18, 12]
        )

    def forward(self, x):
        x = self.features(x)  # [B, 512, 18, 12]
        x = self.head(x)      # [B, 78, 18, 12]
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, 78, 1, 1]
        x = x.view(x.size(0), self.num_classes, self.count_range)  # [B, 13, 6]
        return x

class YOCOLARGE(nn.Module):
    def __init__(self, num_classes=13, count_range=6):
        self.num_classes = num_classes
        self.count_range = count_range
        self.output_dim = num_classes * count_range

        self.features = nn.Sequential(
            # Input: [B, 3, 1200, 800]
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 1200, 800]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 64, 600, 400]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, 600, 400]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 128, 300, 200]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [B, 256, 300, 200]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 256, 150, 100]

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # [B, 512, 150, 100]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 512, 75, 50]

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # [B, 1024, 75, 50]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 1024, 37, 25]

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),  # [B, 1024, 37, 25]
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # [B, 1024, 18, 12]
        )

        self.head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),  # [B, 512, 18, 12]
            nn.LeakyReLU(0.3),

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # [B, 256, 18, 12]
            nn.LeakyReLU(0.3),

            nn.Conv2d(256, self.output_dim, kernel_size=1),  # [B, 78, 18, 12]
        )

    def forward(self, x):
        x = self.features(x)  # [B, 1024, 18, 12]
        x = self.head(x)  # [B, 78, 18, 12]
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, 78, 1, 1]
        x = x.view(x.size(0), self.num_classes, self.count_range)  # [B, 13, 6]
        return x

class YOCOSMALL(nn.Module):
    def __init__(self, num_classes=13, count_range=6):
        self.num_classes = num_classes
        self.count_range = count_range
        self.output_dim = num_classes * count_range

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(128, self.output_dim, kernel_size=1),
        )

    def forward(self, x):
        x = self.features(x)  # Shape: [B, 256, 7, 7] for 1200x800 input
        x = self.head(x)      # Shape: [B, 13*6, 7, 7]
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, 13*6, 1, 1]
        x = x.view(x.size(0), self.num_classes, self.count_range)  # [B, 13, 6]
        return x

class YOCOTINY(nn.Module):
    def __init__(self, num_classes=13, count_range=6):
        self.num_classes = num_classes
        self.count_range = count_range
        self.output_dim = num_classes * count_range

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(128, self.output_dim, kernel_size=1),
        )

    def forward(self, x):
        x = self.features(x)  # Shape: [B, 256, 7, 7] for 1200x800 input
        x = self.head(x)      # Shape: [B, 13*6, 7, 7]
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, 13*6, 1, 1]
        x = x.view(x.size(0), self.num_classes, self.count_range)  # [B, 13, 6]
        return x
