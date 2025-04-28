import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(BasicBlock, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU(out_channels)

        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.prelu2(out)

        return out


class YOCO(nn.Module):
    def __init__(self, num_classes=13, count_range=6):
        super(YOCO, self).__init__()
        self.num_classes = num_classes
        self.count_range = count_range
        self.output_dim = num_classes * count_range

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
        )

        self.layer1 = BasicBlock(32, 64, downsample=True)    # 600x400
        self.layer2 = BasicBlock(64, 128, downsample=True)   # 300x200
        self.layer3 = BasicBlock(128, 256, downsample=True)  # 150x100
        self.layer4 = BasicBlock(256, 512, downsample=True)  # 75x50
        self.layer5 = BasicBlock(512, 512, downsample=True)  # 37x25
        self.layer6 = BasicBlock(512, 512, downsample=True)  # 18x12

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(256),

            nn.Conv2d(256, self.output_dim, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.head(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), self.num_classes, self.count_range)
        return x


class YOCOLARGE(nn.Module):
    def __init__(self, num_classes=13, count_range=6):
        super(YOCOLARGE, self).__init__()
        self.num_classes = num_classes
        self.count_range = count_range
        self.output_dim = num_classes * count_range

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
        )

        self.layer1 = BasicBlock(32, 64, downsample=True)    # 600x400
        self.layer2 = BasicBlock(64, 128, downsample=True)   # 300x200
        self.layer3 = BasicBlock(128, 256, downsample=True)  # 150x100
        self.layer4 = BasicBlock(256, 512, downsample=True)  # 75x50
        self.layer5 = BasicBlock(512, 768, downsample=True)  # 37x25
        self.layer6 = BasicBlock(768, 768, downsample=True)  # 18x12

        self.head = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(512),

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(256),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(128),

            nn.Conv2d(128, self.output_dim, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.head(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), self.num_classes, self.count_range)
        return x


class YOCOSMALL(nn.Module):
    def __init__(self, num_classes=13, count_range=6):
        super(YOCOSMALL, self).__init__()
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
        super(YOCOTINY, self).__init__()
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


# Function to print the number of trainable parameters of each layer
def print_trainable_params(model):
    print(f"\n{model.__class__.__name__} Layer Parameter Sizes:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_size = param.numel()
            total_params += param_size
            print(f"{name} | {param_size} parameters")
    print(f"Total Parameters: {total_params}")


# Main function
if __name__ == "__main__":
    # Instantiate models
    models = [YOCO(), YOCOLARGE(), YOCOSMALL(), YOCOTINY()]

    for model in models:
        print_trainable_params(model)
