import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1, bias=True)   # 6x28x28  (54 params)
        self.pool1 = nn.MaxPool2d(2, 2)                                       # 6x14x14
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=1, bias=True)   # 12x14x14 (648 params)
        self.pool2 = nn.MaxPool2d(2, 2)                                       # 12x7x7
        self.fc1 = nn.Linear(12 * 7 * 7, 16, bias=True)                       # 588->16  (9,424 params)
        self.fc2 = nn.Linear(16, 10, bias=True)                               # 16->10   (170 params)
        # Total: ~10.3K params

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class UltraTinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=True)    # 4x28x28   (36 params)
        self.pool1 = nn.MaxPool2d(2, 2)                                       # 4x14x14
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=True)    # 8x14x14   (288 params)
        self.pool2 = nn.MaxPool2d(2, 2)                                       # 8x7x7 = 392
        self.fc1 = nn.Linear(8 * 7 * 7, 16, bias=True)                        # 392->16   (6,288 params)
        self.fc2 = nn.Linear(16, 10, bias=True)                               # 16->10    (170 params)
        # Total: ~6.8K params

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32, bias=True)    # 784->32  (25,120 params)
        self.fc2 = nn.Linear(32, 10, bias=True)      # 32->10   (330 params)
        # Total: ~25.5K params

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(name="tiny_cnn"):
    models = {
        "tiny_cnn": TinyCNN,
        "ultra_tiny_cnn": UltraTinyCNN,
        "tiny_mlp": TinyMLP,
    }
    return models[name]()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for name in ["tiny_cnn", "ultra_tiny_cnn", "tiny_mlp"]:
        model = get_model(name)
        print(f"{name}: {count_params(model):,} parameters")