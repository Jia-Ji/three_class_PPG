import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock2D(nn.Module):
    expansion = 1  # No expansion in BasicBlock

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, drop_out_p=0.2):
        super(BasicBlock2D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=drop_out_p)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # Downsampling layer if needed

    def forward(self, x):
        identity = x  # Save input for residual connection
        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust dimensions if needed

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Residual connection
        out = self.relu(out)

        return out

class BasicBlock1D(nn.Module):
    expansion = 1  # No expansion in BasicBlock

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_p=0.2):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # Downsampling layer if needed

    def forward(self, x):
        identity = x  # Save input for residual connection
        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust dimensions if needed

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Residual connection
        out = self.relu(out)

        return out


class SelfAttention1D(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super(SelfAttention1D, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                                               batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, channels, seq_len)
        """
        x_seq = x.transpose(1, 2)  # (batch, seq_len, channels)
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        attn_out = self.norm(attn_out + x_seq)
        return attn_out.transpose(1, 2)


class ResNet2D(nn.Module):
    def __init__(self, block, signal_channels, layers, num_classes=2, dropout_p=0.2):
        super(ResNet2D, self).__init__()
        self.in_channels = 64  # Initial channels before block stacking

        # Initial Convolutional Layer (Conv + BN + ReLU + MaxPool)
        self.conv1 = nn.Conv2d(signal_channels, 64, kernel_size=6, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global Average Pooling & Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize Weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride, dropout_p):
        """Create a ResNet layer with multiple residual blocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_p))  # First block may downsample
        self.in_channels = out_channels * block.expansion  # Update channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights with Kaiming He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Define forward pass of ResNet."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return F.softmax(x, dim=1)

class ResNet1D(nn.Module):
    def __init__(self, block, num_layers, signal_channels, layer_norm, feat_dim, dropout_p,
                 attention_heads=8, use_attention=True):
        super(ResNet1D, self).__init__()

        assert num_layers in [10, 18, 34], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                           f'to be 18, or 34 '

        if num_layers == 10:
            layers = [1, 1, 1, 1]
        elif num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34:
            layers = [3, 4, 6, 3]

        self.in_channels = 64  # Initial channels before block stacking
        self.layer_norm = layer_norm
        self.feat_dim = feat_dim

        # Initial Convolutional Layer (Conv + BN + ReLU + MaxPool)
        self.conv1 = nn.Conv1d(signal_channels, 64, kernel_size=6, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Blocks
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_p, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], dropout_p, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], dropout_p, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], dropout_p, stride=2)

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = SelfAttention1D(512 * block.expansion, num_heads=attention_heads, dropout=dropout_p)

        # Global Average Pooling & Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, self.feat_dim)

        if self.layer_norm:
                self.feat_norm_layer = nn.LayerNorm(self.feat_dim)
        # Initialize Weights
        # self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, dropout_p, stride):
        """Create a ResNet layer with multiple residual blocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_p))  # First block may downsample
        self.in_channels = out_channels * block.expansion  # Update channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights with Kaiming He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Define forward pass of ResNet."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_attention:
            x = self.attention(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.layer_norm:
            x = self.feat_norm_layer(x)

        return x

def resnet18_2D(signal_channels, stride, layer_norm, feat_dim, dropout_p):
    return ResNet2D(BasicBlock2D, 18, signal_channels=signal_channels, layer_norm=layer_norm, feat_dim=feat_dim)

def resnet10_2D(signal_channels, stride, layer_norm, feat_dim):
    return ResNet2D(BasicBlock2D, 10, signal_channels=signal_channels, layer_norm=layer_norm, feat_dim=feat_dim)

def resnet18_1D(signal_channels, stride, layer_norm, feat_dim, dropout_p, use_attention=True, attention_heads=8):
    return ResNet1D(
        BasicBlock1D,
        18,
        signal_channels=signal_channels,
        layer_norm=layer_norm,
        feat_dim=feat_dim,
        dropout_p=dropout_p,
        use_attention=use_attention,
        attention_heads=attention_heads,
    )

def resnet34_1D(signal_channels, stride, layer_norm, feat_dim, dropout_p, use_attention=True, attention_heads=8):
    return ResNet1D(
        BasicBlock1D,
        34,
        signal_channels=signal_channels,
        layer_norm=layer_norm,
        feat_dim=feat_dim,
        dropout_p=dropout_p,
        use_attention=use_attention,
        attention_heads=attention_heads,
    )