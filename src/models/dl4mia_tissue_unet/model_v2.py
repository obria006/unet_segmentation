"""
Module for UNet model. This is the second version of the UNet model.
This second version was inspired by Ryan Peters version of UNet that
impliments residual blocks and batch norms in the UNet.
"""
# FIXME Ryans Unet uses bias=True but other is bias=False
# FIXME why isnt decoder input channels double of that levels encoder output

import torch
import torch.nn as nn


def ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding="same"):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False),
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding="same", bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def FinalBlock(in_channels, out_channels):
    return nn.Sequential(
        # ConvBlock(in_channels, in_channels,  kernel_size = 1),
        # ConvBlock(in_channels, in_channels,  kernel_size = 1),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same")
    )


def MiddleBlock(in_channels, out_channels):
    return nn.Sequential(
        ConvBlock(in_channels, out_channels),
        nn.Dropout(p=0.2),
        ConvBlock(out_channels, out_channels),
    )


class ResidualBlock(nn.Module):
    """Residual encoder block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bn: bool = True,
        do_p: float = 0,
    ):
        """
        Args:
            in_channels (int): # of input channels
            out_channels (int): 3 of output cahnnels
            stride (int): Convolutional stride
            bn (bool): True to include batch normalization
            do (float): 0-1 dropout percentage
        """
        super(ResidualBlock, self).__init__()
        if do_p:
            assert 0 <= do_p <= 1
            raise NotImplementedError("Residual block dropout")

        # Define attrs from arguemnts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bn = bn
        self.do_p = do_p

        # Create layers
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.do1 = nn.Dropout(p=self.do_p)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.do2 = nn.Dropout(p=self.do_p)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=None)
        # FIXME: What is this? I havent seen batchnrom after downsample in res block
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x

        # First conv-BN-DO-RELU
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        if self.do_p:
            # x = self.do1(x)
            raise NotImplementedError("Residual block dropout")
        x = self.relu(x)

        # Second conv-BN-DO
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        if self.do_p:
            # x = self.do2(x)
            raise NotImplementedError("Residual block dropout")

        # 1x1 conv the input
        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        skip_connection = self.relu(x)

        x = self.maxpool(skip_connection)

        return x, skip_connection


class UNetEncoder(nn.Module):
    """
    Encoder pathway for UNet using residual blocks and returning skip
    connections
    """

    @property
    def out_channels(self) -> int:
        """Return output channels as start_filts * 2**(depth-1)"""
        return self.start_filts * 2 ** (self.depth - 1)

    def __init__(
        self,
        in_channels: int,
        depth: int,
        start_filts: int,
        bn: bool = True,
        do_p: float = 0,
    ):
        """
        Args:
            in_channels (int): # of channels in input image (1: gray, 3:RGB)
            depth (int): # of maxpools (downsize steps)
            start_filts (int): # of convolution filters for first convolution
            bn (bool): True to include batch normalization
            do_p (float): 0-1 dropout percentage
        """
        super(UNetEncoder, self).__init__()

        # Set attrs from arguments
        self.in_channels = in_channels
        self.depth = depth
        self.start_filts = start_filts
        self.bn = bn
        self.do_p = do_p

        # Specify down-convolutional steps
        self.down_convs = []
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)

            res_block = ResidualBlock(
                in_channels=ins, out_channels=outs, bn=self.bn, do_p=self.do_p
            )
            self.down_convs.append(res_block)

        self.down_convs = nn.ModuleList(self.down_convs)

    def forward(self, x):
        """
        Retuns:
            x (torch.tensor): Output feature maps from encoder
            encoder_outs (list): List of skip connections from before maxpool
                in order of first to last maxpool
        """
        encoder_outs = []
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        return x, encoder_outs


class UNetDecoderBlock(nn.Module):
    """Helper block to concatenate skip connection upon decoder upsampling and"""

    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2)

        # CovBlock in_channels is in_channels+out_channels because we concatenate skip connection
        self.conv_block_0 = ConvBlock(in_channels + out_channels, out_channels)
        self.conv_block_1 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)

        x = torch.cat((x, skip_connection), 1)

        x = self.conv_block_0(x)
        x = self.conv_block_1(x)

        return x


class UNetDecoder(nn.Module):
    """Decoder pathway for UNet with using skip connections in `forward()`"""

    @property
    def out_channels(self) -> int:
        """Returns number of output channels as self.in_channels // 2**(depth)"""
        return self.in_channels // (2 ** (self.depth))

    def __init__(self, in_channels: int, depth: int):
        """
        Args:
            in_channels (int): # of input channels
            depth (int): # of upsamples
        """
        super(UNetDecoder, self).__init__()

        # Set attrs from arguments
        self.in_channels = in_channels
        self.depth = depth

        self.up_convs = []
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = ins // 2
            decode_block = UNetDecoderBlock(in_channels=ins, out_channels=outs)
            self.up_convs.append(decode_block)

        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x, skip_connections: list):
        """
        Retuns:
            x (torch.tensor): Output feature maps from encoder/middle block
            skip_connections (list): List of skip connections from before maxpool
                in order of first to last maxpool
        """
        assert len(skip_connections) == self.depth, "Depth skip connection mismatch"
        for i, module in enumerate(self.up_convs):
            # Must start at the last skip connection
            skip_connection = skip_connections[-(i + 1)]
            x = module(x, skip_connection)
        return x


class UNet(nn.Module):
    """
    UNet with encoder (3x3 conv - BN (opt) - DO (opt) - RELU)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        depth: int = 3,
        start_filts: int = 64,
        batch_norm: bool = True,
        dropout_p: float = 0,
    ):
        """
        Args:
            num_classes (int): # of segmentable classes (i.e. 1 if just fg)
            in_channels (int): # of channels in input image (1: gray, 3:RGB)
            depth (int): # of maxpools (downsize steps)
            start_filts (int): # of convolution filters for first convolution
            batch_norm (bool): True includes batch norm in UNet
            dropout (bool): True includes dropout in first conv
        """
        super().__init__()

        # attrs from arguments
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.depth = depth
        self.start_filts = start_filts
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p

        # Encoder downsizes `depth` number of times
        self.encoder = UNetEncoder(
            in_channels=self.in_channels,
            depth=self.depth,
            start_filts=self.start_filts,
            bn=self.batch_norm,
            do_p=self.dropout_p,
        )

        # Middle block (conv-BN-relu-DO-conv-BN-relu) to double channels
        enc_out_channels = self.encoder.out_channels
        self.middle = MiddleBlock(
            in_channels=enc_out_channels, out_channels=2 * enc_out_channels
        )

        # Decoder upsamples `depth` number of times
        self.decoder = UNetDecoder(in_channels=2 * enc_out_channels, depth=self.depth)

        # Finally 1x1 convolution to single image
        dec_out_channels = self.decoder.out_channels
        self.final = FinalBlock(
            in_channels=dec_out_channels, out_channels=self.num_classes
        )

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x, skip_connections)
        x = self.final(x)
        return x


if __name__ == "__main__":
    mdl = UNet(num_classes=1, in_channels=1, depth=3, start_filts=32)
    # print(mdl)
