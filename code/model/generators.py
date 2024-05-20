import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

from model.cc_attention import CrissCrossAttention
from model.coordatt import CoordAtt
from model.epsablock import EPSABlock



#加了残差(cbam ,res in ca)
class UNetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, embedding_num=40, embedding_dim=128,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UNetGenerator, self).__init__()
        # construct unet structure

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, embedding_dim=embedding_dim)  # add the innermost layer
        k=0
        for _ in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            print(k+1)
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
            #unet_block = unet_block+[ResnetBlock(ngf *8,ngf*8,norm_layer=norm_layer)]
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer
        self.embedder = nn.Embedding(embedding_num, embedding_dim)

    def forward(self, x, style_or_label=None):
        """Standard forward"""
        if style_or_label is not None and 'LongTensor' in style_or_label.type():
            return self.model(x, self.embedder(style_or_label))
        else:
            return self.model(x, style_or_label)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, embedding_dim=128, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        res = ResBlock(input_nc, inner_nc)
        #cca = CrissCrossAttention(inner_nc)

        cbam =CBAM(inner_nc)



        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv,res,cbam]

            up = [uprelu, upconv, nn.Tanh()]

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc + embedding_dim, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv,res,cbam]
            up = [uprelu, upconv, upnorm]

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm,res,cbam]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]

        self.submodule = submodule
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)

    def forward(self, x, style=None):
        #print("--------",x.shape)
        if self.innermost:
            encode = self.down(x)
            if style is None:
                return encode
            enc = torch.cat([style.view(style.shape[0], style.shape[1], 1, 1), encode], 1)
            dec = self.up(enc)
            return torch.cat([x, dec], 1), encode.view(x.shape[0], -1)
        elif self.outermost:
            enc = self.down(x)
            if style is None:
                return self.submodule(enc)
            sub, encode = self.submodule(enc, style)
            dec = self.up(sub)
            return dec, encode
        else:  # add skip connections
            enc = self.down(x)
            if style is None:
                return self.submodule(enc)
            sub, encode = self.submodule(enc, style)
            dec = self.up(sub)
            return torch.cat([x, dec], 1), encode

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_attention = self.sigmoid_channel(avg_out + max_out)

        # Spatial Attention
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_attention = self.sigmoid_spatial(self.conv(spatial_input))

        # Apply attention
        x = x * channel_attention * spatial_attention

        return x



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #self.ca = ChannelAttention(out_channels )
        #self.sa = SpatialAttention()
        #self.cca =  CrissCrossAttention(out_channels)
        self.ca = CoordAtt(out_channels,out_channels)


    def forward(self, x):
        residual = x

        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn2(out)

        #out = self.ca(out) * out
       # out = self.sa(out) * out

        out = self.ca(out)

        out += residual


        return out



