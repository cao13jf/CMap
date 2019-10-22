import torch
import torch.nn as nn
import torch.nn.functional as F

# Import sync_bacth for multiple GPUs
try:
    from .sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass

#============================================
#   Define normalization
#============================================
def normalization(nchanels, norm="bn"):
    if norm == "bn":
        nlayer = nn.BatchNorm3d(nchanels)
    elif norm == "in":
        nlayer = nn.InstanceNorm3d(nchanels)
    elif norm == "gn":
        nlayer = nn.GroupNorm(4, nchanels)
    elif norm == "sync_bn":
        nlayer = SynchronizedBatchNorm3d(nchanels)
    else:
        raise ValueError("Normalization type {} is supported, choose for 'bn', 'in', 'gn', and 'sync_bn'".format(norm))
    return nlayer


#==============================================
#   Define convolution
#==============================================

# Define convolution 3D block without dialtion
class Conv3dBlock(nn.Module):

    def __init__(self, num_in, num_out, kernel_size=(1, 1, 1), stride=1, padding=None, groups=1, norm="bn"):
        super(Conv3dBlock, self).__init__()
        if padding is None:
            padding = (kernel_size - 1)//2
        self.norm = normalization(num_in, norm=norm)
        self.act_fun = nn.ReLU(inplace=True)
        self.conv3d = nn.Conv3d(num_in, num_out, kernel_size, padding=padding, groups=groups, stride=stride, bias=False)

    def forward(self, x):
        x = self.act_fun(self.norm(x))
        x = self.conv3d(x)

        return x

# Define convolution with dialtion
class DilatedConv3dBlock(nn.Module):

    def __init__(self, num_in, num_out, kernel_size=(1, 1, 1), stride=1, groups=1, dilations=(1, 1, 1), norm="bn"):
        super(DilatedConv3dBlock, self).__init__()
        padding = tuple([(ks - 1) // 2 * ds for ks, ds in zip(kernel_size, dilations)])
        self.norm = normalization(num_in, norm=norm)
        self.act_fun = nn.ReLU(inplace=True)
        self.conv3d = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, stride=stride, groups=groups, dilation=dilations,
                                padding=padding, bias=False)

    def forward(self, x):
        x = self.act_fun(self.norm(x))
        x = self.conv3d(x)

        return x


#=================================================
#   Define specific structure
#=================================================
#  Define MFUnit
class MDUnit(nn.Module):
    def __init__(self, num_in, num_out, stride=1, norm=None):
        super(MDUnit, self).__init__()
        self.conv1x3x3_d1 = DilatedConv3dBlock(num_in, num_in, kernel_size=(1, 3, 3), stride=stride, dilations=(1, 3, 3), norm=norm)
        self.conv1x3x3_d3 = DilatedConv3dBlock(num_in, num_in, kernel_size=(1, 3, 3), stride=stride, dilations=(1, 3, 3), norm=norm)
        self.conv1x3x3_d6 = DilatedConv3dBlock(num_in, num_in, kernel_size=(1, 3, 3), stride=stride, dilations=(1, 6, 6), norm=norm)  # use stride=2 to reduce size
        self.conv1x3x3_d9 = DilatedConv3dBlock(num_in, num_in, kernel_size=(1, 3, 3), stride=stride, dilations=(1, 9, 9), norm=norm)

        self.conv9x1x1 = Conv3dBlock(num_in, num_out, kernel_size=(9, 1, 1), stride=1, padding=(4, 0, 0), norm=norm)


    def forward(self, x):
        h_d1 = self.conv1x3x3_d1(x)  # parameters are set in "groups", no need for specific grouping
        h_d2 = self.conv1x3x3_d3(x)  # parameters are set in "groups", no need for specific grouping
        h_d3 = self.conv1x3x3_d6(x)  # parameters are set in "groups", no need for specific grouping
        h_d4 = self.conv1x3x3_d9(x)  # parameters are set in "groups", no need for specific grouping
        h = h_d1 + h_d2 + h_d3 + h_d4

        return self.conv9x1x1(h)


#==================================================
#   define network
#==================================================
class SliceNet(nn.Module):
    def __init__(self, in_channels=1, n_first=32, conv_channels=128, norm="bn", out_class=2, groups=1):
        super(SliceNet, self).__init__()

        # encoder
        self.first_conv = nn.Conv3d(in_channels, n_first, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=2, bias=False)  # [W/2, H/2]
        self.encoder_block1 = MDUnit(num_in=n_first, num_out=conv_channels, stride=2, norm=norm)  # [W/4, H/4]
        self.encoder_block2 = MDUnit(num_in=conv_channels, num_out=2*conv_channels, stride=2, norm=norm)  # [W/8, H/8]
        self.encoder_block3 = MDUnit(num_in=2*conv_channels, num_out=3*conv_channels, stride=2, norm=norm)  # [W/16, H/16]

        # decoder
        self.up_sample1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)  # [W/8, H/8]
        self.deconder_block1 = MDUnit(num_in=3*conv_channels+2*conv_channels, num_out=2*conv_channels, stride=1, norm=norm)
        self.up_sample2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)  # [W/4, W/4]
        self.deconder_block2 = MDUnit(num_in=2*conv_channels+conv_channels, num_out=conv_channels, stride=1, norm=norm)
        self.up_sample3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)  # [W/2, H/2]
        self.deconder_block3 = MDUnit(num_in=conv_channels+n_first, num_out=n_first, stride=1, norm=norm)
        self.up_sample4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)  # [W, H]

        self.seg = nn.Conv3d(in_channels=n_first, out_channels=out_class, kernel_size=1, stride=1, bias=False)

        self.softmax = nn.Softmax(dim=1)

        # initilization
        #  Weights initlization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #TODO: different between method and method_
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant(m.bias, 0.0)

    def forward(self, x):
        h = self.first_conv(x)  # [W/2, H/2]
        h_encoder1 = self.encoder_block1(h)  # [W/4, H/4]
        h_encoder2 = self.encoder_block2(h_encoder1)  # [W/8, H/8]
        h_encoder3 = self.encoder_block3(h_encoder2)  # [W/16, H/16]

        up1 = self.up_sample1(h_encoder3) # [W/8, W/8]
        h_decoder1 = self.deconder_block1(torch.cat([up1, h_encoder2], dim=1))
        up2 = self.up_sample2(h_decoder1)  # [W/4, H/4]
        h_decoder2 = self.deconder_block2(torch.cat([up2, h_encoder1], dim=1))
        up3 = self.up_sample2(h_decoder2)  # [W/2, H/2]
        h_decoder3 = self.deconder_block3(torch.cat([up3, h], dim=1))
        up4 = self.up_sample4(h_decoder3)

        seg = self.seg(up4)
        out = self.softmax(seg)

        return out