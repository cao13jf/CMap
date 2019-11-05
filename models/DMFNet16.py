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

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=None, groups=1, norm=None):
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
class MFUnit(nn.Module):
    def __init__(self, num_in, num_out, stride=1, groups=1, norm=None):
        super(MFUnit, self).__init__()
        num_mid = num_in if num_in < num_out else num_out
        self.conv1x1x1_in1 = Conv3dBlock(num_in=num_in, num_out=num_out//4,kernel_size=1, stride=1, norm=norm)
        self.conv1x1x1_in2 = Conv3dBlock(num_in=num_out//4, num_out=num_mid,kernel_size=1, stride=1, norm=norm)
        self.conv3x3x3_m1 = DilatedConv3dBlock(num_mid, num_out, kernel_size=(3, 3, 3), stride=stride, groups=groups, norm=norm)  # use stride=2 to reduce size
        self.conv3x3x3_m2 = DilatedConv3dBlock(num_out, num_out, kernel_size=(3, 3, 1), stride=1, groups=groups, norm=norm)

        # add skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3dBlock(num_in, num_out, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                self.conv2x2x2_shortcut = Conv3dBlock(num_in, num_out, kernel_size=2, stride=2, padding=0, norm=norm)

    def forward(self, x):
        h = self.conv1x1x1_in1(x)  # parameters are set in "groups", no need for specific grouping
        h = self.conv1x1x1_in2(h)
        h = self.conv3x3x3_m1(h)
        h = self.conv3x3x3_m2(h)

        shortcut = x
        if hasattr(self, "conv1x1x1_shortcut"):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        elif hasattr(self, "conv2x2x2_shortcut"):
            shortcut = self.conv2x2x2_shortcut(shortcut)

        return h + shortcut

#  define DMFUnit module
class DMFUnit(nn.Module):
    def __init__(self, num_in, num_out, stride=1, groups=1, dilation=None, norm=None):
        super(DMFUnit, self).__init__()
        self.weight0 = nn.Parameter(torch.ones(1))
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))

        num_mid = num_in if num_in < num_out else num_out
        self.conv1x1x1_in1 = Conv3dBlock(num_in, num_in//4, kernel_size=1, stride=1, norm=norm)
        self.conv1x1x1_in2 = Conv3dBlock(num_in//4, num_mid, kernel_size=1, stride=1, norm=norm)

        self.conv3x3x3_m1 = nn.ModuleList()
        if dilation is None:
            dilation = [1, 2, 3]
        for i in range(3):
            self.conv3x3x3_m1.append(
                DilatedConv3dBlock(num_mid, num_out=num_out, kernel_size=(3,3,3), stride=stride, groups=groups,
                                   dilations=(dilation[i],)*3, norm=norm)
            )
        self.conv3x3x3_m2 = DilatedConv3dBlock(num_out, num_out, kernel_size=(3,3,1), stride=1, groups=groups,
                                               dilations=(1,1,1), norm=norm)
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3dBlock(num_in, num_out, kernel_size=1, stride=1, padding=0, norm=norm)
            elif stride == 2:
                self.conv2x2x2_shortcut = Conv3dBlock(num_in, num_out, kernel_size=2, stride=2, padding=0, norm=norm)

    def forward(self, x):
        h = self.conv1x1x1_in1(x)
        h = self.conv1x1x1_in2(h)
        h = self.weight0 * self.conv3x3x3_m1[0](h) + self.weight1 * self.conv3x3x3_m1[1](h) + self.weight2 * self.conv3x3x3_m1[2](h)
        h = self.conv3x3x3_m2(h)

        # add shortcut
        shortcut = x
        if hasattr(self, "conv1x1x1_shortcut"):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        elif hasattr(self, "conv2x2x2_shortcut"):
            shortcut = self.conv2x2x2_shortcut(shortcut)

        return h + shortcut


#===================================================
#   Define classes based on pre-defined units
#===================================================
#   define MFNet
class MFNet(nn.Module):
    def __init__(self, in_channels=1, n_first=32, conv_channels=128, groups=16, norm="bn", out_class=2):
        super(MFNet, self).__init__()

        self.first_conv = nn.Conv3d(in_channels, n_first, kernel_size=3, padding=1, stride=2, bias=False)
        self.encoder_block1 = nn.Sequential(
            MFUnit(num_in=n_first, num_out=conv_channels, groups=groups, stride=2, norm=norm),
            MFUnit(num_in=conv_channels, num_out=conv_channels, groups=groups, stride=1, norm=norm),
            MFUnit(num_in=conv_channels, num_out=conv_channels, groups=groups, stride=1, norm=norm)
        )
        self.encoder_block2 = nn.Sequential(
            MFUnit(num_in=conv_channels, num_out=2*conv_channels, groups=groups, stride=2, norm=norm),
            MFUnit(num_in=2*conv_channels, num_out=2*conv_channels, groups=groups, stride=1, norm=norm),
            MFUnit(num_in=2*conv_channels, num_out=2*conv_channels, groups=groups, stride=1, norm=norm)
        )
        self.encoder_block3 = nn.Sequential(
            MFUnit(num_in=2*conv_channels, num_out=3*conv_channels, groups=groups, stride=2, norm=norm),
            MFUnit(num_in=3*conv_channels, num_out=3*conv_channels, groups=groups, stride=1, norm=norm),
            MFUnit(num_in=3*conv_channels, num_out=2*conv_channels, groups=groups, stride=1, norm=norm)
        )

        #==================================================================
        #   decoder --- distance branch
        #==================================================================
        self.upsample1_1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.decoder_block1_1 = MFUnit(2*conv_channels+2*conv_channels, 2*conv_channels, groups=groups, stride=1, norm=norm)

        self.upsample1_2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.decoder_block1_2 = MFUnit(2*conv_channels+conv_channels, conv_channels, groups=groups, stride=1, norm=norm)

        self.upsample1_3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.decoder_block1_3 = MFUnit(conv_channels+n_first, n_first, groups=groups, stride=1, norm=norm)
        self.upsample1_4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

        # num_in, num_out, kernel_size=1, stride=1, padding=None, groups=1, norm=Non
        self.seg_dis = Conv3dBlock(n_first, 1, kernel_size=1, stride=1, norm=norm)

        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()


        #===============================================================
        #    decoder -- binary branch
        #===============================================================
        self.upsample2_1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.decoder_block2_1 = MFUnit(2*conv_channels+2*conv_channels, 2*conv_channels, groups=groups, stride=1, norm=norm)

        self.upsample2_2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.decoder_block2_2 = MFUnit(2*conv_channels+conv_channels, conv_channels, groups=groups, stride=1, norm=norm)

        self.upsample2_3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.decoder_block2_3 = MFUnit(conv_channels+n_first, n_first, groups=groups, stride=1, norm=norm)
        self.upsample2_4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

        # num_in, num_out, kernel_size=1, stride=1, padding=None, groups=1, norm=Non
        self.seg_bin = Conv3dBlock(n_first, 2, kernel_size=1, stride=1, norm=norm)

        # self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        #  further process on the distance
        #  Weights initlization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #TODO: different between method and method_
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant(m.bias, 0.0)

    def forward(self, x):

        x0 = self.first_conv(x)

        #  encoder
        x1 = self.encoder_block1(x0)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)

        #  decoder -- distance branch
        y1_1_toadd = self.upsample1_1(x3)
        y1_1 = torch.cat([x2, y1_1_toadd], dim=1)
        y1_1 = self.decoder_block1_1(y1_1)
        y1_2_toadd = self.upsample1_2(y1_1)
        y1_2 = torch.cat([x1, y1_2_toadd], dim=1)
        y1_2 = self.decoder_block1_2(y1_2)
        y1_3_toadd = self.upsample1_3(y1_2)
        y1_3 = torch.cat([x0, y1_3_toadd], dim=1)
        y1_3 = self.decoder_block1_3(y1_3)
        y1_4_toadd = self.upsample1_4(y1_3)
        y1_4 = self.seg_dis(y1_4_toadd)
        seg_dis = self.sigmoid(y1_4)

        #  decoder -- binary branch
        y2_1 = torch.add(self.upsample2_1(x3), y1_1_toadd)
        y2_1 = torch.cat([x2, y2_1], dim=1)
        y2_1 = self.decoder_block2_1(y2_1)
        y2_2 = torch.add(self.upsample2_2(y2_1), y1_2_toadd)
        y2_2 = torch.cat([x1, y2_2], dim=1)
        y2_2 = self.decoder_block2_2(y2_2)
        y2_3 = torch.add(self.upsample2_3(y2_2), y1_3_toadd)
        y2_3 = torch.cat([x0, y2_3], dim=1)
        y2_3 = self.decoder_block2_3(y2_3)
        y2_4 = torch.add(self.upsample2_4(y2_3), y1_4_toadd)
        y2_4 = self.seg_bin(y2_4)
        seg_bin = self.softmax(y2_4)

        return seg_dis, seg_bin


#   Define DMFNet based on MFNet
class DMFNet(MFNet):
    def __init__(self, in_channels=1, n_first=32, conv_channels=32, groups=16, norm="bn", out_class=2):
        super(DMFNet, self).__init__(in_channels, n_first, conv_channels, groups, norm, out_class)

        #  overwrite the first two encoder blocks of MFNet with DMFNet units.
        self.encoder_block1 = nn.Sequential(
            DMFUnit(n_first, conv_channels, groups=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(conv_channels, conv_channels, groups=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(conv_channels, conv_channels, groups=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )
        self.encoder_block2 = nn.Sequential(
            DMFUnit(conv_channels, 2*conv_channels, groups=groups, stride=2, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2*conv_channels, 2*conv_channels, groups=groups, stride=1, norm=norm, dilation=[1, 2, 3]),
            DMFUnit(2*conv_channels, 2*conv_channels, groups=groups, stride=1, norm=norm, dilation=[1, 2, 3])

        )
