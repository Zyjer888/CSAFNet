import torch
import torch.nn as nn
import torch.nn.functional as F
import os


from utils.tensor_ops import cus_sample, upsample_add
from backbone.VGG import (
    Backbone_VGG_in1,
    Backbone_VGG_in3,
)
from module.MyModules import (
    CFSA,
    MSDC,
    CFDF,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class CSAFNet(nn.Module):
    def __init__(self, pretrained=True):
        super(CSAFNet, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.encoder1, self.encoder2, self.encoder4, self.encoder8, self.encoder16 = Backbone_VGG_in3(pretrained=pretrained)
        self.depth_encoder1, self.depth_encoder2, self.depth_encoder4, self.depth_encoder8, self.depth_encoder16 = Backbone_VGG_in1(pretrained=pretrained)

        self.AlignR_1 = CFSA(64)
        self.AlignR_2 = CFSA(128)
        self.AlignR_4 = CFSA(256)
        self.AlignR_8 = CFSA(512)
        self.AlignR_16 = CFSA(512)



        self.CAKL1 = CFDF(64)
        self.CAKL2 = CFDF(128)
        self.CAKL4 = CFDF(256)
        self.CAKL8 = CFDF(512)
        self.CAKL16 = CFDF(512)

        self.MFA8 = MSDC(512)
        self.MFA4 = MSDC(256)
        self.MFA2 = MSDC(128)
        self.MFA1 = MSDC(64)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(128, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.Conv1 = nn.Conv2d(512, 256, 3, 1, 1)
        self.Conv2 = nn.Conv2d(256, 128, 3, 1, 1)
        self.Conv3 = nn.Conv2d(128, 64, 3, 1, 1)

    def forward(self, RGBT):
        in_data, in_depth = RGBT

        in_data_1 = self.encoder1(in_data)
        in_data_1_d = self.depth_encoder1(in_depth)

        in_data_2 = self.encoder2(in_data_1)
        in_data_2_d = self.depth_encoder2(in_data_1_d)

        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)

        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)

        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)

        in_data_16_Align = self.AlignR_16(in_data_16_d, in_data_16)
        in_data_8_Align = self.AlignR_8(in_data_8_d, in_data_8)
        in_data_4_Align = self.AlignR_4(in_data_4_d, in_data_4)
        in_data_2_Align = self.AlignR_2(in_data_2_d, in_data_2)
        in_data_1_Align = self.AlignR_1(in_data_1_d, in_data_1)


        in_data_16_Align_ms = in_data_16_Align
        in_data_8_Align_ms = self.MFA8(in_data_8_Align, in_data_16_Align)
        in_data_4_Align_ms = self.MFA4(in_data_4_Align, in_data_8_Align)
        in_data_2_Align_ms = self.MFA2(in_data_2_Align, in_data_4_Align)
        in_data_1_Align_ms = self.MFA1(in_data_1_Align, in_data_2_Align)

        in_data_16_fusion = self.CAKL16(in_data_16_d, in_data_16_Align_ms)
        in_data_8_fusion = self.CAKL8(in_data_8_d, in_data_8_Align_ms)
        in_data_4_fusion = self.CAKL4(in_data_4_d, in_data_4_Align_ms)
        in_data_2_fusion = self.CAKL2(in_data_2_d, in_data_2_Align_ms)
        in_data_1_fusion = self.CAKL1(in_data_1_d, in_data_1_Align_ms)

        in_data_16_fusion_up = F.relu(self.bn16_f(self.deconv_16_fusion(in_data_16_fusion)))
        in_data_8_fusion_up = torch.cat((in_data_8_fusion, in_data_16_fusion_up), 1)
        in_data_8_fusion_up = F.relu(self.bn8_f(self.deconv_8_fusion(in_data_8_fusion_up)))
        in_data_4_fusion_up = torch.cat((in_data_4_fusion, in_data_8_fusion_up), 1)
        in_data_4_fusion_up = F.relu(self.bn4_f(self.deconv_4_fusion(in_data_4_fusion_up)))
        in_data_2_fusion_up = torch.cat((in_data_2_fusion, in_data_4_fusion_up), 1)
        in_data_2_fusion_up = F.relu(self.bn2_f(self.deconv_2_fusion(in_data_2_fusion_up)))
        in_data_1_fusion_up = torch.cat((in_data_1_fusion, in_data_2_fusion_up), 1)
        in_data_1_fusion_up = F.relu(self.bn1_f(self.conv_1_fusion(in_data_1_fusion_up)))

        out = self.reg_layer(in_data_1_fusion_up)
        return out

def fusion_model():
    return CSAFNet()

if __name__ == "__main__":
    model = CSAFNet().to(device)
    model.eval()
    dummy_rgb = torch.randn(1, 3, 256, 256).to(device)
    dummy_depth = torch.randn(1, 3, 256, 256).to(device)
