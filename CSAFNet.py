import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
from utils.tensor_ops import cus_sample, upsample_add
from backbone.VGG import (
    Backbone_VGG_in1,
    Backbone_VGG_in3,
)
import os
from module.MyModules import (
    AFM,
    MFA,
    PAWithKL,
    Up,
    Up16
)
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

class DEFNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DEFNet, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        #RGB Encoder
        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        )= Backbone_VGG_in3(pretrained=pretrained)
        #Thermal Encoder
        (
            self.depth_encoder1,
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
        ) = Backbone_VGG_in1(pretrained=pretrained)


        #定义窗口对齐
        self.AlignR_1 = AFM(64)
        self.AlignR_2 = AFM(128)
        self.AlignR_4 = AFM(256)
        self.AlignR_8 = AFM(512)
        self.AlignR_16 = AFM(512)
        #定义反卷积
        self.up_16 = Up16(512)
        self.up_8 = Up(512)
        self.up_4 = Up(256)
        self.up_2 = Up(128)



        self.deconv_16 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn16 = nn.BatchNorm2d(512)
        self.deconv_8 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.deconv_4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv_16_fusion = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn16_f = nn.BatchNorm2d(512)
        self.deconv_8_fusion = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn8_f = nn.BatchNorm2d(256)
        self.deconv_4_fusion = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn4_f = nn.BatchNorm2d(128)
        self.deconv_2_fusion = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn2_f = nn.BatchNorm2d(64)
        self.conv_1_fusion = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1_f = nn.BatchNorm2d(128)





        self.CAKL1 = PAWithKL(64)
        self.CAKL2 = PAWithKL(128)
        self.CAKL4 = PAWithKL(256)
        self.CAKL8 = PAWithKL(512)
        self.CAKL16 = PAWithKL(512)



        self.MFA8 = MFA(512)
        self.MFA4 = MFA(256)
        self.MFA2 = MFA(128)
        self.MFA1 = MFA(64)


        self.reg_layer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )


        # self.fdm = FDM()



    def forward(self, RGBT):
        in_data = RGBT[0]
        in_depth = RGBT[1]

        #图像进入模型，由（2，3，256，256）→（2，64，256，256），后缀_d代表其他模态数据
        in_data_1 = self.encoder1(in_data)
        del in_data
        in_data_1_d = self.depth_encoder1(in_depth)
        del in_depth

        # 由（2，64，256，256）→（2，128，128，128）
        in_data_2 = self.encoder2(in_data_1)
        in_data_2_d = self.depth_encoder2(in_data_1_d)

        # 由（2，128，128，128）→（2，256，64，64）
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)

        # 由（2，256，64，64）→（2，256，32，32）
        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)

        # 由（2，256，32，32）→（2，512，16，16）
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)


        #窗口内空间对齐后的RGB特征
        in_data_16_Align = self.AlignR_16(in_data_16_d, in_data_16)
        in_data_8_Align = self.AlignR_8(in_data_8_d, in_data_8)
        in_data_4_Align = self.AlignR_4(in_data_4_d, in_data_4)
        in_data_2_Align = self.AlignR_2(in_data_2_d, in_data_2)
        in_data_1_Align = self.AlignR_1(in_data_1_d, in_data_1)

        in_data_16_Align_up = self.up_16(in_data_16_Align)
        in_data_8_Align_up = self.up_8(in_data_8_Align)
        in_data_4_Align_up = self.up_4(in_data_4_Align)
        in_data_2_Align_up = self.up_2(in_data_2_Align)



        in_data_16_Align_ms = in_data_16_Align
        in_data_8_Align_ms = self.MFA8(in_data_8_Align,in_data_16_Align_up)
        in_data_4_Align_ms = self.MFA4(in_data_4_Align, in_data_8_Align_up)
        in_data_2_Align_ms = self.MFA2(in_data_2_Align, in_data_4_Align_up)
        in_data_1_Align_ms = self.MFA1(in_data_1_Align, in_data_2_Align_up)

        in_data_16_fusion = self.CAKL16(in_data_16_d, in_data_16_Align_ms)
        in_data_8_fusion = self.CAKL8(in_data_8_d, in_data_8_Align_ms)
        in_data_4_fusion = self.CAKL4(in_data_4_d, in_data_4_Align_ms)
        in_data_2_fusion = self.CAKL2(in_data_2_d, in_data_2_Align_ms)
        in_data_1_fusion = self.CAKL1(in_data_1_d, in_data_1_Align_ms)


        #Feature up
        in_data_16_fusion_up = F.relu(self.bn16_f(self.deconv_16_fusion(in_data_16_fusion)))

        in_data_8_fusion_up = torch.cat((in_data_8_fusion,in_data_16_fusion_up),1)
        in_data_8_fusion_up = F.relu(self.bn8_f(self.deconv_8_fusion(in_data_8_fusion_up)))
        in_data_4_fusion_up = torch.cat((in_data_4_fusion,in_data_8_fusion_up),1)
        in_data_4_fusion_up = F.relu(self.bn4_f(self.deconv_4_fusion(in_data_4_fusion_up)))
        in_data_2_fusion_up = torch.cat((in_data_2_fusion, in_data_4_fusion_up),1)
        in_data_2_fusion_up = F.relu(self.bn2_f(self.deconv_2_fusion(in_data_2_fusion_up)))
        in_data_1_fusion_up = torch.cat((in_data_1_fusion, in_data_2_fusion_up),1)
        in_data_1_fusion_up = F.relu(self.bn1_f(self.conv_1_fusion(in_data_1_fusion_up)))


        out = self.reg_layer(in_data_1_fusion_up)
        return out



def fusion_model():
    model = DEFNet()
    return model

if __name__ == "__main__":
    model = DEFNet()
    x = torch.randn(2,3,256,256)
    depth = torch.randn(2,3,256,256)
    fuse = model([x,depth])
    print(fuse.shape)



