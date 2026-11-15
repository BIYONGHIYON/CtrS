import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class Resbackbone(nn.Module):
    def __init__(self, in_channels):
        super(Resbackbone, self).__init__()
        channel = 64
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channel, kernel_size=3, padding=1, stride=1,
                               bias=True)

        self.res1 = ResBlock(channel)
        self.res2 = ResBlock(channel)
        self.res3 = ResBlock(channel)
        self.res4 = ResBlock(channel)

        self.backbone = nn.Sequential(
            self.res1,
            self.res2,
            self.res3,
            self.res4,

        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=1, stride=1, kernel_size=3, padding=1,
                               bias=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.backbone(x)
        x = self.conv2(x)
        return x


class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 64

        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, padding=1, kernel_size=3, stride=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, padding=1, kernel_size=3, stride=1,
                                bias=True)
        self.relu = nn.ReLU(
            inplace=True) 
    def forward(self, x):
        rs1 = self.relu(self.conv20(x))
        rs2 = self.conv21(rs1)  
        # print(f"rs2:{rs2.shape},x:{x.shape}")
        rs = torch.add(x, rs2) 
        return rs


class SSA(nn.Module):
    def __init__(self,channels):
        super(SSA, self).__init__()

        self.fixed  = 4


        self.conv1t6 = nn.Conv2d(in_channels=1, out_channels=self.fixed , kernel_size=3, stride=1, padding=1,
                                 bias=True)
        self.conv6t12 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv6t6 = nn.Conv2d(in_channels=self.fixed , out_channels=self.fixed , kernel_size=3, stride=1, padding=1,
                                 bias=True)
        self.relu = nn.ReLU()
        self.conv7t6_3 = nn.Conv2d(in_channels=channels + 1, out_channels=self.fixed, kernel_size=3, stride=1,
                                   padding=1, bias=True)

    def forward(self, pan, ms_pro):  # ms_pro 6xhxw  pan 1xhxw
        pan_guide = self.relu(self.conv1t6(pan))
        pan_guide = self.conv6t6(pan_guide)  # 6*h*w
        ms_pro = self.conv7t6_3(ms_pro)
        # pan reshape -> 6xhw
        # ms_pro reshape -> 6xhw  -> transpose > 6xwh
        b, c, h, w = ms_pro.size()
        pan_reshape = pan_guide.view(b, c, -1)  # 6xhw
        ms_pro_transpose = ms_pro.permute(0, 1, 3, 2)
        ms_pro_reshape = ms_pro_transpose.reshape(b, c, -1)  # 6xhw
        # print(ms_pro_reshape.shape)
        # ms使用了转置

        # pan_matrix * ms_pro_matrix -> 6xhwxwh -> softmax ->6xhxw
        pan_matrix = torch.mul(pan_reshape, ms_pro_reshape)  # b x c x hw x wh
        # print(pan_matrix.shape)
        pan_matrix = pan_matrix.view(b, c, h, w)
        # print(pan_matrix.shape)
        pan_matrix = pan_matrix.permute(0, 1, 3, 2)
        pan_matrix = pan_matrix.contiguous().view(b, c, -1)
       
        pan_matrix = F.softmax(pan_matrix, dim=-1)
        # print(pan_matrix.shape)

        # transpose_R * ms_pro_matrix -> 6xhwxwh -> reshape -> output= 6xhxw
        output = torch.mul(pan_reshape, pan_matrix)  # b x c x hw
        output = output.contiguous().view(b, c, h, w)  # reshape to 6xhxw

        # print(f"res{res.shape}")
        return output







class PansharpeningNet(nn.Module):
    def __init__(self, channels):
        super(PansharpeningNet, self).__init__()

        self.fixed = 4

        # Downsampling and Upsampling layers
        self.downsample1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.downsample2 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.downsample200 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        # Feature extraction and fusion layers with individual names
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()

        self.conv1t64 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv64tnum = nn.Conv2d(64, channels, kernel_size=3, padding=1)
        self.convnumt64 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)

        self.conv1t64_pan = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv64tnum_pan = nn.Conv2d(64, channels, kernel_size=3, padding=1)

        self.convnum3tnum1 = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)
        self.convnum3tnum2 = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)
        self.convnum3tnum3 = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)

        self.conv48tnum1 = nn.Conv2d(in_channels=self.fixed * channels, out_channels=channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)
        self.conv48tnum2 = nn.Conv2d(in_channels=self.fixed * channels, out_channels=channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)
        self.conv48tnum3 = nn.Conv2d(in_channels=self.fixed * channels, out_channels=channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)

        self.conv7t6_1 = nn.Conv2d(in_channels=channels + 1, out_channels=self.fixed, kernel_size=3, stride=1,
                                   padding=1, bias=True)
        self.conv7t6_2 = nn.Conv2d(in_channels=channels + 1, out_channels=self.fixed, kernel_size=3, stride=1,
                                   padding=1, bias=True)
        self.conv7t6_3 = nn.Conv2d(in_channels=channels + 1, out_channels=self.fixed, kernel_size=3, stride=1,
                                   padding=1, bias=True)

        self.conv2ctc1 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                   bias=True)
        self.conv2ctc2 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                   bias=True)
        self.conv2ctc3 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                   bias=True)

        self.SSA_blocks = nn.ModuleList([SSA(channels=channels) for _ in range(channels)])

        self.SSA_blocks1 = nn.ModuleList([SSA(channels=channels) for _ in range(channels)])
        self.SSA_blocks2 = nn.ModuleList([SSA(channels=channels) for _ in range(channels)])
        self.conv1t1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        self.cov2t64 = nn.Conv2d(in_channels=2 + channels, out_channels=64, kernel_size=3, stride=1, padding=1,
                                 bias=True)

        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.backbone = nn.Sequential(
                self.res1,
                self.res2,
                self.res3,
                self.res4,
            )
        self.conv64t48 = nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.fixed = 4;

        # Downsampling and Upsampling layers
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample1 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample100 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample100 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample101  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample102\
            = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Fusion layers
        self.resblock = Resblock()
        self.prelu = nn.PReLU()
        self.conv1t64 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv64tnum = nn.Conv2d(64, channels, kernel_size=3, padding=1)
        self.convnumt64 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)

        self.conv1t64bf2 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv64tnumbf2 = nn.Conv2d(64, channels, kernel_size=3, padding=1)
        self.convnumt64bf2 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)

        self.convnumtnum = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1)
        self.convctc1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # convctc1
        self.convctc2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # convctc2
        self.convnum3tnum = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels * 6, stride=1, kernel_size=3, padding=1,
                               bias=True)

        self.SSA_blocks = nn.ModuleList(
            [SSA(channels=channels) for _ in range(channels)])  # Adding multi-scale blocks

        self.conv48tnum1 = nn.Conv2d(in_channels=self.fixed * channels, out_channels=channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)
        self.conv48tnum2 = nn.Conv2d(in_channels=self.fixed * channels, out_channels=channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)

        self.conv48t48_1 = nn.Conv2d(in_channels=6 * channels, out_channels=6 * channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)
        self.conv48t48_2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                     bias=True)
        self.conv48t48_3 = nn.Conv2d(in_channels=6 * channels, out_channels=6 * channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)

        self.conv7t6 = nn.Conv2d(in_channels=channels + 1, out_channels=self.fixed, kernel_size=3, stride=1, padding=1,
                                 bias=True)

        self.convnum3tn64 = nn.Conv2d(in_channels=channels * 3, out_channels=64, kernel_size=3, stride=1, padding=1,
                                      bias=True)

        self.group_size = 6 
        self.channels = channels
        self.cov2t64 = nn.Conv2d(in_channels=2 + channels, out_channels=64, kernel_size=3, stride=1, padding=1,
                                 bias=True)
        self.conv64t48 = nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1t1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

        self.convnumt48 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                    bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.conv8t4 = nn.Conv2d(in_channels=8, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                    bias=True)



   
    def forward(self, pan, lms):
        # Stage 1: Downsample and upsample each input
        pan_down_up = self.upsample1(self.downsample1(pan))
        pan_updown = self.prelu(self.conv1t1(pan_down_up))
        input = torch.cat([lms, pan, pan_updown], dim=1)

        y = self.cov2t64(input)
        y = self.backbone(y)
        y1 = self.conv64t48(y)
        rs = y1 + lms

        # Process input channels separately
        split_input = [rs[:, i:i + self.group_size, :, :] for i in range(0, 48, self.group_size)]
        split_ms = [lms[:, i, :, :].unsqueeze(1) for i in range(self.channels)]

        outputs = []
        for input_channel, ms_channel, ssa in zip(split_input, split_ms, self.SSA_blocks):
            x = torch.cat([rs, ms_channel], dim=1)

            ms_one = ssa(pan, x)
            outputs.append(ms_one)

        pan_ms = torch.cat(outputs, dim=1)
        pan_ms = self.prelu1(self.conv48tnum1(pan_ms))
        fsm_msb = rs + pan_ms
        res = fsm_msb

        res_down = self.downsample100(res)

        # Stage 2: MS upsample and next stage fusion
        ms_up = self.upsample100(ms)
        rs = res_down

        split_input = [rs[:, i:i + self.group_size, :, :] for i in range(0, 48, self.group_size)]
        split_ms = [ms_up[:, i, :, :].unsqueeze(1) for i in range(self.channels)]

        outputs = []
        for input_channel, ms_channel, ssa in zip(split_input, split_ms, self.SSA_blocks1):
            x = torch.cat([rs, ms_channel], dim=1)

            pan_down = self.downsample100(pan)
            ms_one = ssa(pan_down, x)
            outputs.append(ms_one)

        pan_ms = torch.cat(outputs, dim=1)
        pan_ms = self.prelu2(self.conv48tnum2(pan_ms))
        fsm_msb = rs + pan_ms
        res_down = fsm_msb

        # Stage 3: Small scale stage
        res_small = self.downsample200(res_down)
        rs = res_small

        split_input = [rs[:, i:i + self.group_size, :, :] for i in range(0, 48, self.group_size)]
        split_ms = [ms[:, i, :, :].unsqueeze(1) for i in range(self.channels)]

        outputs = []
        for input_channel, ms_channel, ssa in zip(split_input, split_ms, self.SSA_blocks2):
            x = torch.cat([rs, ms_channel], dim=1)

            pan_down = self.downsample2(pan)
            ms_one = ssa(pan_down, x)
            outputs.append(ms_one)

        pan_ms = torch.cat(outputs, dim=1)
        pan_ms = self.prelu3(self.conv48tnum3(pan_ms))
        res_small = pan_ms

        # Output processing
        ms_output = ms
        ms_output = torch.cat([ms_output, res_small], dim=1)
        ms_output = self.conv2ctc1(ms_output)

        ms_output = self.upsample101(ms_output)
        ms_output = torch.cat([ms_output, res_down], dim=1)
        ms_output = self.conv2ctc2(ms_output)

        ms_output = self.upsample102(ms_output)
        ms_output = torch.cat([ms_output, res], dim=1)
        ms_output = self.conv2ctc3(ms_output)




        return ms_output
