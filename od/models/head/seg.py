import torch
import torch.nn as nn
import torch.nn.functional as F
from od.models.modules.common import Conv, Concat, BottleneckCSP


class SegmentationHead(nn.Module):
    ''' 
    SegmentationHead for Yolov5

            head structure:
                                                      PAN output PP3
                                                            |
                                                            V
                                                    inner_h4([1, 128, 76, 76])
                                                            |
                                                            | upsample
                                          concat            |
        outer_c3([1, 128, 152, 152])  --------------------->|
                                                            |
                                                            |
                                                            v
                                                    [1, 256, 152, 152]
                                                            |
                                                          CSP2_1
                                                            |
                                                           CBL
                                                            |
                                                            | upsample
                                                            |
                                          concat            |
        outer_f2([1, 64, 304, 304])  ---------------------> |
                                                            |
                                                            |
                                                            v
                                                    [1, 320, 304, 304]
                                                            |
                                                          CSP2_1
                                                            |
                                                           CBL
                                                            |
                                                            | upsample
                                                            |
                                                           conv
                                                            |
                                                            v
                                                    [1, nc, 608, 608]
    '''
    def __init__(self, nc, inner_h2=320, inner_h3=256, inner_h4=128, outer_f2=64, outer_c3=128):  # segmentation layer
        super(SegmentationHead, self).__init__()
        
        self.channels_out = {
            'inner_h2': inner_h2, # img_size: (608/(2**2)) 
            'inner_h3': inner_h3, # img_size: (608/(2**3))
            'inner_h4': inner_h4, # img_size: (608/(2**4))
            'outer_f2': outer_f2, # Focus output, img_size: (608/(2**2))
            'outer_c3': outer_c3, # second CBL output, img_size: (608/2**3))
        }

        self.nc = nc  # number of classes

        self.inner_h2 = self.channels_out['inner_h2']
        self.inner_h3 = self.channels_out['inner_h3']
        self.inner_h4 = self.channels_out['inner_h4']
        self.outer_f2 = self.channels_out['outer_f2']
        self.outer_c3 = self.channels_out['outer_c3']

        self.concat = Concat(dimension=1)
        self.Upsample_P3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.CSP_P3 = BottleneckCSP(self.inner_h3, self.inner_h3, n=2)
        self.CBL_P3 = Conv(self.inner_h3, self.inner_h3)
        
        self.Upsample_P2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.CSP_P2 = BottleneckCSP(self.inner_h2, self.inner_h2, n=2)
        self.CBL_P2 = Conv(self.inner_h2, self.inner_h2)
 
        self.Upsample_P1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(320, self.nc, kernel_size=1, stride=1)

    def forward(self, PP3, F2, C3):
        # PP3, F2, C3 = inputs
        up_3 = self.Upsample_P3(PP3)
        # print("PP3:", PP3.shape)
        # print("F2:", F2.shape)
        # print("C3:", C3.shape)
        # print("up_3:", up_3.shape)
        up_3 = self.concat([up_3, C3])
        up_3 = self.CSP_P3(up_3)
        up_3 = self.CBL_P3(up_3)
        up_2 = self.Upsample_P2(up_3)

        up_2 = self.concat([up_2, F2])
        up_2 = self.CSP_P2(up_2)
        up_2 = self.CBL_P2(up_2)
        up_1 = self.Upsample_P1(up_2)
        out = self.conv(up_1)
        return out

if __name__ == '__main__':
    seg = SegmentationHead(nc=5)
    inputs = [torch.rand(1, 128, 76, 76), torch.rand(1, 64, 304, 304), torch.rand(1, 128, 152, 152)]
    out = seg.forward(inputs)
    print("out:", out.shape)