from torch.nn.modules.conv import Conv3d
import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

class MNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool3d((2,2,2), stride=(2,2,2))
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv3d(64,num_classes, 1)

        self.e1_e3 = nn.Conv3d(64, 256, kernel_size=3, stride=(2,2,2), padding=1)
        self.conv_mid2 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)

        self.e2_e4 = nn.Conv3d(128, 512, kernel_size=3, stride=(2,2,2), padding=1)
        self.conv_mid1 = nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3) 
        
        x = self.dconv_down4(x)

        mid = self.e2_e4(conv2)
        mid = self.maxpool(mid)
        print(mid.shape, x.shape)
        x = torch.cat([mid,x], dim=1)
        x = self.conv_mid1(x)


        x = self.upsample(x)

        mid = self.e1_e3(conv1)
        mid = self.maxpool(mid)
        mid = torch.cat([conv3,mid], dim=1)
        mid = self.conv_mid2(mid)

        x = torch.cat([x, mid], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)   
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    
if __name__ == "__main__":
  
  x = MNet()
  print(sum(p.numel() for p in x.parameters()))
  print(x(torch.randn(1,1,128,128,16)).shape)

