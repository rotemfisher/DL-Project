import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"))
        else:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False))
        
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True) if not down else nn.LeakyReLU(0.2, inplace=True))
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ClockTranslator(nn.Module):
    """
    A simplified U-Net architecture for translating digital clock images to analog.
    """
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(ClockTranslator, self).__init__()
        
        # Encoder (Downsampling)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        ) # 128
        
        self.down1 = UNetBlock(features, features * 2, down=True) # 64
        self.down2 = UNetBlock(features * 2, features * 4, down=True) # 32
        self.down3 = UNetBlock(features * 4, features * 8, down=True) # 16
        self.down4 = UNetBlock(features * 8, features * 8, down=True) # 8
        self.down5 = UNetBlock(features * 8, features * 8, down=True) # 4
        self.down6 = UNetBlock(features * 8, features * 8, down=True) # 2
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"), 
            nn.ReLU()
        ) # 1x1
        
        # Decoder (Upsampling)
        self.up1 = UNetBlock(features * 8, features * 8, down=False, use_dropout=True)
        self.up2 = UNetBlock(features * 8 * 2, features * 8, down=False, use_dropout=True)
        self.up3 = UNetBlock(features * 8 * 2, features * 8, down=False, use_dropout=True)
        self.up4 = UNetBlock(features * 8 * 2, features * 8, down=False)
        self.up5 = UNetBlock(features * 8 * 2, features * 4, down=False)
        self.up6 = UNetBlock(features * 4 * 2, features * 2, down=False)
        self.up7 = UNetBlock(features * 2 * 2, features, down=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), # Output pixels in [-1, 1]
        )

    def forward(self, x):
        # Encoder
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        
        # Middle
        bn = self.bottleneck(d7)
        
        # Decoder with Skip Connections
        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        
        return self.final_up(torch.cat([u7, d1], dim=1))

if __name__ == "__main__":
    # Test the model with a dummy input
    x = torch.randn((1, 3, 256, 256))
    model = ClockTranslator()
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
