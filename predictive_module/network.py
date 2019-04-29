import torch
import torch.nn as nn
import pdb

class DecodeBlock(nn.Module):
    def __init__(self, channels, num_features):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(channels, num_features, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features),
            nn.Conv2d(num_features, num_features, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features),
            nn.MaxPool2d(2, stride=2),
        )
    def forward(self, input_mat):
        block_output = self.decode(input_mat)
        return block_output

class Decode(nn.Module):
    def __init__(self, channels=64, num_features=64):
        super().__init__()
        self.decode_blocks = nn.Sequential(
            DecodeBlock(channels, num_features),
            DecodeBlock(num_features, num_feautres*2),
            DecodeBlock(num_features*2, num_features*4),
            DecodeBlock(num_features*4, num_features*8),
        )
    def forward(self, image):
        decoding = self.decode_blocks(image)
        return decoding

class BottleNeck(nn.Module):
    def __init__(self, channels=512, num_features=64, drop=0.4):
        super().__init__():
        self.bottle = nn.Sequential(
            nn.Dropout(drop),
            nn.Conv2d(channels, num_features*16, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features*16),
            nn.Conv2d(num_features*16, num_features*16, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features*16),
        )
    def forward(self, input_mat):
        bottled_out = self.bottle(input_mat)
        return bottled_out

class EncodeBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__():
        self.deconv = nn.Sequential(
            nn.Upsample(size=(num_features, num_features), scale_factor=(2, 2), mode="bilinear"),
            nn.ReLu(),
            nn.BatchNorm2d(num_features),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features/2, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features/2),
            nn.Conv2d(num_features/2, num_features/2, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features/2),
        )

   def _crop_cat(self, upsampled, bypass, crop=False):
       if crop:
           c = (bypass.size()[2] - upsampled.size()[2]) // 2
           bypass = F.pad(bypass, (-c, -c, -c, -c))
       return torch.cat((upsampled, bypass), 1)

   def forward(self, input_mat, concat_mat):
        deconv = self.deconv(input_mat)
        catted = self._crop_cat(deconv, concat_mat)
        conv = self.conv(catted)
        return conv
        
class Encode(nn.Module):
    def __init__(self, num_features=64):
        super().__init__()
        self.encode_blocks = nn.Sequential(
            EncodeBlock(num_features*16),
            EncodeBlock(num_features*8),
            EncodeBlock(num_features*4),
            EncodeBlock(num_features*2),
        )
    def forward(self, image):
        encoding = self.encode_blocks(image)
        return encoding



class GenerativeModel(nn.Module):
    def __init__(self, img_shape=(256, 256, 9)):
        super().__init__()
        #Input is 256x256x9 of DensePose result
        height, width, channels = img_shape
        filters = 64

        self.decode = Decode(channels, filters)
        self.bottle = BottleNeck(channels*8, filters, dropout=0.4)
        self.encode = Encode(filters)
        self.output = nn.Conv2d(filters, channels, kernel_size=(1,1), padding=0, stride=1)

    def forward(self, dense_image):
        x = self.decode(dense_image)
        x = self.bottle(x)
        x = self.encode(x)
        x = self.output(x)
        return x
