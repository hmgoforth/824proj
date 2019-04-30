import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class EncodeBlock(nn.Module):
    def __init__(self, channels, num_features):
        super(EncodeBlock, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(channels, num_features, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features),
        )
    def forward(self, input_mat):
        block_output = self.encode(input_mat)
        return block_output

class ResidualBlock(nn.Module):
    def __init__(self, channels, num_features=64, classify=True):
        super(ResidualBlock, self).__init__()
        self.classify = classify
        self.conv1 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Conv2d(channels, num_features, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True))
        if classify:
            self.classifier = nn.Linear(num_features, num_features)
    def forward(self, input_mat):
        input_mat = self.conv1(input_mat)
        x = self.conv2(input_mat)
        x += input_mat
        if self.classify:
            x = self.classifier(x.permute(0,2,3,1)) 
            return x.permute(0,3,1,2)
        return x

class BottleNeck(nn.Module):
    def __init__(self, num_features=256):
        super(BottleNeck, self).__init__()
        self.rb1 = ResidualBlock(num_features, num_features)
        self.rb2 = ResidualBlock(num_features, num_features)
        self.rb3 = ResidualBlock(num_features, num_features)
        self.rb4 = ResidualBlock(num_features, num_features)
        self.rb5 = ResidualBlock(num_features, num_features)
        self.rb6 = ResidualBlock(num_features, num_features)
    def forward(self, input_mat):
        x = self.rb1(input_mat)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.rb6(x)
        return x

class GenerativeModel(nn.Module):
    def __init__(self, img_shape=(30, 9, 256, 256), num_features=64):
        super(GenerativeModel, self).__init__()
        #Input is 256x256x9 of DensePose result
        batches, channels, height, width = img_shape

        self.enc_block1 = EncodeBlock(channels, num_features) # output: B x 64 x 256 x 256
        self.enc_block2 = EncodeBlock(num_features, num_features*2) # output: B x 128 x 128 x 128
        self.enc_block3 = EncodeBlock(num_features*2, num_features*4) # output: B x 256 x 64 x 64
 
        self.bottle = BottleNeck() # output: B x 256 x 64 x 64

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256))
 
        self.output = nn.Sequential(
            nn.Conv2d(256, 3, 3, padding=1), 
            nn.Tanh(),
        ) # output: B x 3 x 256 x 256

    def forward(self, dense_image):
        enc1 = self.enc_block1(dense_image) # B x 64 x 256 x 256
        x = F.max_pool2d(enc1, 2, stride=2) # B x 64 x 128 x 128
        enc2 = self.enc_block2(x)           # B x 128 x 128 x 128
        x = F.max_pool2d(enc2, 2, stride=2) # B x 128 x 64 x 64
        enc3 = self.enc_block3(x)           # B x 256 x 64 x 64
        bottle = self.bottle(enc3)             # B x 256 x 64 x 64
        dec1 = self.deconv1(bottle)# B x 256 x 128 x 128
        dec2 = self.deconv2(dec1)  # B x 256 x 256 x 256
        output = self.output(dec2)
        return output

class Blending(nn.Module):
    def __init__(self, im_size=(30, 9, 256, 256), num_features=64):
        super(Blending, self).__init__()
        #Input is output of predictive & warp with the target dense pose
        # predict output: B x 3 x 256 x 256
        # warp image output: B x 3 x 256 x 256
        # target dense pose: B x 3 x 256 x 256
        channels = im_size[1]

        self.conv1 = nn.Conv2d(channels, num_features, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.res1 = ResidualBlock(num_features, num_features)
        self.res2 = ResidualBlock(num_features, num_features)
        self.res3 = ResidualBlock(num_features, num_features, classify=False)
        self.classify = nn.Linear(num_features, 3)
        
    def forward(self, input_mat):
       x1 = self.conv1(input_mat)
       x2 = self.conv2(x1)
       x3 = self.res1(x2)
       x4 = self.res2(x3)
       x5 = self.res3(x4) 
       to_linear = x5.permute(0,2,3,1)
       output = self.classify(to_linear)
       return output.permute(0,3,1,2)

if __name__ == '__main__':
    unet_test_data = torch.randn(10, 9, 256, 256).cuda(async=True)
    #blend_test_data = torch.randn(10, 9, 256, 256).cuda(async=True)

    unet_model = GenerativeModel(unet_test_data.shape, num_features=64)
    #blend_model = Blending(blend_test_data.shape, num_features=64)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        unet_model = unet_model.cuda()
    #    blend_model = blend_model.cuda()

    unet_model.train()
    #blend_model.train()
   
    predicted_image = unet_model(unet_test_data)

    #blended_image = blend_model(blend_test_data)
    pdb.set_trace() 
