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
    def __init__(self, channels, num_features=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Conv2d(channels, num_features, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU())
    def forward(self, input_mat):
        input_mat = self.conv1(input_mat)
        x = self.conv2(input_mat)
        x += input_mat
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

class PredictiveModel(nn.Module):
    def __init__(self, img_shape=(30, 9, 256, 256), num_features=64):
        super(PredictiveModel, self).__init__()
        #Input is 256x256x9 of DensePose result
        batches, channels, height, width = img_shape

        self.enc_block1 = EncodeBlock(9, 64) # output: B x 64 x 256 x 256
        self.enc_block2 = EncodeBlock(64, 128) # output: B x 128 x 128 x 128
        self.enc_block3 = EncodeBlock(128, 256) # output: B x 256 x 64 x 64
 
        self.bottle = BottleNeck() # output: B x 256 x 64 x 64

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256))
 
        self.output = nn.Sequential(
            nn.Conv2d(256, 3, 3, padding=1), 
            nn.Tanh(),
        ) # output: B x 3 x 256 x 256

    def forward(self, x):
        x = self.enc_block1(x) # B x 64 x 256 x 256
        x = F.max_pool2d(x, 2, stride=2) # B x 64 x 128 x 128
        x = self.enc_block2(x)           # B x 128 x 128 x 128
        x = F.max_pool2d(x, 2, stride=2) # B x 128 x 64 x 64
        x = self.enc_block3(x)           # B x 256 x 64 x 64
        x = self.bottle(x)             # B x 256 x 64 x 64
        x = self.deconv1(x)# B x 256 x 128 x 128
        x = self.deconv2(x)  # B x 256 x 256 x 256
        x = self.output(x)
        return x

class Blending(nn.Module):
    def __init__(self, im_size=(30, 9, 256, 256), num_features=64):
        super(Blending, self).__init__()
        #Input is output of predictive & warp with the target dense pose
        # predict output: B x 3 x 256 x 256
        # warp image output: B x 3 x 256 x 256
        # target dense pose: B x 3 x 256 x 256
        channels = im_size[1]

        self.conv1 = nn.Conv2d(9, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 64)
        self.res3 = ResidualBlock(64, 64)
        self.classify = nn.Linear(num_features, 3)
        
    def forward(self, x):
       x = self.conv1(x)
       x = self.conv2(x)
       x = self.res1(x)
       x = self.res2(x)
       x = self.res3(x) 
       x = x.permute(0, 2, 3, 1)
       x = self.classify(x)
       return x.permute(0, 3, 1, 2)


if __name__ == '__main__':
    #unet_test_data = torch.randn(10, 9, 256, 256).cuda(async=True)
    pdb.set_trace()
    blend_test_data = torch.randn(10, 9, 256, 256).cuda()

    #unet_model = PredictiveModel(unet_test_data.shape, num_features=64)
    blend_model = Blending(blend_test_data.shape, num_features=64)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
    #    unet_model = unet_model.cuda()
        blend_model = blend_model.cuda()

    #unet_model.train()
    blend_model.train()
   
    #predicted_image = unet_model(unet_test_data)

    blended_image = blend_model(blend_test_data)
    pdb.set_trace()

    main() 
