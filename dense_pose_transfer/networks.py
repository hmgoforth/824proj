import torch.nn as nn
import torch
import pdb
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from scipy import ndimage
import torchvision.models as M

import utils

''' #################

BEGIN TOP LEVEL GAN NETWORK

''' #################

class DensePoseGAN(nn.Module):
    '''
    Combines DensePoseTransferNet, and Discriminator
    '''
    def __init__(self):
        # initialize predictive module

        # initialize Generator Network
        self.generator = DensePoseTransferNet()

        # initialize Discriminator Network
        self.discriminator = Discriminator()


    def forward(self, source_im, source_iuv, target_iuv, use_gt=False):
        # source_im:  B x (R, G, B) x 256 x 256 float [0, 1]
        # source_iuv: B x (I, U, V) x 256 x 256
        # target_iuv: B x (I, U, V) x 256 x 256

        # Generate predicted image
        # If use_gt, we use the source image as the predictive result.  This is for training the Disc Network
        if use_gt:
            source_body_mask, _ = utils.get_body_and_part_mask_from_iuv(source_iuv)
            target_body_mask, _ = utils.get_body_and_part_mask_from_iuv(target_iuv)
            predictive_result = source_im
        else:
            predictive_result, source_body_mask, target_body_mask = self.generator(source_im, source_iuv, target_iuv)

        # Classify predicted image as either from distribution or not
        classification = self.discriminator(predictive_result, source_body_mask, target_body_mask)

        return predictive_result, classification
        # predictive_result  B x (R, G, B) x 256 x 256
        # classification:    B x 1

''' #################

END TOP LEVEL GAN NETWORK

''' #################

''' #################

BEGIN TOP LEVEL GENERATIVE NETWORK

''' #################

class DensePoseTransferNet(nn.Module):
    '''
    Combines predictive module, warping module, and background inpainting network
    '''
    def __init__(self):
        # initialize predictive module

        # initialize warping module
        self.warping_module = InpaintingAutoencoder()

        # initialize background inpainting module
        self.bg_inpainting = UNet()

        # initialize blending module

    def forward(self, source_im, source_iuv, target_iuv):
        # source_im:  B x (R, G, B) x 256 x 256 float [0, 1]
        # source_iuv: B x (I, U, V) x 256 x 256
        # target_iuv: B x (I, U, V) x 256 x 256

        # predictive
        predictive_result = self.predictive_module(source_im, source_iuv, target_iuv)

        # warping
        source_texture_map = utils.texture_from_images_and_iuv(source_im, source_iuv)
        inpainted_source_texture_map = self.warping_module(source_texture_map)
        warping_result = utils.images_from_texture_and_iuv_batch(inpainted_source_texture_map, source_iuv)

        # background inpainting
        source_body_mask, source_part_mask = utils.get_body_and_part_mask_from_iuv(source_iuv)
        target_body_mask, target_part_mask = utils.get_body_and_part_mask_from_iuv(target_iuv)

        # KEVIN: INPUT MASKS SHOULD BE SCALED TO 255 OR 1?
        # Answer: Masks should be float32's from 0 to 1
        inpainted_bg = self.bg_inpainting(source_im, source_part_mask, target_part_mask)

        # blending
        blending_result = self.blending_module(predictive_result, warping_module)

        # final result
        final_result = utils.combine_foreground_background(blending_result, target_body_mask, inpainted_bg)

        return final_result, source_body_mask, target_body_mask
        # final_result:     B x (R, G, B) x 256 x 256
        # source_body_mask: B x 24 x 256 x 256
        # target_body_mask: B x 24 x 256 x 256

''' #################

END TOP LEVEL GENERATIVE NETWORK

''' #################

''' #################

BEGIN BACKGROUND INPAINTING

''' #################

class UNet(nn.Module):
    """
    UNet for background image inpainting
    """
    def __init__(self, in_c=28, input_shape=256, nf_enc=[64]*2+[128]*9, nf_dec=[128]*4 + [64]):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=nf_enc[0], kernel_size=7, padding=3, stride=1),
            torch.nn.LeakyReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[0], out_channels=nf_enc[1], kernel_size=3, padding=1,stride=2),
            torch.nn.LeakyReLU()               
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[1], out_channels=nf_enc[2], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU()              
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[2], out_channels=nf_enc[3], kernel_size=3, padding=1,stride=2),
            torch.nn.LeakyReLU()              
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[3], out_channels=nf_enc[4], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU()              
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[4], out_channels=nf_enc[5], kernel_size=3, padding=1,stride=2),
            torch.nn.LeakyReLU()              
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[5], out_channels=nf_enc[6], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU()              
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[6], out_channels=nf_enc[7], kernel_size=3, padding=1,stride=2),
            torch.nn.LeakyReLU()              
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[7], out_channels=nf_enc[8], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU()              
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[8], out_channels=nf_enc[9], kernel_size=3, padding=1,stride=2),
            torch.nn.LeakyReLU()              
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[9], out_channels=nf_enc[10], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU()              
        )

        self.deconv0 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[10]+nf_enc[8], out_channels=nf_dec[0], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU()              
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_channels=nf_dec[0]+nf_enc[6], out_channels=nf_dec[1], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU()              
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(in_channels=nf_dec[1]+nf_enc[4], out_channels=nf_dec[2], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU()              
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(in_channels=nf_dec[2]+nf_enc[2], out_channels=nf_dec[3], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU()              
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(in_channels=nf_dec[3]+nf_enc[0], out_channels=nf_dec[4], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU()              
        )

        self.finalConv = nn.Sequential(
            nn.Conv2d(in_channels=nf_dec[4], out_channels=3, kernel_size=3, padding=1,stride=1),
            torch.nn.Tanh()              
        )

        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Init weights using Xavier
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, image, mask, pose):
        """ Input:
            image (batch_sizex3x256x256)
            mask (batch_sizex1x256x256)
            pose (batch_sizex24x256x256)
        """
        # self.img_features.eval()
        # img_feats = self.img_features(image) #no dropout for feature extraction
        # if isinstance(img_feats, tuple):
        #     img_feats = img_feats[2]
        # text_feats = self.text_features(question_encoding)
        x_in = torch.cat((image, mask, pose),dim=1) #256
        x0 = self.conv0(x_in) #256
        x1 = self.conv1(x0)
        x2 = self.conv2(x1) #128
        x3 = self.conv3(x2) 
        x4 = self.conv4(x3) #64
        x5 = self.conv5(x4)
        x6 = self.conv6(x5) #32
        x7 = self.conv7(x6)
        x8 = self.conv8(x7) #16
        x9 = self.conv9(x8)
        xenc = self.conv10(x9) #8

        x = self.upSample(xenc)
        x = torch.cat((x, x8),dim=1)
        x = self.deconv0(x) #16


        x = self.upSample(x)
        x = torch.cat((x, x6),dim=1)
        x = self.deconv1(x) #32

        x = self.upSample(x)
        x = torch.cat((x, x4),dim=1)
        x = self.deconv2(x) #64

        x = self.upSample(x)
        x = torch.cat((x, x2),dim=1)
        x = self.deconv3(x) #128

        x = self.upSample(x)
        x = torch.cat((x, x0),dim=1)
        x = self.deconv4(x) #256, channels=64
        xfinal = self.finalConv(x) #channels now at 3
        # print(xfinal.shape) 
        # Output: torch.Size([batch_size, 3, 256, 256])
        return xfinal

"""

def unet(x_in, pose_in, nf_enc, nf_dec):
    x0 = my_conv(x_in, nf_enc[0], ks=7)  # 256
    x1 = my_conv(x0, nf_enc[1], strides=2)  # 128
    x2 = concatenate([x1, pose_in])
    x3 = my_conv(x2, nf_enc[2])
    x4 = my_conv(x3, nf_enc[3], strides=2)  # 64
    x5 = my_conv(x4, nf_enc[4])
    x6 = my_conv(x5, nf_enc[5], strides=2)  # 32
    x7 = my_conv(x6, nf_enc[6])
    x8 = my_conv(x7, nf_enc[7], strides=2)  # 16
    x9 = my_conv(x8, nf_enc[8])
    x10 = my_conv(x9, nf_enc[9], strides=2)  # 8
    x = my_conv(x10, nf_enc[10])

    skips = [x9, x7, x5, x3, x0]
    filters = [nf_enc[10], nf_dec[0], nf_dec[1], nf_dec[2], nf_enc[3]]

    for i in range(5):
        out_sz = 8*(2**(i+1))
        x = Lambda(interp_upsampling, output_shape = (out_sz, out_sz, filters[i]))(x)
        x = concatenate([x, skips[i]])
        x = my_conv(x, nf_dec[i])

    return x

def my_conv(x_in, nf, ks=3, strides=1, activation='lrelu', name=None):
        x_out = Conv2D(nf, kernel_size=ks, padding='same', strides=strides)(x_in)

        if activation == 'lrelu':
            x_out = LeakyReLU(0.2, name=name)(x_out)
        elif activation != 'none':
            x_out = Activation(activation, name=name)(x_out)

        return x_out

# Background creation
x = unet(concatenate([bg_src, bg_src_mask]), pose_src, [64]*2 + [128]*9, [128]*4 + [64])
"""

''' #################

END BACKGROUND INPAINTING

''' #################

''' #################

BEGIN LOSS FUNCTIONS

''' #################

def vgg_preprocess(x_in):
    z = 255.0 * (x_in + 1.0) / 2.0
    z[:, 0, :, :] -= 103.939
    z[:, 1, :, :] -= 116.779
    z[:, 2, :, :] -= 123.68
    return z

class VGG19FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = M.vgg19(pretrained=True)
        self.features = nn.Sequential(
            *list(self.model.features.children())#[:9]
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, img):
        # Assume input is from -1 to 1 --> preprocess
        # If image is 0 to 1, we are fine
        return self.features(img)
    
def vgg_loss(y_pred, y_true, lmbda=1.0):
    return torch.mean(torch.sum(torch.abs(y_pred - y_true),dim=1))

class Discriminator(nn.Module):
    """
    Discriminator
    """
    def __init__(self, in_c=51):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            # Use BCEWithLogitsLoss so we don't need sigmoid layer
            #nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, img, pose_src, pose_tgt):
        """ Input:
            img (batch_sizex3x256x256)
            pose_src (batch_sizex24x256x256)
            pose_tgt (batch_sizex24x256x256)
        """
        # Concatenate along channels
        x = torch.cat((img, pose_src, pose_tgt), dim=1) #channels = 51
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
                
        return x

"""
def discriminator(param):
    img_h = param['IMG_HEIGHT']
    img_w = param['IMG_WIDTH']
    n_joints = param['n_joints']
    pose_dn = param['posemap_downsample']

    x_tgt = Input(shape=(img_h, img_w, 3))
    x_src_pose = Input(shape=(img_h / pose_dn, img_w / pose_dn, n_joints))
    x_tgt_pose = Input(shape=(img_h / pose_dn, img_w / pose_dn, n_joints))

    x = my_conv(x_tgt, 64, ks=5)
    x = MaxPooling2D()(x) # 128
    x = concatenate([x, x_src_pose, x_tgt_pose])
    x = my_conv(x, 128, ks=5)
    x = MaxPooling2D()(x) # 64
    x = my_conv(x, 256)
    x = MaxPooling2D()(x) # 32
    x = my_conv(x, 256)
    x = MaxPooling2D()(x) # 16
    x = my_conv(x, 256)
    x = MaxPooling2D()(x) # 8
    x = my_conv(x, 256)  # 8

    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[x_tgt, x_src_pose, x_tgt_pose], outputs=y, name='discriminator')
    return model
"""

''' #################

END LOSS FUNCTIONS

''' #################

''' #################

BEGIN WARPING MODULE

''' #################

class InpaintingAutoencoder(nn.Module):
    ''' 
    Inpainting autoencoder (warping module) for person texture inpainting
    '''
    def __init__(self):
        super().__init__()

        self.encoder = BodyEncoder()
        self.context_net = ContextNet()
        self.decoder = BodyDecoder()

    def forward(self, full_texture_maps):
        # input: full_texture_maps, B x 24 x 3 x 256 x 256
        # output: inpainted full_texture_maps, B x 24 x 3 x 256 x 256

        full_texture_embeddings = self.encoder(full_texture_maps)
        context_embeddings = self.context_net(full_texture_embeddings)
        inpainted = self.decoder(context_embeddings)

        return inpainted
    
class ContextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear((24 * 128 * 16 * 16), 256),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU())

    def forward(self, embedding):
        # input: embedding, B x 24 x 128 x 16 x 16
        # output: context_embedding

        B = embedding.shape[0]
        embedding_flat = embedding.view(B, -1)
        out = self.layer1(embedding_flat)
        out = self.layer2(out)
        out = out.unsqueeze(1).unsqueeze(3).unsqueeze(3).repeat(1, 24, 1, 16, 16)

        return torch.cat((out, embedding), dim=2)

class BodyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d((256 + 128) * 24, 64 * 24, 3, stride=2, padding=1, output_padding=1, groups=24),
            nn.ReLU(),
            BodyInstanceNorm(),

            nn.ConvTranspose2d(64 * 24, 32 * 24, 3, stride=2, padding=1, output_padding=1, groups=24),
            nn.ReLU(),
            BodyInstanceNorm(),

            nn.ConvTranspose2d(32 * 24, 32 * 24, 3, stride=2, padding=1, output_padding=1, groups=24),
            nn.ReLU(),
            BodyInstanceNorm(),

            nn.ConvTranspose2d(32 * 24, 3 * 24, 3, stride=2, padding=1, output_padding=1, groups=24),
            nn.Sigmoid(),
        )

    def forward(self, embeddings):
        # input: embeddings, B x 24 x (256 + 128) x 16 x 16
        # output: inpainted result, B x 24 x 3 x 256 x 256
        B = embeddings.shape[0]
        decoder_out = self.decoder(embeddings.view(B, -1, 16, 16))
        return decoder_out.view(B, 24, 3, 256, 256)

class BodyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3 * 24, 32 * 24, 3, stride=2, padding=1, groups=24),
            nn.ReLU(),
            BodyInstanceNorm(),

            nn.Conv2d(32 * 24, 32 * 24, 3, stride=2, padding=1, groups=24),
            nn.ReLU(),
            BodyInstanceNorm(),

            nn.Conv2d(32 * 24, 64 * 24, 3, stride=2, padding=1, groups=24),
            nn.ReLU(),
            BodyInstanceNorm(),

            nn.Conv2d(64 * 24, 128 * 24, 3, stride=2, padding=1, groups=24),
            nn.ReLU(),
            BodyInstanceNorm(),
        )

    def forward(self, body_texture):
        # input: body_texture, B x 24 x 3 x 256 x 256
        # output: encoded body_texture, B x 24 x 128 x 16 x 16

        B = body_texture.shape[0]
        encoder_out = self.encoder(body_texture.view(B, -1, 256, 256))
        return encoder_out.view(B, 24, 128, 16, 16)

class BodyInstanceNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-5

    def forward(self, embedding):
        # input: embedding, B x (24 * C) x H x W
        # output: instance norm embedding, B x (24 * C) x H x W
        # normalize each C x H x W to have zero mean and unit variance

        B = embedding.shape[0]
        C = embedding.shape[1] // 24
        H = embedding.shape[2]
        W = embedding.shape[3]
        embedding_stat = embedding.view(B, 24, -1)
        mean = torch.mean(embedding_stat, dim=2, keepdim=True)
        std = torch.std(embedding_stat, dim=2, keepdim=True)

        embedding_normalize = (embedding_stat - mean) / (std + self.eps)
        embedding_res = embedding_normalize.view(B, 24 * C, H, W)

        return embedding_res

''' #################

END WARPING MODULE

''' #################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test UNet')
    parser.add_argument('--imPath', type=str, default='../posewarp/data/hunter_256.jpeg')
    parser.add_argument('--maskPath', type=str, default='../posewarp/data/hunter_256_IUV.npy')
    args = parser.parse_args()

    # img = Image.open(args.imPath)
    # img = transforms.ToTensor()(img.convert('RGB'))
    # img = img.cuda(async=True)

    img = np.ones((1,3,256,256), dtype=np.float32)
    img = torch.from_numpy(img).cuda(async=True)

    mask = np.ones((1,1,256,256), dtype=np.float32)
    mask = torch.from_numpy(mask).cuda(async=True)

    pose = np.ones((1,24,256,256), dtype=np.float32)
    pose = torch.from_numpy(pose).cuda(async=True)

    model = UNet().cuda()

    model.eval()
    output = model(img,mask,pose)
    print(output.shape)

    model2 = VGG19FeatureNet().cuda()
    model2.eval()
    y_true = model2(img)
    y_pred = model2(img)
    print(y_pred.shape)
    loss = vgg_loss(y_pred, y_true)
    print(loss)

    model3 = Discriminator().cuda()
    model3.eval()
    disc_output = model3(img, pose, pose)
    print(disc_output)
    print(disc_output.shape)


