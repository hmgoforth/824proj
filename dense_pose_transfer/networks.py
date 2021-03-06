import torch.nn as nn
import torch
import pdb
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from scipy import ndimage
import torchvision.models as M
import torch.nn.functional as F

from pdb import set_trace as st

from collections import OrderedDict

import utils

NORM = transforms.Normalize(
    mean[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

''' #################

BEGIN TOP LEVEL GAN NETWORK

''' #################

class DensePoseGAN(nn.Module):
    '''
    Combines DensePoseTransferNet, and Discriminator
    '''
    def __init__(self, pretrained_person_inpainter=None, pretrained_background_inpainter=None, pretrained_predictive_module=None, debug=False):
        super().__init__()
        self._debug = debug
        # initialize Generator Network
        self.generator = DensePoseTransferNet(pretrained_person_inpainter, pretrained_background_inpainter, pretrained_predictive_module)

        # initialize Discriminator Network
        self.discriminator = Discriminator()


    def forward(self, source_im, source_iuv, target_iuv, use_gt=False):
        # source_im:  B x (R, G, B) x 256 x 256 float [0, 1]
        # source_iuv: B x (I, U, V) x 256 x 256
        # target_iuv: B x (I, U, V) x 256 x 256

        # Generate predicted image
        # If use_gt, we use the source image as the predictive result.  This is for training the Disc Network
        if use_gt:
            _ , source_part_mask = utils.get_body_and_part_mask_from_iuv(source_iuv)
            _ , target_part_mask = utils.get_body_and_part_mask_from_iuv(target_iuv)
            predictive_result = source_im
        else:
            predictive_result, source_part_mask, target_part_mask = self.generator(source_im, source_iuv, target_iuv)

        # Classify predicted image as either from distribution or not
        classification = self.discriminator(predictive_result, source_part_mask.float(), target_part_mask.float())

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
    def __init__(self, pretrained_person_inpainter=None, pretrained_background_inpainter=None, pretrained_predictive_module=None, debug=False):
        super().__init__()
        self._debug = debug
        # initialize predictive module
        self.predictive_module = PredictiveModel()

        if pretrained_predictive_module is not None:
            print('LOADING PRETRAINED PREDICTIVE MODULE')
            self.predictive_module.load_state_dict(torch.load(pretrained_predictive_module))

        # initialize warping modules
        self.warping_module = InpaintingAutoencoder()

        if pretrained_person_inpainter is not None:
            print('LOADING PRETRAINED PERSON INPAINTER')
            self.warping_module.load_state_dict(torch.load(pretrained_person_inpainter))

        # initialize background inpainting module
        self.bg_inpainting = UNet()
        if pretrained_background_inpainter is not None:
            print('LOADING PRETRAINED BACKGROUND INPAINTER')        
            state_dict = torch.load(pretrained_background_inpainter)
            state_dict_rename = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                state_dict_rename[name] = v

            self.bg_inpainting.load_state_dict(state_dict_rename)

        # initialize blending module
        self.blending_module = Blending()

    def forward(self, source_im, source_iuv, target_iuv):
        # source_im:  B x (R, G, B) x 256 x 256 float [0, 1]
        # source_iuv: B x (I, U, V) x 256 x 256
        # target_iuv: B x (I, U, V) x 256 x 256

        source_body_mask, source_part_mask = utils.get_body_and_part_mask_from_iuv(source_iuv)
        target_body_mask, target_part_mask = utils.get_body_and_part_mask_from_iuv(target_iuv)

        # predictive
        if self._debug:
            assert len(source_im[source_im > 1.0]) == 0, "Source image not between 0 and 1"
        pred_transform = transforms.Compose([NORM])
        predictive_source_im = pred_transform(source_im)
        predictive_result = self.predictive_module(predictive_source_im * source_body_mask.float(), source_iuv, target_iuv)
        if self._debug:
            assert len(source_im[source_im < 0.0]) == 0, "Source image for other networks is mean/std normalized"

        # warping
        source_texture_map = utils.texture_from_images_and_iuv(source_im, source_iuv)
        # utils.plot_texture_map(source_texture_map[0])
        inpainted_source_texture_map = self.warping_module(source_texture_map)
        # utils.plot_texture_map(inpainted_source_texture_map[0])
        warping_result = utils.images_from_texture_and_iuv_batch(inpainted_source_texture_map, target_iuv)

        # background inpainting
        # KEVIN: INPUT MASKS SHOULD BE SCALED TO 255 OR 1?
        # Answer: Masks should be float32's from 0 to 1
        inpainted_bg = self.bg_inpainting(source_im, source_body_mask, source_part_mask)

        # blending
        blending_result = self.blending_module(predictive_result, warping_result, target_iuv)

        # final result
        final_result = utils.combine_foreground_background(blending_result, target_body_mask, inpainted_bg)

        return final_result, source_part_mask, target_part_mask
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
    def __init__(self, in_c=6, plus_c=24, input_shape=256, nf_enc=[64]*2+[128]*9, nf_dec=[128]*4 + [64]):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=nf_enc[0], kernel_size=7, padding=3, stride=1),
            torch.nn.LeakyReLU(0.2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=plus_c+nf_enc[0], out_channels=nf_enc[1], kernel_size=3, padding=1,stride=2),
            torch.nn.LeakyReLU(0.2)               
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[1], out_channels=nf_enc[2], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU(0.2)              
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[2], out_channels=nf_enc[3], kernel_size=3, padding=1,stride=2),
            torch.nn.LeakyReLU(0.2)              
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[3], out_channels=nf_enc[4], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU(0.2)              
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[4], out_channels=nf_enc[5], kernel_size=3, padding=1,stride=2),
            torch.nn.LeakyReLU(0.2)              
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[5], out_channels=nf_enc[6], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU(0.2)              
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[6], out_channels=nf_enc[7], kernel_size=3, padding=1,stride=2),
            torch.nn.LeakyReLU(0.2)              
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[7], out_channels=nf_enc[8], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU(0.2)              
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[8], out_channels=nf_enc[9], kernel_size=3, padding=1,stride=2),
            torch.nn.LeakyReLU(0.2)              
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[9], out_channels=nf_enc[10], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU(0.2)              
        )

        self.deconv0 = nn.Sequential(
            nn.Conv2d(in_channels=nf_enc[10]+nf_enc[8], out_channels=nf_dec[0], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU(0.2)              
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_channels=nf_dec[0]+nf_enc[6], out_channels=nf_dec[1], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU(0.2)              
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(in_channels=nf_dec[1]+nf_enc[4], out_channels=nf_dec[2], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU(0.2)              
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(in_channels=nf_dec[2]+nf_enc[2], out_channels=nf_dec[3], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU(0.2)              
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(in_channels=nf_dec[3]+nf_enc[0], out_channels=nf_dec[4], kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU(0.2)              
        )

        self.finalConv = nn.Sequential(
            nn.Conv2d(in_channels=nf_dec[4], out_channels=3, kernel_size=3, padding=1,stride=1),
            torch.nn.LeakyReLU(0.2),
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

        # Input is Body mask, invert it
        mask = mask == 0
        mask = mask.float()
        x_in = image*mask #256
        x_in = self.conv0(torch.cat((x_in, mask.repeat(1,3,1,1)),dim=1))
        x0 = torch.cat((x_in, pose.float()),dim=1) #256
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
        x = torch.cat((x, x_in),dim=1)
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
    #z = 255.0 * (x_in + 1.0) / 2.0
    z = 255.0 * x_in
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
        # If image is 0 to 1, we are fine
        return self.features(vgg_preprocess(img))

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss,self).__init__()
        
    def forward(self,y_pred, y_true, lmbda=1.0):
        return vgg_loss(y_pred, y_true, lmbda=lmbda)


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

''' #################

BEGIN PREDICTIVE MODULE

''' #################

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

        self.enc_block1 = EncodeBlock(channels, num_features) # output: B x 64 x 256 x 256
        self.enc_block2 = EncodeBlock(num_features, num_features*2) # output: B x 128 x 128 x 128
        self.enc_block3 = EncodeBlock(num_features*2, num_features*4) # output: B x 256 x 64 x 64
 
        self.bottle = BottleNeck() # output: B x 256 x 64 x 64

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(256))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(256))
 
        self.output = nn.Sequential(
            nn.Conv2d(256, 3, 3, padding=1), 
            nn.Tanh(),
        ) # output: B x 3 x 256 x 256

    def forward(self, source_im, source_iuv, target_iuv):
        pred_input = torch.cat((source_im, source_iuv, target_iuv), 1)
        enc1 = self.enc_block1(pred_input) # B x 64 x 256 x 256
        x = F.max_pool2d(enc1, 2, stride=2) # B x 64 x 128 x 128
        enc2 = self.enc_block2(x)           # B x 128 x 128 x 128
        x = F.max_pool2d(enc2, 2, stride=2) # B x 128 x 64 x 64
        enc3 = self.enc_block3(x)           # B x 256 x 64 x 64
        bottle = self.bottle(enc3)             # B x 256 x 64 x 64
        dec1 = self.deconv1(bottle)# B x 256 x 128 x 128
        dec2 = self.deconv2(dec1)  # B x 256 x 256 x 256
        output = self.output(dec2)
        return output
''' ##########################

END PREDICTIVE MODULE

''' ##########################
 
''' ##########################

BEGIN BLENDING MODULE

''' ##########################
 
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
        # self.res3 = ResidualBlock(num_features, num_features)
        # self.classify = nn.Linear(num_features, 3)
        self.res3 = ResidualBlock(num_features, 3)
        
    def forward(self, pred_output, warp_output, target_pose):
       blend_input = torch.cat((pred_output, warp_output, target_pose), 1)
       x1 = self.conv1(blend_input)
       x2 = self.conv2(x1)
       x3 = self.res1(x2)
       x4 = self.res2(x3)
       x5 = self.res3(x4) 
       # to_linear = x5.view(x5.size()[0],x5.size()[2],x5.size()[3],x5.size()[1])
       # to_linear = x5.view(0,2,3,1)
       # o = self.classify(to_linear)
       # return o.view(o.size()[0],o.size()[3],o.size()[1],o.size()[2])
       # return o.permute(0,3,1,2)
       return x5

''' ################################

END BLENDING MODULE

''' ################################ 

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


