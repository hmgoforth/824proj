import torch
import torch.nn as nn

from pdb import set_trace as st

class InpaintingAutoencoder(nn.Module):

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




