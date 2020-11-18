from torch import nn


class DCG(nn.Module):
	def __init__(self, channels, input_size):
		super(DCG, self).__init__()

		self.init_size = 4
		# self.l1 = nn.Sequential(nn.Linear(input_size, 256 * self.init_size ** 2))

		def transConv2d_block(in_filters, out_filters, activation='relu', kernel=4, stride=2, padding=1):
			if activation == 'relu':
				afun = nn.ReLU(inplace=True)
			elif activation == 'sigmoid':
				afun = nn.Sigmoid()
			elif activation == 'tanh':
				afun = nn.Tanh()
			else:
				raise NotImplementedError('activation function {} is not implemented!'.format(activation))

			block = [nn.ConvTranspose2d(in_filters, out_filters, kernel, stride, padding, bias=False),
			         nn.BatchNorm2d(out_filters),
			         afun]
			return block

		self.model = nn.Sequential(
			# *transConv2d_block(256, 256),
			*transConv2d_block(input_size, 512, stride=1, padding=0),
			*transConv2d_block(512, 256),
			*transConv2d_block(256, 128),
			*transConv2d_block(128, 64),
			*transConv2d_block(64, channels, 'tanh')
		)

	def forward(self, z):
		# out = self.l1(z)
		# out = out.view(out.shape[0], 256, self.init_size, self.init_size)
		img = self.model(z)
		return img


class DCD(nn.Module):
	def __init__(self, channels):
		super(DCD, self).__init__()

		def discriminator_block(in_filters, out_filters, bn=True):
			block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
			if bn:
				block.append(nn.BatchNorm2d(out_filters))
			return block

		self.model = nn.Sequential(
			*discriminator_block(channels, 16, bn=False),
			*discriminator_block(16, 32),
			*discriminator_block(32, 64),
			*discriminator_block(64, 128),
			# *discriminator_block(128, 256),
			# *discriminator_block(256, 512)
		)

		# The height and width of downsampled image
		ds_size = 4
		self.adv_layer = nn.Sequential(nn.Conv2d(128, 1, 4, 1, 0, bias=False), nn.Sigmoid())

	def forward(self, img):
		out = self.model(img)
		# out = out.view(out.shape[0], -1)
		validity = self.adv_layer(out)

		return validity
