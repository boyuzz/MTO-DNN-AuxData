import functools
import logging
import torch
import torch.nn as nn
from torchvision import models
from torch.nn import init
import torch.nn.functional as F

logger = logging.getLogger('base')
####################
# initialize
####################

class ListModule(object):
	#Should work with all kind of module
	def __init__(self, module, prefix, *args):
		self.module = module
		self.prefix = prefix
		self.num_module = 0
		for new_module in args:
			self.append(new_module)

	def append(self, new_module):
		if not isinstance(new_module, nn.Module):
			raise ValueError('Not a Module')
		else:
			self.module.add_module(self.prefix + str(self.num_module), new_module)
			self.num_module += 1

	def __len__(self):
		return self.num_module

	def __getitem__(self, i):
		if i < 0 or i >= self.num_module:
			raise IndexError('Out of bound')
		return getattr(self.module, self.prefix + str(i))


def weights_init_normal(m, std=0.02):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, std)
		if m.bias is not None:
			m.bias.data.zero_()
	elif classname.find('Linear') != -1:
		init.normal_(m.weight.data, 0.0, std)
		if m.bias is not None:
			m.bias.data.zero_()
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
		init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
		m.weight.data *= scale
		if m.bias is not None:
			m.bias.data.zero_()
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
		m.weight.data *= scale
		if m.bias is not None:
			m.bias.data.zero_()
	elif classname.find('BatchNorm2d') != -1:
		init.constant_(m.weight.data, 1.0)
		init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.orthogonal_(m.weight.data, gain=1)
		if m.bias is not None:
			m.bias.data.zero_()
	elif classname.find('Linear') != -1:
		init.orthogonal_(m.weight.data, gain=1)
		if m.bias is not None:
			m.bias.data.zero_()
	elif classname.find('BatchNorm2d') != -1:
		init.constant_(m.weight.data, 1.0)
		init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
	def normal_init(m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			nn.init.normal_(m.weight.data, 0.0, std)
		elif classname.find('BatchNorm') != -1:
			nn.init.normal_(m.weight.data, 1.0, std)
			nn.init.constant_(m.bias.data, 0)

		# scale for 'kaiming', std for 'normal'.
	logger.info('Initialization method [{:s}]'.format(init_type))
	if init_type == 'normal':
		# weights_init_normal_ = functools.partial(weights_init_normal, std=std)
		net.apply(normal_init)
	elif init_type == 'kaiming':
		weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
		net.apply(weights_init_kaiming_)
	elif init_type == 'orthogonal':
		net.apply(weights_init_orthogonal)
	else:
		raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################

#======Networks for mnist================
class MnistResNet18(models.ResNet):
	def __init__(self, num_classes):
		super(MnistResNet18, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
		self.conv1 = torch.nn.Conv2d(1, 64,
		                             kernel_size=7,
		                             stride=2,
		                             padding=3, bias=False)
		self.avgpool = nn.AvgPool2d(1, stride=1)

	def forward(self, x):
		return super(MnistResNet18, self).forward(x)


class LeNet(nn.Module):
	def __init__(self, num_classes):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, num_classes)
		# for m in self.modules():
		#     weights_init_kaiming(m)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return x

