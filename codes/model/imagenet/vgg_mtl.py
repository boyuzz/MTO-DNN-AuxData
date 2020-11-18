# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
# @Time    : 25/12/2019 3:32 pm
# @Author  : Boyu Zhang
# @Contact : boyuzhang@swin.edu.au
# @File    : vgg_mtl.py
# @Software: PyCharm
"""

from torchvision.models.vgg import VGG, load_state_dict_from_url, model_urls, make_layers, cfgs
from torch import nn
from model.networks import ListModule


class MTLVGG(VGG):
	def __init__(self, branches, features, num_classes=1000, init_weights=True):
		super(MTLVGG, self).__init__(features, num_classes, init_weights)

		self.classifier = None
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			# nn.Linear(4096, num_classes),
		)
		self.fcs = ListModule(self, 'fc_')
		for num_classes in branches:
			self.fcs.append(nn.Linear(4096, num_classes))

		if init_weights:
			self._initialize_weights()

	def forward(self, x, branch=0):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		if not self.training:
			branch = 0
		x = self.fcs[branch](x)

		return x


def _vgg(branches, arch, cfg, batch_norm, pretrained, progress, **kwargs):
	if pretrained:
		kwargs['init_weights'] = False
	model = MTLVGG(branches,make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
	if pretrained:
		state_dict = load_state_dict_from_url(model_urls[arch],
											  progress=progress)
		own_state = model.state_dict()
		pretrained_dict = {k: v for k, v in state_dict.items() if k in own_state}
		# 2. overwrite entries in the existing state dict
		own_state.update(pretrained_dict)
		# 3. load the new state dict
		model.load_state_dict(own_state)

	return model


def vgg16_bn_mtl(branches, pretrained=False, progress=True, **kwargs):
	"""VGG 16-layer model (configuration "D") with batch normalization

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _vgg(branches, 'vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19_bn_mtl(branches, pretrained=False, progress=True, **kwargs):
	"""VGG 19-layer model (configuration 'E') with batch normalization

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _vgg(branches, 'vgg19_bn', 'E', True, pretrained, progress, **kwargs)
