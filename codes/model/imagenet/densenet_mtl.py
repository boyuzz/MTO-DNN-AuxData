# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
# @Time    : 25/12/2019 3:59 pm
# @Author  : Boyu Zhang
# @Contact : boyuzhang@swin.edu.au
# @File    : densenet_mtl.py
# @Software: PyCharm
"""

from torchvision.models.densenet import DenseNet, load_state_dict_from_url, model_urls, _DenseBlock, _Transition
from torch import nn
from model.networks import ListModule
import torch.nn.functional as F
import re


class MTLDenseNet(DenseNet):
	def __init__(self, branches, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
		super(MTLDenseNet, self).__init__(growth_rate, block_config,
                 num_init_features, bn_size, drop_rate, num_classes)
		self.classifier = None

		self.fcs = ListModule(self, 'fc_')
		num_features = num_init_features
		for i, num_layers in enumerate(block_config):
			block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
								bn_size=bn_size, growth_rate=growth_rate,
								drop_rate=drop_rate)
			self.features.add_module('denseblock%d' % (i + 1), block)
			num_features = num_features + num_layers * growth_rate
			if i != len(block_config) - 1:
				trans = _Transition(num_input_features=num_features,
									num_output_features=num_features // 2)
				self.features.add_module('transition%d' % (i + 1), trans)
				num_features = num_features // 2

		for num_classes in branches:
			self.fcs.append(nn.Linear(num_features, num_classes))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.constant_(m.bias, 0)

	def forward(self, x, branch=0):
		features = self.features(x)
		out = F.relu(features, inplace=True)
		out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
		if not self.training:
			branch = 0
		out = self.fcs[branch](out)
		return out


def _load_state_dict(model, model_url, progress):
	# '.'s are no longer allowed in module names, but previous _DenseLayer
	# has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
	# They are also in the checkpoints in model_urls. This pattern is used
	# to find such keys.
	pattern = re.compile(
		r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

	state_dict = load_state_dict_from_url(model_url, progress=progress)
	for key in list(state_dict.keys()):
		res = pattern.match(key)
		if res:
			new_key = res.group(1) + res.group(2)
			state_dict[new_key] = state_dict[key]
			del state_dict[key]
	model.load_state_dict(state_dict)


def _densenet(branches, arch, growth_rate, block_config, num_init_features, pretrained, progress,
			  **kwargs):
	model_mtl = MTLDenseNet(branches, growth_rate, block_config, num_init_features, **kwargs)
	model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)

	if pretrained:
		_load_state_dict(model, model_urls[arch], progress)
		state_dict = model.state_dict()

		own_state = model_mtl.state_dict()
		pretrained_dict = {k: v for k, v in state_dict.items() if k in own_state}
		# 2. overwrite entries in the existing state dict
		own_state.update(pretrained_dict)
		# 3. load the new state dict
		model_mtl.load_state_dict(own_state)
	return model_mtl


def densenet121_mtl(branches, pretrained=False, progress=True, **kwargs):
	r"""Densenet-121 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _densenet(branches, 'densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
					 **kwargs)