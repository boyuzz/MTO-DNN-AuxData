# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
# @Time    : 25/12/2019 3:19 pm
# @Author  : Boyu Zhang
# @Contact : boyuzhang@swin.edu.au
# @File    : alexnet_mtl.py
# @Software: PyCharm
"""

from torchvision.models.alexnet import AlexNet, load_state_dict_from_url, model_urls
from torch import nn
from model.networks import ListModule


class MTLAlexNet(AlexNet):
	def __init__(self, branches):
		super(MTLAlexNet, self).__init__()

		self.classifier = None
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			# nn.Linear(4096, num_classes),
		)
		self.fcs = ListModule(self, 'fc_')
		for num_classes in branches:
			self.fcs.append(nn.Linear(4096, num_classes))

	def forward(self, x, branch=0):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.classifier(x)
		if not self.training:
			branch = 0
		x = self.fcs[branch](x)

		return x


def alexnet_mtl(branches, pretrained=False, progress=True, **kwargs):
	r"""AlexNet model architecture from the
	`"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	model = MTLAlexNet(branches, **kwargs)
	if pretrained:
		state_dict = load_state_dict_from_url(model_urls['alexnet'],
											  progress=progress)
		own_state = model.state_dict()
		pretrained_dict = {k: v for k, v in state_dict.items() if k in own_state}
		# 2. overwrite entries in the existing state dict
		own_state.update(pretrained_dict)
		# 3. load the new state dict
		model.load_state_dict(own_state)

	return model