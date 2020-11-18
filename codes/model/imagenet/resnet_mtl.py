from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock, model_urls, load_state_dict_from_url
from torch import nn
from model.networks import ListModule


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MTLResNet(ResNet):

	def __init__(self, block, layers, branches, ):
		super(MTLResNet, self).__init__(block, layers)
		self.fc = None
		self.fcs = [nn.Linear(512 * block.expansion, num_classes) for num_classes in branches]
		self.fcs = ListModule(self, 'fc_')
		for num_classes in branches:
			self.fcs.append(nn.Linear(512 * block.expansion, num_classes))

	def forward(self, x, branch=0):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		if not self.training:
			branch = 0
		x = self.fcs[branch](x)

		return x


def _resnet(arch, block, layers, branches, pretrained, progress, **kwargs):
	model = MTLResNet(block, layers, branches, **kwargs)
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


def resnet18_mtl(branches, pretrained=False, progress=True, **kwargs):
	"""Constructs a ResNet-18 model.
	"""
	return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], branches, pretrained, progress,
				   **kwargs)


def resnet34_mtl(branches, pretrained=False, progress=True, **kwargs):
	"""Constructs a ResNet-34 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], branches, pretrained, progress,
				   **kwargs)


def resnet50_mtl(branches, pretrained=False, progress=True, **kwargs):
	"""Constructs a ResNet-50 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], branches, pretrained, progress,
				   **kwargs)


def resnet101_mtl(branches, pretrained=False, progress=True, **kwargs):
	"""Constructs a ResNet-101 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], branches, pretrained, progress,
				   **kwargs)


def resnet152_mtl(branches, pretrained=False, progress=True, **kwargs):
	"""Constructs a ResNet-152 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], branches, pretrained, progress,
				   **kwargs)


def resnext50_32x4d_mtl(branches, pretrained=False, progress=True, **kwargs):
	"""Constructs a ResNeXt-50 32x4d model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	kwargs['groups'] = 32
	kwargs['width_per_group'] = 4
	return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
				   branches, pretrained, progress, **kwargs)


def resnext101_32x8d_mtl(branches, pretrained=False, progress=True, **kwargs):
	"""Constructs a ResNeXt-101 32x8d model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	kwargs['groups'] = 32
	kwargs['width_per_group'] = 8
	return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
				   branches, pretrained, progress, **kwargs)