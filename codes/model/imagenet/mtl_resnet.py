from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock, model_urls, load_state_dict_from_url
from torch import nn



def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MTLResNet(ResNet):

	def __init__(self, block, layers, mtl_num_classes, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
		super(MTLResNet, self).__init__(block, layers, num_classes, zero_init_residual, groups,
		                                width_per_group, replace_stride_with_dilation, norm_layer)

		self.fcs = [nn.Linear(512 * block.expansion, num_classes) for num_classes in mtl_num_classes]

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

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


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
	model = MTLResNet(block, layers, **kwargs)
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


def resnet18(**kwargs):
	"""Constructs a ResNet-18 model.
	"""
	model = MTLResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
	return model


def resnet34(**kwargs):
	"""Constructs a ResNet-34 model.
	"""
	model = MTLResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
	return model


def resnet50(**kwargs):
	"""Constructs a ResNet-50 model.
	"""
	model = MTLResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	return model


def resnet101(**kwargs):
	"""Constructs a ResNet-101 model.
	"""
	model = MTLResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
	return model


def resnet152(**kwargs):
	"""Constructs a ResNet-152 model.
	"""
	model = MTLResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
	return model