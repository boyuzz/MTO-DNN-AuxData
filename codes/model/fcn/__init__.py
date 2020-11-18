import torchvision.models as models

from model.fcn.fcn import fcn8s, fcn16s, fcn32s
from model.fcn.segnet import segnet
from model.fcn.unet import unet
from model.fcn.pspnet import pspnet
from model.fcn.icnet import icnet
from model.fcn.linknet import linknet
from model.fcn.frrn import frrn
from model.fcn.unetplus import unetplus
from model.fcn.resunet import UNetWithResnet50Encoder
from model.fcn.ternausnet import unet11, AlbuNet, UNet16


def get_model(name, n_classes, version=None):
	# name = model_dict["arch"]
	model = _get_model_instance(name)
	# param_dict = copy.deepcopy(model_dict)
	# param_dict.pop("arch")

	if name in ["frrnA", "frrnB"]:
		model = model(n_classes)    # , **param_dict

	elif name in ["fcn32s", "fcn16s", "fcn8s"]:
		model = model(n_classes=n_classes)
		vgg16 = models.vgg16(pretrained=True)
		model.init_vgg16_params(vgg16)

	elif name == "segnet":
		model = model(n_classes=n_classes)
		vgg16 = models.vgg16(pretrained=True)
		model.init_vgg16_params(vgg16)

	elif name == "unet":
		model = model(n_classes=n_classes)

	elif name == "unetplus":
		model = model(n_classes=n_classes)

	elif name == "resunet":
		model = model(n_classes=n_classes)

	elif name == "pspnet":
		model = model(n_classes=n_classes)

	elif name == "icnet":
		model = model(n_classes=n_classes)

	elif name == "icnetBN":
		model = model(n_classes=n_classes)
	elif name == 'unet11':
		model = model()
	elif name in ['AlbuNet', 'UNet16']:
		model = model(n_classes=n_classes)

	else:
		raise NotImplementedError('{} is not implemented!'.format(name))

	return model


def _get_model_instance(name):
	try:
		return {
			"fcn32s": fcn32s,
			"fcn8s": fcn8s,
			"fcn16s": fcn16s,
			"unet": unet,
			"unetplus": unetplus,
			"resunet": UNetWithResnet50Encoder,
			"segnet": segnet,
			"pspnet": pspnet,
			"icnet": icnet,
			"icnetBN": icnet,
			"linknet": linknet,
			"frrnA": frrn,
			"frrnB": frrn,
			"unet11": unet11,
			"AlbuNet": AlbuNet,
			"UNet16": UNet16
		}[name]
	except:
		raise ("Model {} not available".format(name))
