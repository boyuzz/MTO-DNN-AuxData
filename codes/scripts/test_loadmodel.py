import model.imagenet as imagenet
from torch import optim
from torch import nn
from dataloader import create_dataloader, create_dataset

from torch.optim import lr_scheduler


def weights_replace(new_dict, self_dict):
	# new_dict = {k: v for k, v in new_dict.items() if v.shape == self_dict[k].shape}
	new_dict = {k: v for k, v in new_dict.items() if 'fc' not in k}
	self_dict.update(new_dict)
	return self_dict


network_pre = getattr(imagenet, 'resnet18')(pretrained=True)
network = getattr(imagenet, 'resnet18')(num_classes=7)
network_pre = network_pre.cuda()
network = network.cuda()

pretrained_dict = network_pre.state_dict()

model_dict = network.state_dict()

model_dict = weights_replace(pretrained_dict, model_dict)
network.load_state_dict(model_dict)


init_lr = 0.01
optimizer = optim.SGD([{'params': [param for name, param in network.named_parameters() if 'fc' not in name]}, {'params': network.fc.parameters(), 'lr': 0.001}], lr=3e-4, momentum=0.9)

loss_func = nn.CrossEntropyLoss().cuda()

opt = {
	'name': 'RSSCN7',
	'lmdb': True,
	'resample': False,
	'dataroot': '../../data/RSSCN7/',
	'mode': 'file',
	'batch_size': 64,
	"use_shuffle": True,
	"n_workers": 0,
	"num_classes": 7
}
train_set = create_dataset(opt, train=True)
train_loader = create_dataloader(train_set, opt)
# pre_optimizer = optim.SGD([{'params': [param for name, param in network.named_parameters() if 'fc' not in name]}, {'params': network.fc.parameters(), 'lr': 1}], lr=1, momentum=0.9)
#
# new_dict = pre_optimizer.state_dict()
# self_dict = optimizer.state_dict()
# self_dict['param_groups'][0].update(new_dict['param_groups'][0])
#
# # self_dict.update(new_dict)
# optimizer.load_state_dict(self_dict)
# optimizer_dict = weights_replace(optimizer.state_dict(), pre_optimizer.state_dict())

# schedulers = []
# schedulers.append(lr_scheduler.MultiStepLR(optimizer, [30, 80], 0.1))
multistep = [30, 80]
lambda1 = lambda step: 0.1**sum([step >= mst for mst in multistep])
arxiv_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda1])
val_freq = 200

for (data, target) in train_loader:
	arxiv_scheduler.step()

	data, target = data.cuda(), target.cuda()
	
	optimizer.zero_grad()
	logits = network(data)

	loss = 0

	# logger.info(logits[0])
	loss += loss_func(logits, target)

	loss.backward()

	for name, param in network.named_parameters():
		if param.grad is not None:
			a = param.grad.clone().cpu().data.numpy()
			print('grad', name, a.max(), a.min())

	optimizer.step()

	model_dict = network.state_dict()
	model_dict = weights_replace(pretrained_dict, model_dict)
	network.load_state_dict(model_dict)

	# if step % val_freq == 0:
	# 	classifier_group = optimizer.param_groups[-1]
	# 	current_lr = classifier_group['lr']
	# 	# classifier_group['lr'] = init_lr
	# 	arxiv_scheduler.base_lrs[-1] = init_lr
	#
	# 	exponential_param = pow(current_lr / init_lr, 1/val_freq)
	# 	# print('val:', current_lr, exponential_param)
	# 	lambda2 = lambda step: exponential_param ** (step % val_freq if step % val_freq != 0 else val_freq)
	# 	arxiv_scheduler.lr_lambdas[-1] = lambda2


