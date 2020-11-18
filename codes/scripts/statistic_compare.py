import scipy.stats as stats
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from re import findall

SMALL_SIZE = 9
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)


def get_metric(path, metric='loss'):
	loss = -1
	loss_list = []
	if metric == 'loss':
		split = 'Lowest loss:'
	else:
		split = 'Accuracy:'
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if 'Lowest loss:' in line and 'Task 0-rank 0' in line:
				# task_id = int(line.split('Task ')[-1].split(',')[0])
				# loss = float(line.split(split)[-1].split(',')[0])
				loss = float(findall(r"\d+\.?\d*e?[-+]?\d+", line.split(split)[1])[0])
				# f1_score = float(line.split('F1 score: ')[-1])
			if 'overall accuracy:' in line:
				loss_list.append(loss)
	try:
		return loss_list
	except UnboundLocalError:
		print('error file {}'.format(path))


# def get_training_loss


def get_checkpoint(path):
	oa_list = []
	nt_list = []
	count = 0
	c_trans = 0
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if 'Task 0' in line and '# Iter:' in line:
				rank_num = int(line.split('rank')[-1].split(',')[0])
				if rank_num % 2 == 0:
					# task_id = int(line.split('Task ')[-1].split(',')[0])
					loss = float(findall(r"\d+\.?\d*e?[-+]?\d+", line.split('Loss:')[1])[0])
					n_trans = float(findall(r"\d+\.?\d*", line.split('Count:')[1])[0])
					# accuracy = float(re.findall(r"\d+\.?\d*", line.split('Loss:')[1])[0])
					oa_list.append(loss)
					if n_trans > c_trans:
						c_trans = n_trans
						nt_list.append((count, loss))
					count += 1
	return oa_list, nt_list


def get_overall_accuracy(path, find_best=False):
	oa = 0
	n_trans = 0
	percentage = 0
	bestlist = []
	# try:
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if '# Task' in line:
				pbest = float(findall(r"\d+\.?\d*", line.split('Accuracy:')[1])[0])
				bestlist.append(pbest)

			if '# Task 0 #' in line:
			# if 'Loss:' in line and 'Task 0' in line:
				oa = float(findall(r"\d+\.?\d*", line.split('Accuracy:')[1])[0])
				# oa = float(re.findall(r"\d+\.?\d*e?[-+]?\d+", line.split('Loss:')[1])[0])
				# oa = float(line.split('accuracy:')[1][:7])
				# break
			if 'Task 0-rank 0' in line:
				n_trans = float(findall(r"\d+\.?\d*", line.split('Transfer Count:')[1])[0])
				# iters = float(re.findall(r"\d+\.?\d*", line.split('Iter:')[1])[0])
				# percentage = n_trans*100/iters
	if find_best:
		try:
			oa = max(bestlist)
		except ValueError:
			print('error file is {}'.format(path))
			oa = 0
	# except:
	# 	print('exception catched', line)
	# 	oa = 0
	return oa, n_trans


def multi_oa(path):
	bestlist = []
	# try:
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if 'overall accuracy' in line:
				pbest = float(findall(r"\d+\.?\d*", line.split('overall accuracy:')[1])[0])
				bestlist.append(pbest)

	return bestlist

def compare_sto():
	# models = ['alexnet', 'vgg16_bn', 'resnet18', 'resnet50', 'densenet121']
	models = ['squeezenet1_1', 'mobilenet_v2', 'densenet121']
	# models = ['vgg16_bn', 'densenet121']
	# datasets = ['UCMerced', 'WHU19', 'RSSCN7','AID']
	datasets = ['RSSCN7', 'OxfordPets', 'UCMerced']
	draw_figure = False

	for d in datasets:
		if draw_figure:
			fig = plt.figure(figsize=(30, 20))
		print('datasets {}'.format(d))
		for j, m in enumerate(models):
			print('models {}'.format(m))
			avg_acc_single = []
			avg_loss_single = []
			avg_trans = []
			avg_acc_mto = []
			avg_loss_mto = []

			for i in range(1):
				# f_single = '../../cv_mto/rval/{}_single_{}_rval_ozstar_n1_seed{}.txt'.format(d, m, i)
				# f_single = '../../cv_mto/rval5/{}_single_{}_ozstar_n1.txt'.format(d, m)
				f_single = '../../cv_mto/rval5/{}_w_rancl_{}_rval_ozstar_n4.txt'.format(d, m)
				# f_mto = '../../results/1007/{}_w_VS_2_rancl_100000_{}_n4_seed{}.txt'.format(d, m, i)
				# f_mto = '../../cv_mto/rval/{}_w_{}_rval_ozstar_n4_seed{}.txt'.format(d, m, i)
				f_mto = '../../cv_mto/rval5/{}_w_{}_rval_ozstar_n4.txt'.format(d, m)

				if not draw_figure:
					# oa, ntrans = get_overall_accuracy(f_single, find_best=True)
					oa = multi_oa(f_single)
					loss = get_metric(f_single)
					avg_acc_single.extend(oa)
					avg_loss_single.extend(loss)

					# oa, ntrans = get_overall_accuracy(f_mto)
					oa = multi_oa(f_mto)
					loss = get_metric(f_mto)
					avg_acc_mto.extend(oa)
					avg_loss_mto.extend(loss)
					# avg_trans.append(ntrans)
				else:
					ax1 = fig.add_subplot(len(models), 5, j*5+i+1)
					oa_list_sto, _ = get_checkpoint(f_single)
					min_loss_sto = min(oa_list_sto)
					min_idx_sto = np.argmin(oa_list_sto)
					avg_acc_single.append(oa_list_sto[-1])

					oa_list_mto, nt_list = get_checkpoint(f_mto)
					avg_trans.append(nt_list[-1])
					min_loss_mto = min(oa_list_mto)
					min_idx_mto = np.argmin(oa_list_mto)
					avg_acc_mto.append(oa_list_mto[-1])

					ax1.plot(oa_list_sto)
					ax1.scatter(min_idx_sto, min_loss_sto,
					            color='m', marker='o', s=30)
					# ax1.hlines(min_loss_sto, 0, max(len(oa_list_sto), len(oa_list_mto)), linestyles='dashed')
					ax1.plot(oa_list_mto)
					ax1.scatter(min_idx_mto, min_loss_mto,
					            color='m', marker='o', s=30)
					# ax1.hlines(min_loss_mto, 0, max(len(oa_list_sto), len(oa_list_mto)), linestyles='dashed')
					ax1.scatter(list(zip(*nt_list))[0], list(zip(*nt_list))[1],
					                   color='', marker='o', edgecolors='g', s=30)
					ax1.legend(['sto', 'mto'])
					ax1.set_ylabel('Val loss')
					ax1.set_xlabel('steps (*100)')
					ax1.yaxis.get_major_formatter().set_powerlimits((0, 1))

					# ax2 = ax1.twinx()  # this is the important function
					# ax2.plot(nt_list)
					# ax2.set_ylabel('mto n_trans')

			print(avg_acc_single)
			print(avg_loss_single)
			print(avg_acc_mto)
			print(avg_loss_mto)

			print('avg single {}'.format(sum(avg_acc_single)/len(avg_acc_single)))
			print('avg single {}'.format(sum(avg_loss_single)/len(avg_loss_single)))
			print('avg mto {}'.format(sum(avg_acc_mto)/len(avg_acc_mto)))
			print('avg mto {}'.format(sum(avg_loss_mto)/len(avg_loss_mto)))
			print('trans percentage {}'.format(avg_trans))
			# print('average trans percentage {}'.format(sum(avg_trans)/len(avg_trans)))
			print('-------------------------')
		if draw_figure:
			plt.tight_layout()
			fig.savefig('{}.pdf'.format(d))
		print('============================')


def compare_n():
	# plt.rcParams['font.sans-serif'] = ['Times']
	model = ['densenet121', 'mobilenet_v2', 'squeezenet1_1']
	# model = 'mobilenet_v2'
	# model = 'squeezenet1_1'
	# datasets = ['UCMerced', 'RSSCN7', 'WHU19', 'AID']
	datasets = ['UCMerced', 'OxfordPets', 'RSSCN7']
	# datasets = ['AID']
	# ntasks = [0, 50, 100, 200, 400]
	ntasks = [1, 2, 4, 6]
	fig = plt.figure(figsize=(12,9))
	# fig, axes = plt.subplots(len(model), len(datasets), sharex='col', sharey=True, figsize=(10, 9))
	for n, d in enumerate(datasets):
		plt.figure()
		# avg_loss = np.zeros((5,len(ntasks)))
		# avg_acc = np.zeros((5,len(ntasks)))
		avg_loss = np.zeros(len(ntasks))
		avg_acc = np.zeros(len(ntasks))

		for k, m in enumerate(model):
			files = []
			# 1007: 不同交互频率
			# 1003_n: STO和MTO的结果
			# files.append('../../results/1003_n/{}_single_{}_n1_seed{}.txt'.format(d, model, i))
			# files.append('../../results/1007/{}_w_VS_2_rancl_{}_n2_seed{}.txt'.format(d, model, i))
			# files.append('../../results/1007/{}_w_VS_2_rancl_{}_n3_seed{}.txt'.format(d, model, i))
			# files.append('../../results/1003_n/{}_w_VS_2_rancl_{}_n4_seed{}.txt'.format(d, model, i))
			# files.append('../../results/1007/{}_w_VS_2_rancl_{}_n5_seed{}.txt'.format(d, model, i))
			# files.append('../../results/1007/{}_w_VS_2_rancl_{}_n6_seed{}.txt'.format(d, model, i))
			# files.append('../../results/1007/{}_w_VS_2_rancl_{}_n7_seed{}.txt'.format(d, model, i))
			# files.append('../../results/1007/{}_w_VS_2_rancl_{}_n8_seed{}.txt'.format(d, model, i))
			files.append('../../cv_mto/rval5/{}_single_{}_rval_ozstar_n1.txt'.format(d, m))
			files.append('../../cv_mto/rval5/{}_w_{}_rval_ozstar_n2.txt'.format(d, m))
			files.append('../../cv_mto/rval5/{}_w_{}_rval_ozstar_n4.txt'.format(d, m))
			files.append('../../cv_mto/rval5/{}_w_{}_rval_ozstar_n6.txt'.format(d, m))

			for j, f in enumerate(files):
				loss = get_metric(f)
				# acc, _ = get_overall_accuracy(f)
				acc = multi_oa(f)
				avg_loss[j] = sum(loss)/len(loss)
				avg_acc[j] = sum(acc)/len(acc)

			# print(avg_loss)
			# print(avg_acc)
			# loss_trend = np.mean(avg_loss, axis=0)
			# acc_trend = np.mean(avg_acc, axis=0)
			loss_trend = avg_loss #[format(num, '.2f') for num in avg_loss]
			acc_trend = avg_acc #[format(num, '.2f') for num in avg_acc]

			ax = fig.add_subplot(3, 3, n*len(model)+k+1)
			# TODO: 使用科学记数法，放大字体
			lns1 = ax.plot(ntasks, loss_trend, marker="o", color='C1', label='Val loss', linestyle='dashed')
			# ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
			ax.set_ylabel('Val loss')
			ax.set_xlabel('Number of tasks')
			# ax.set_title(d if d !="WHU19" else "WHU19-RS")
			if m == 'mobilenet_v2':
				m = 'MobileNetV2'
			if m == 'squeezenet1_1':
				m = 'SqueezeNet'
			if m == 'densenet121':
				m = 'DenseNet-121'
			ax.set_title('{} {}'.format(d,m))
			# plt.legend([], loc='right')

			ax2 = ax.twinx()  # this is the important function
			lns2 = ax2.plot(ntasks, acc_trend, marker="v", color='C2', label='Accuracy')
			ax2.set_ylabel('Accuracy (%)')

			lns = lns1 + lns2
			labs = [l.get_label() for l in lns]
			ax.legend(lns, labs, loc=5)

	fig.tight_layout()
	fig.savefig('ntasks.pdf')

# compare_sto()
compare_n()

# plt.plot(get_checkpoint('../../results/RSSCN7_single_resnet18_seed0.txt'))
# plt.plot(get_checkpoint('../../results/RSSCN7_w_VS_rancl_resnet18_seed0.txt'))
#
# plt.show()
# kd = [95.50, 95.52, 95.94, 95.58, 95.51, 95.56, 95.76, 95.79, 95.53, 95.65]
# at = [95.44, 95.33, 95.94, 95.58, 94.94, 95.50, 95.35, 94.24, 95.39, 94.98]
# wst = [95.30, 95.19, 94.60, 95.47, 95.16, 95.44, 95.25, 94.50, 95.37, 95.16]
# single = get_metric('../imagenet32_single.txt')
# kd = get_metric('../imagenet32_k.txt')
# at = get_metric('../imagenet32_a.txt')
# wst = get_metric('../imagenet32_ws.txt')
# # atl = get_metric('../cifar10_at_last_vl.txt')
# print(single)
# print(kd)
# print(at)
# print(wst)

# print(stats.shapiro(single), stats.levene(single,at,wst))
# print(stats.shapiro(at), stats.levene(at))
# print(stats.shapiro(wst), stats.levene(wst))
# length = len(single)
# print(sum(single)/length, sum(kd)/length, sum(at)/length, sum(wst)/length)
#
# print(stats.mannwhitneyu(single, kd))
# print(stats.mannwhitneyu(single, at))
# print(stats.mannwhitneyu(single, wst))
