import csv
import numpy as np
import os
import shutil
from utils.util import is_image_file
# from utils.util import is_image_file
# be cautious about imbalance datasets

def get_im_label(root, folder_path):
	label_dict = []
	for c, folder in enumerate(folder_path):
		# file = os.path.join(self.root, self.base_folder, f)
		path = os.path.join(root, folder)
		# if not os.path.isdir(path):
		# 	continue
		data_images = os.listdir(path)
		relative_path = path.split('/')[-1]
		data_images = [os.path.join(relative_path, p) for p in data_images if is_image_file(p)]
		label_id = c
		labels = [label_id] * len(data_images)
		dicts = [{'image': im, 'label': l} for im, l in zip(data_images, labels)]
		label_dict.extend(dicts)

	return label_dict


def ISIC_split():
	nfold = 5
	classes = 7
	folds = [[] for i in range(nfold)]
	root = '../../data/RSSCN7/'
	root_train = '../../data/RSSCN7/train'
	root_test = '../../data/RSSCN7/test'

	train_folders = os.listdir(root_train)
	test_folders = os.listdir(root_test)
	train_folders = sorted(train_folders)
	test_folders = sorted(test_folders)

	# if self.train != 'test':
	label_dict = get_im_label(root_train, train_folders)
	test_label_dict = get_im_label(root_test, test_folders)

	# original_csv = '../../data/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
	# with open(original_csv, 'r') as csvfile:
	# 	spamreader = csv.reader(csvfile, delimiter=',')
	# 	for idx, row in enumerate(spamreader):
	# 		if idx == 0:
	# 			continue
	# 		else:
	# 			labels = np.array(list(map(int, row[1:])))
	# 			label_dict.append({'image': row[0], 'label': int(np.flatnonzero(labels))})

	for c in range(classes):
		category = [item for item in label_dict if item["label"] == c]
		randperm = np.random.permutation(range(len(category)))
		for f in range(nfold):
			start = int(f*len(randperm)//nfold)
			end = int((f+1)*len(randperm)//nfold)
			data_idx = randperm[start:end]
			data_this_fold = list(map(lambda i: category[i], data_idx))
			folds[f].extend(data_this_fold)

	for f in range(nfold):
		val_set = folds[f]
		randperm = np.random.permutation(range(len(val_set)))
		val_set = list(map(lambda i: val_set[i], randperm))

		train_set = []
		for i in range(nfold):
			if i != f:
				train_set += folds[i]
		randperm = np.random.permutation(range(len(train_set)))
		train_set = list(map(lambda i: train_set[i], randperm))

		with open(os.path.join(root, 'val_fold{}.csv'.format(f)), 'w', newline="") as csvfile:
			csvwriter = csv.writer(csvfile, delimiter=',')
			for line in val_set:
				csvwriter.writerow([line['image'], line['label']])

		with open(os.path.join(root, 'train_fold{}.csv'.format(f)), 'w', newline="") as csvfile:
			csvwriter = csv.writer(csvfile, delimiter=',')
			for line in train_set:
				csvwriter.writerow([line['image'], line['label']])

	with open(os.path.join(root, 'test.csv'), 'w', newline="") as csvfile:
		csvwriter = csv.writer(csvfile, delimiter=',')
		for line in test_label_dict:
			csvwriter.writerow([line['image'], line['label']])


def five_fold_split():
	nfold = 5
	folds = [[] for i in range(nfold)]
	root = '../../data/AID/'

	data_folders = os.listdir(root)
	data_folders = list(filter(lambda p: os.path.isdir(os.path.join(root, p)), data_folders))
	classes = len(data_folders)
	label_dict = get_im_label(root, data_folders)

	for c in range(classes):
		# category = [item for item in label_dict if item["label"] == c]
		category = list(filter(lambda p: p["label"] == c, label_dict))
		randperm = np.random.permutation(len(category))
		for f in range(nfold):
			start = int(f*len(randperm)//nfold)
			end = int((f+1)*len(randperm)//nfold)
			data_idx = randperm[start:end]
			data_this_fold = list(map(lambda i: category[i], data_idx))
			folds[f].extend(data_this_fold)

	for f in range(nfold):
		test_set = folds[f]

		# vfold = (f+1) % nfold
		# val_set = folds[vfold]

		train_set = []
		for i in range(nfold):
			if i != f :  # and i != vfold
				train_set += folds[i]
		randperm = np.random.permutation(range(len(train_set)))
		train_set = list(map(lambda i: train_set[i], randperm))

		# with open(os.path.join(root, 'val_fold{}.csv'.format(f)), 'w', newline="") as csvfile:
		# 	csvwriter = csv.writer(csvfile, delimiter=',')
		# 	for line in val_set:
		# 		try:
		# 			csvwriter.writerow([line['image'], line['label']])
		# 		except OSError:
		# 			print(line)

		with open(os.path.join(root, 'test_fold{}.csv'.format(f)), 'w', newline="") as csvfile:
			csvwriter = csv.writer(csvfile, delimiter=',')
			for line in test_set:
				try:
					csvwriter.writerow([line['image'], line['label']])
				except OSError:
					print(line)

		with open(os.path.join(root, 'train_fold{}.csv'.format(f)), 'w', newline="") as csvfile:
			csvwriter = csv.writer(csvfile, delimiter=',')
			for line in train_set:
				try:
					csvwriter.writerow([line['image'], line['label']])
				except OSError:
					print(line)


def folder_split():
	# val_percent = 0.2
	test_percent = 0.2
	data_path = '../../data/RSSCN7/train'
	# val_path = '../../data/AID/val'
	test_path = '../../data/RSSCN7/test'

	# if not os.path.exists(val_path):
	# 	os.mkdir(val_path)

	if not os.path.exists(test_path):
		os.mkdir(test_path)

	categories = os.listdir(data_path)
	for c in categories:
		im_files = np.array(os.listdir(os.path.join(data_path, c)))
		# val_cate_folder = os.path.join(val_path, c)
		# if not os.path.exists(val_cate_folder):
		# 	os.mkdir(val_cate_folder)

		test_cate_folder = os.path.join(test_path, c)
		if not os.path.exists(test_cate_folder):
			os.mkdir(test_cate_folder)

		num = len(im_files)
		rand_perm = np.random.permutation(range(num))
		# val_indexes = rand_perm[:int(val_percent*num)].astype(np.int)
		# test_indexes = rand_perm[int(val_percent*num):int(test_percent*num+val_percent*num)].astype(np.int)
		test_indexes = rand_perm[:int(test_percent*num)].astype(np.int)
		# val_files = [os.path.join(os.path.join(data_path, c), file) for file in im_files[val_indexes]]
		test_files = [os.path.join(os.path.join(data_path, c), file) for file in im_files[test_indexes]]

		# [shutil.move(src, val_cate_folder) for src in val_files]
		[shutil.move(src, test_cate_folder) for src in test_files]


folder_split()