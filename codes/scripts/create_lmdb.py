import sys
import os.path
import pickle
import lmdb
import cv2


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.progress_bar import ProgressBar


def create_lmdb(paths_labels, lmdb_save_path):
	# configurations
	# img_folder = 'D:/dataloader/RSSCN7/train_lr/*'  # glob matching pattern
	# lmdb_save_path = 'D:/dataloader/RSSCN7/rsscn7_bicLRx4.lmdb'  # must end with .lmdb

	dataset = []
	data_size = 0

	print('Read images...')
	pbar = ProgressBar(len(paths_labels))
	for i, (v, l) in enumerate(paths_labels):
		img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
		dataset.append(img)
		data_size += img.nbytes
		# pbar.update('Read {}'.format(v))
	env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
	print('Finish reading {} images.\nWrite lmdb...'.format(len(paths_labels)))

	pbar = ProgressBar(len(paths_labels))
	with env.begin(write=True) as txn:  # txn is a Transaction object
		for i, (v, l) in enumerate(paths_labels):
			base_name = os.path.splitext(os.path.basename(v))[0]
			key = base_name.encode('ascii')
			data = dataset[i]
			if dataset[i].ndim == 2:
				H, W = dataset[i].shape
				C = 1
			else:
				H, W, C = dataset[i].shape
			meta_key = (base_name + '.meta').encode('ascii')
			meta = '{:d}, {:d}, {:d}'.format(H, W, C)
			label_key = (base_name+'.label').encode('ascii')
			label = '{:d}'.format(l)
			# The encode is only essential in Python 3
			txn.put(key, data)
			txn.put(meta_key, meta.encode('ascii'))
			txn.put(label_key, label.encode('ascii'))
			# pbar.update('Write {}'.format(v))
	print('Finish writing lmdb.')

	# create keys cache
	keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
	env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
	with env.begin(write=False) as txn:
		print('Create lmdb keys cache: {}'.format(keys_cache_file))
		keys = [key.decode('ascii') for key, _ in txn.cursor()]
		pickle.dump(keys, open(keys_cache_file, "wb"))
	print('Finish creating lmdb keys cache.')
