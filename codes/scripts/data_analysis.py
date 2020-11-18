# encoding: utf-8
import os
from matplotlib import pyplot as plt


label_dict = {}
with open('D:/data/RSSRAI/ClsName2id.txt', 'r', encoding='utf-8-sig') as f:
	im_list = f.readlines()
	for line in im_list:
		record = line.split(':')
		label_dict[record[0]] = record[1]

rootdir = 'D:/data/RSSRAI/train'
folders = os.listdir(rootdir) #列出文件夹下所有的目录与文件
num_per_class = []
for i in range(0, len(folders)):
	try:
		new_name = label_dict[folders[i]]
		path = os.path.join(rootdir, folders[i])
		new_path = os.path.join(rootdir, new_name)
		os.rename(path, new_path)
	except KeyError:
		print('already modified')
	# if os.path.isdir(path):
	# 	sublist = os.listdir(path)
	# 	string = u'class {} has {} images'.format(list[i], len(sublist))
	#
	# 	print(string)
	# 	num_per_class.append(len(sublist))

# plt.plot(folders, num_per_class)
# plt.show()