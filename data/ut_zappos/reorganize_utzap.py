# This source code is from attribute-as-operator
#   (https://github.com/Tushar-N/attributes-as-operators/blob/master/utils/reorganize_utzap.py)
# Copyright (c) 2018
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import os
import torch
import glob
import shutil
import tqdm

root = 'ut-zappos-material'
os.makedirs(root+'/images')

data = torch.load(root+'/metadata.t7')
for instance in tqdm.tqdm(data):
	image, attr, obj = instance['_image'], instance['attr'], instance['obj']
	old_file = '%s/_images/%s'%(root, image)
	new_dir = '%s/images/%s_%s/'%(root, attr, obj)
	os.makedirs(new_dir, exist_ok=True)
	shutil.copy(old_file, new_dir)

