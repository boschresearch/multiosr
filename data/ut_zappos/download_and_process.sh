#!/bin/bash

mkdir -p archives

# download
wget --show-progress -O archives/ut-zap50k-images.zip http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
wget --show-progress -O archives/attr-ops-data.tar.gz https://www.cs.utexas.edu/~tushar/attribute-ops/attr-ops-data.tar.gz

# extract
unzip archives/ut-zap50k-images.zip -d ut-zappos-material
tar -zxvf archives/attr-ops-data.tar.gz attr-ops-data/data/ut-zap50k/metadata.t7
mv attr-ops-data/data/ut-zap50k/metadata.t7 ut-zappos-material
rm -r attr-ops-data

# process
mv ut-zappos-material/ut-zap50k-images ut-zappos-material/_images
python reorganize_utzap.py
python split_utzap.py
