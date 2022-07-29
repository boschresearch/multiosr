import torch
import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def load_list_file(filepath):
    output_list = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            output_list.append(line.strip())

    return output_list


def load_split_image_names(dataset_dir, split_type, split):
    split_filepath = os.path.join(dataset_dir, f'split_{split_type}', f'{split}.txt')
    with open(split_filepath, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]

    return image_names


def load_image_name_attributes_dict(dataset_dir, split_type):
    # load name to index dict
    material_names = load_list_file(os.path.join(dataset_dir, f'split_{split_type}', '0_selected_material_names.txt'))
    type_names = load_list_file(os.path.join(dataset_dir, f'split_{split_type}', '1_selected_type_names.txt'))
    material_name_to_index_dict = {n: i for i, n in enumerate(material_names)}
    type_name_to_index_dict = {n: i for i, n in enumerate(type_names)}

    # load metadata
    metadata = torch.load(os.path.join(dataset_dir, 'metadata.t7'))

    # load labels
    image_name_attributes_dict = {}
    for instance in metadata:
        if (instance['attr'] not in material_name_to_index_dict) or (instance['obj'] not in type_name_to_index_dict):
            continue

        image_name = instance['image']
        img_label = (material_name_to_index_dict[instance['attr']], type_name_to_index_dict[instance['obj']])
        image_name_attributes_dict[image_name] = img_label

    return image_name_attributes_dict


def load_image_metadata_dict(dataset_dir, split_type, split):
    # load image_filenames
    image_names = load_split_image_names(dataset_dir, split_type, split)
    image_filenames = [os.path.join(dataset_dir, 'images', image_name) for image_name in image_names]

    # load labels
    image_name_attributes_dict = load_image_name_attributes_dict(dataset_dir, split_type)
    image_attributes = [image_name_attributes_dict[image_name] for image_name in image_names]

    return image_filenames, image_attributes


class UTZapposDataset(Dataset):
    def __init__(self, dataset_dir, split_title, split):
        assert split in ['train', 'val', 'test']
        self.split_title = split_title
        self.split = split
        self.dataset_dir = dataset_dir

        self.image_filenames, self.image_attributes = load_image_metadata_dict(
            self.dataset_dir, self.split_title, self.split)

        self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, i):
        # get image
        image_filename = self.image_filenames[i]
        image = Image.open(image_filename).convert("RGB")

        return self.transform(image), torch.tensor(self.image_attributes[i]).long()

    def __len__(self):
        return len(self.image_filenames)

    @staticmethod
    def get_config():
        config = {
            'ds_x_index': 0,
            'ds_attribute_index': 1
        }
        return config
