import torch
from torch.utils.data import Dataset

from diagvibsix.dataset.dataset import Dataset as MtmdDatasetRaw
from diagvibsix.utils.auxiliaries import load_yaml
from diagvibsix.utils.dataset_utils import get_mt_labels


class DiagVibDataset(Dataset):
    def __init__(self, spec_yaml, seed):
        cache_path = spec_yaml + '.seed_{}.cache.pkl'.format(seed)
        self.dataset_raw = MtmdDatasetRaw(dataset_spec=load_yaml(spec_yaml), seed=seed, cache_path=cache_path)
        self.num_samples = len(self.dataset_raw.images)

    def __getitem__(self, i):
        image, question_answer, tag = self.dataset_raw.getitem(i).values()
        attribute = get_mt_labels(question_answer)

        return torch.tensor(image[0] / 255, dtype=torch.float32), torch.tensor(attribute)

    def __len__(self):
        return self.num_samples

    def get_config(self):
        config = {
            'ds_x_index': 0,
            'ds_attribute_index': 1,
            'ds_attritube_type_list': self.dataset_raw.tasks
        }
        return config
