import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset

class CUBDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self, root_dir,
                 target_name, confounder_names,
                 augment_data=False,
                 model_type=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        self.data_dir = os.path.join(
            self.root_dir,
            'data',
            '_'.join([self.target_name] + self.confounder_names))

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # # Read in metadata
        # self.metadata_df = pd.read_csv(
        #     os.path.join(self.data_dir, 'metadata.csv'))

        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'shuffle_metadata_seed5.csv'))

        
        # # データセット作成用
        # shuffled_indices = np.random.permutation(len(self.metadata_df))

        # # Step 3: Calculate the number of samples for each split 
        # ## train : val : test = 6 : 2 : 2
        # n_total = len(self.metadata_df) 
        # n_train = int(0.6 * n_total)
        # n_val = int(0.2 * n_total)
        # n_test = n_total - n_train - n_val

        # # Step 4: Assign 0, 1, 2 based on the calculated counts
        # self.metadata_df.loc[shuffled_indices[:n_train], 'split'] = 0
        # self.metadata_df.loc[shuffled_indices[n_train:n_train + n_val], 'split'] = 1
        # self.metadata_df.loc[shuffled_indices[n_train + n_val:], 'split'] = 2

        # # Step 5: Save the updated metadata to a new CSV file
        # updated_metadata_path = os.path.join(self.data_dir,'shuffle_metadata_seed5.csv')
        # self.metadata_df.to_csv(updated_metadata_path, index=False)

        # print(f'Updated metadata saved to {updated_metadata_path}')

        # self.metadata_df = pd.read_csv(
        #     os.path.join(self.data_dir, 'shuffle_metadata_seed5.csv'))
        
        # self.metadata_df = self.metadata_df[0:1000] ## デバック用に変更
        ###

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # Set transform
        if model_attributes[self.model_type]['feature_type']=='precomputed':
            self.features_mat = torch.from_numpy(np.load(
                os.path.join(root_dir, 'features', model_attributes[self.model_type]['feature_filename']))).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = None
            self.train_transform = get_transform_cub(
                self.model_type,
                train=True,
                augment_data=augment_data)
            self.eval_transform = get_transform_cub(
                self.model_type,
                train=False,
                augment_data=augment_data)


def get_transform_cub(model_type, train, augment_data):
    scale = 256.0/224.0
    target_resolution = model_attributes[model_type]['target_resolution']
    assert target_resolution is not None

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform
