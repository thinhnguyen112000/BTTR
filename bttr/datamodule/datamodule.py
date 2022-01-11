import os
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm
import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import random
from .vocab import CROHMEVocab
import pickle
import cv2
vocab = CROHMEVocab()

MAX_SIZE = 32e4  # change here accroading to your GPU memory


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    batch = batch[0]
    return Batch(batch[0], batch[1], batch[2], batch[3])


class LatexDataloader(torch.utils.data.IterableDataset):
    def __init__(self, name_dir_images: str,
                 dir_path: str, batch_size: int,
                 batch_Imagesize: int = MAX_SIZE,
                 maxlen: int = 200,
                 maxImagesize: int = MAX_SIZE,
                 type_data: str = 'train',
                 isLoaded: bool = True,
                 isShuffle: bool = False,
                 device='cuda'):

        self.name_dir_images = name_dir_images
        self.dir_path = dir_path
        self.type_data = type_data

        self.batch_size = batch_size
        self.batch_Imagesize = batch_Imagesize
        self.maxlen = maxlen
        self.maxImagesize = maxImagesize
        self.device = device
        self.isLoaded = isLoaded
        self.isShuffle = isShuffle

        self.data = self.get_batch_data()

    def get_batch_data(self):
        total_list_data = []
        list_data = []
        with open(self.dir_path + self.type_data + '.txt', mode='r') as f:
            print(f'Start loading {self.type_data} data...')
            datasets = f.readlines()

        number_samples = int(len(datasets) * 10 / 100)
        counter = 0
        for line in tqdm(datasets[:number_samples]):
            img_name = line[:13]
            formula = line[14:]
            image_path = f"{self.dir_path + self.name_dir_images}/{img_name}.jpg".replace('Train_', "")
            if os.path.exists(image_path):
                if (counter % self.batch_size == 0 and counter != 0):
                    total_list_data.append(list_data)
                    list_data = []
                counter += 1
                list_data.append((img_name, image_path, formula))
        if (len(datasets) % self.batch_size):
            list_data.append(random.sample(total_list_data[random.randint(0, len(total_list_data) - 1)],
                                           self.batch_size - len(list_data)))
            total_list_data.append(list_data)
        return total_list_data

    # def __getitem__(self, item):
    #     imgs_name, images, formulars = [], [], []
    #     for img_name, image_path, formular in self.data[item]:
    #         image = cv2.imread(image_path, 0)
    #         image = transforms.ToTensor()(image)
    #         images.append(image)
    #         imgs_name.append(img_name)
    #         formulars.append(formular)
    #
    #     heights_x = [s.size(1) for s in images]
    #     widths_x = [s.size(2) for s in images]
    #
    #     n_samples = len(heights_x)
    #     max_height_x = max(heights_x)
    #     max_width_x = max(widths_x)
    #
    #     x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    #     x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    #
    #     seqs_y = [vocab.words2indices(x) for x in formulars]
    #
    #     for index, image_path in enumerate(images):
    #         x[index, :, : heights_x[index], : widths_x[index]] = image_path
    #         x_mask[index, : heights_x[index], : widths_x[index]] = 0
    #
    #     return imgs_name, torch.FloatTensor(x), x_mask.long(), seqs_y

    def __iter__(self):
        path_pkl = self.dir_path + self.type_data + '.pkl'

        if (self.isLoaded):
            with open(path_pkl, 'r') as f:
                dataloader = pickle.load(f)
                if self.isShuffle:
                    random.shuffle(dataloader)
                for imgs_name, x, x_mask, seqs_y in dataloader:
                    yield imgs_name, torch.FloatTensor(x), x_mask.long(), seqs_y
        else:
            list_pkl = []
            for batch_data in self.data:
                imgs_name, images, formulars = [], [], []
                for img_name, image_path, formular in batch_data:
                    image = cv2.imread(image_path, 0)
                    image = transforms.ToTensor()(image)
                    images.append(image)
                    imgs_name.append(img_name)
                    formulars.append(formular)

                heights_x = [s.size(1) for s in images]
                widths_x = [s.size(2) for s in images]

                n_samples = len(heights_x)
                max_height_x = max(heights_x)
                max_width_x = max(widths_x)

                x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
                x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)

                seqs_y = [vocab.words2indices(x) for x in formulars]

                for index, image in enumerate(images):
                    x[index, :, : heights_x[index], : widths_x[index]] = image
                    x_mask[index, : heights_x[index], : widths_x[index]] = 0
                list_pkl.append((imgs_name, x, x_mask, seqs_y))
                yield imgs_name, torch.FloatTensor(x), x_mask.long(), seqs_y

            with open(path_pkl, 'w') as f:
                pickle.dump(list_pkl, f)

            self.isLoaded = True

    def __len__(self):
        return len(self.data)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            name_dir_images: str = "",
            dir_path: str = "",
            batch_size: int = 8,
            num_workers: int = 5,
    ) -> None:
        super().__init__()
        assert isinstance(dir_path, str)
        self.dir_path = dir_path
        self.name_dir_images = name_dir_images
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"Load data from: {dir_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = LatexDataloader(name_dir_images=self.name_dir_images,
                                                 dir_path=self.dir_path,
                                                 type_data='train',
                                                 isLoaded=False,
                                                 batch_size=self.batch_size,
                                                 isShuffle=True)
            self.val_dataset = LatexDataloader(name_dir_images=self.name_dir_images,
                                               dir_path=self.dir_path,
                                               type_data='val',
                                               isLoaded=False,
                                               batch_size=self.batch_size,
                                               )
        if stage == 'test' or stage is None:
            self.test_dataset = LatexDataloader(name_dir_images=self.name_dir_images,
                                                dir_path=self.dir_path,
                                                type_data='test',
                                                isLoaded=False,
                                                batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )


if __name__ == '__main__':
    total_list_data = []
    list_data = []
    batch_size = 8
    dir_path = 'G:/NCKH_DUT/OCR/dataset/'
    type_data = 'train'
    name_dir_images = 'ProcessImage'
    with open(dir_path + type_data + '.txt', mode='r') as f:
        print(f'Start loading {type_data} data...')
        datasets = f.readlines()

    number_samples = int(len(datasets) * 1 / 100)
    for id, line in tqdm(enumerate(datasets[:number_samples])):
        img_name = line[:13]
        formula = line[14:]
        image_path = f"{dir_path + name_dir_images}/{img_name}.jpg".replace('Train_', "")
        if os.path.exists(image_path):
            if id != 0 and id % batch_size == 0:
                total_list_data.append(list_data)
            list_data.append((img_name, image_path, formula))
    if (len(datasets) % batch_size):
        list_data.append(random.sample(total_list_data[0], batch_size - len(list_data)))
        total_list_data.append(list_data)

    for batch_data in total_list_data:
        imgs_name, images, formulars = [], [], []
        for img_name, image_path, formular in batch_data:
            image = cv2.imread(image_path, 0)
            image = transforms.ToTensor()(image)
            images.append(image)
            imgs_name.append(img_name)
            formulars.append(formular)

        heights_x = [s.size(1) for s in images]
        widths_x = [s.size(2) for s in images]

        n_samples = len(heights_x)
        max_height_x = max(heights_x)
        max_width_x = max(widths_x)

        x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
        x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)

        seqs_y = [vocab.words2indices(x) for x in formulars]

        for index, image_path in enumerate(images):
            x[index, :, : heights_x[index], : widths_x[index]] = image_path
            x_mask[index, : heights_x[index], : widths_x[index]] = 0
