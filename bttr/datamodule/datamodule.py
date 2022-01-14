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
import numpy as np
from collections import Counter
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
    assert len(batch) == 1
    batch = batch[0]
    return Batch(batch[0], batch[1], batch[2], batch[3])


class LatexDataloader(torch.utils.data.IterableDataset):
    def __init__(self, name_dir_images: str,
                 dir_path: str, batch_size: int,
                 maxlen: int = 200,
                 type_data: str = 'train',
                 image_size: tuple = (128, 512),
                 isLoaded: bool = True,
                 isShuffle: bool = False):

        self.name_dir_images = name_dir_images
        self.dir_path = dir_path
        self.type_data = type_data

        self.batch_size = batch_size
        if type_data in ['test', 'val']:
            self.batch_size = 1
        self.maxlen = maxlen
        self.isLoaded = isLoaded
        self.isShuffle = isShuffle
        self.image_size = image_size
        self.data = self.create_batch_data()

    def create_batch_data(self):
        total_list_data = []
        list_data = []
        path_pkl = self.dir_path + self.type_data + '.pkl'
        if (self.isLoaded):
            assert os.path.exists(path_pkl)

            with open(path_pkl, 'rb') as f:
                total_list_data = pickle.load(f)
            print(f'Loaded {self.type_data} datasets: {len(total_list_data)} sample')
        else:
            print(f'Start loading {self.type_data} data...')
            datasets = open(self.dir_path + self.type_data + '.txt', mode='r').read().splitlines()

            number_samples = int(len(datasets) * 100 / 100)
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
                    indices = vocab.words2indices(formula)
                    list_data.append((img_name, image_path, indices))
            if (len(datasets) % self.batch_size):
                list_data.append(random.sample(total_list_data[random.randint(0, len(total_list_data) - 1)],
                                               self.batch_size - len(list_data)))
                total_list_data.append(list_data)

            with open(self.dir_path + self.type_data + '.pkl', 'wb') as f:
                pickle.dump(total_list_data, f)
        return total_list_data

    def __getitem__(self, item):
        return self.compute_batch(self.data[item])

    def create_new_shape(self, h_w, sh_sw):
        h, w = h_w
        sh, sw = sh_sw

        if h > sh and w > sw:
            ratio = w / sw
            new_w = w / ratio
            new_h = h / ratio
            if(new_h > sh):
                ratio = new_h / sh
                new_h = np.round(new_h / ratio).astype(int)
                new_w = np.round(new_w / ratio).astype(int)
            else:
                new_h = np.round(new_h).astype(int)
                new_w = np.round(new_w).astype(int)

        elif h > sh and w < sw:
            ratio = h / sh
            new_h = np.round(h / ratio).astype(int)
            new_w = np.round(w / ratio).astype(int)
        elif h < sh and w > sw:
            ratio = w / sw
            new_h = np.round(h / ratio).astype(int)
            new_w = np.round(w / ratio).astype(int)
        else:
            ratio = min(sh / h, sw / w)
            new_h = np.round(h * ratio).astype(int)
            new_w = np.round(w * ratio).astype(int)


        pad_left, pad_right = np.floor((sw-new_w)/2).astype(int), np.ceil((sw-new_w)/2).astype(int)
        pad_top, pad_bottom = np.floor((sh-new_h)/2).astype(int), np.ceil((sh-new_h)/2).astype(int)
        return (new_h, new_w), (pad_top, pad_bottom, pad_left, pad_right)


    def read_image(self, image_path):
        image = cv2.imread(image_path, 0)
        h, w = image.shape
        sh, sw = self.image_size



        # interpolation method
        interp = cv2.INTER_AREA # shrinking image

        # compute scaling and pad sizing
        new_wh, pad = self.create_new_shape((h, w), (sh, sw))
        new_h, new_w = new_wh
        pad_top, pad_bot, pad_left, pad_right = pad

        # resize and add padding
        scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
        dict_frequency = dict(sum(map(Counter, scaled_img), Counter()))
        best_frequency_value = list(dict_frequency.keys())[list(dict_frequency.values()).index(max(dict_frequency.values()))]
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=int(best_frequency_value))
        scaled_img = transforms.ToTensor()(scaled_img)
        return scaled_img

    def compute_batch(self, batch_data):
        imgs_name, images, seqs_y = [], [], []
        for img_name, image_path, seq_y in batch_data:
            image = self.read_image(image_path)
            images.append(image)
            imgs_name.append(img_name)
            seqs_y.append(seq_y)

        n_samples = len(images)
        height_default, width_default = self.image_size
        x = torch.zeros(n_samples, 1, height_default, width_default)
        x_mask = torch.ones(n_samples, height_default, width_default, dtype=torch.bool)

        for index, image in enumerate(images):
            x[index, :height_default, :width_default] = image
            x_mask[index, : height_default, : width_default] = 0
        return imgs_name, x, x_mask, seqs_y

    def __iter__(self):
        if self.isShuffle:
            random.shuffle(self.data)
        for batch_data in self.data:
            imgs_name, x, x_mask, seqs_y = self.compute_batch(batch_data)
            yield imgs_name, torch.FloatTensor(x), x_mask.long(), seqs_y

    # def __len__(self):
    #     return len(self.data)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            name_dir_images: str = "",
            dir_path: str = "",
            batch_size: int = 8,
            num_workers: int = 5,
            image_size: tuple = (),
    ) -> None:
        super().__init__()
        if image_size is None:
            image_size = []
        assert isinstance(dir_path, str)
        self.dir_path = dir_path
        self.name_dir_images = name_dir_images
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
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
    def create_new_shape(h_w, sh_sw):
        h, w = h_w
        sh, sw = sh_sw

        if h > sh and w > sw:
            ratio = w / sw
            new_w = w / ratio
            new_h = h / ratio
            if (new_h > sh):
                ratio = new_h / sh
                new_h = np.round(new_h / ratio).astype(int)
                new_w = np.round(new_w / ratio).astype(int)
            else:
                new_h = np.round(new_h).astype(int)
                new_w = np.round(new_w).astype(int)

        elif h > sh and w < sw:
            ratio = h / sh
            new_h = np.round(h / ratio).astype(int)
            new_w = np.round(w / ratio).astype(int)
        elif h < sh and w > sw:
            ratio = w / sw
            new_h = np.round(h / ratio).astype(int)
            new_w = np.round(w / ratio).astype(int)
        else:
            ratio = min(sh / h, sw / w)
            new_h = np.round(h * ratio).astype(int)
            new_w = np.round(w * ratio).astype(int)

        pad_left, pad_right = np.floor((sw - new_w) / 2).astype(int), np.ceil((sw - new_w) / 2).astype(int)
        pad_top, pad_bottom = np.floor((sh - new_h) / 2).astype(int), np.ceil((sh - new_h) / 2).astype(int)
        return (new_h, new_w), (pad_top, pad_bottom, pad_left, pad_right)


    def read_image(image_path, image_size):
        image = cv2.imread(image_path, 0)
        cv2.imshow('debug1', image)
        cv2.waitKey(0)
        h, w = image.shape
        sh, sw = image_size

        interp = cv2.INTER_AREA
        new_wh, pad = create_new_shape((h, w), (sh, sw))
        new_h, new_w = new_wh
        pad_top, pad_bot, pad_left, pad_right = pad
        scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
        dict_frequency = dict(sum(map(Counter, scaled_img), Counter()))
        best_frequency_value = list(dict_frequency.keys())[list(dict_frequency.values())
                                                    .index(max(dict_frequency.values()))]
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=int(best_frequency_value))
        cv2.imshow('debug2', scaled_img)
        cv2.waitKey(0)
        scaled_img = transforms.ToTensor()(scaled_img)
        return scaled_img


    read_image('Train_0000001.png', (128, 512))

