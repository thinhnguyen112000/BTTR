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
    assert len(batch) == 1
    batch = batch[0]
    return Batch(batch[0], batch[1], batch[2], batch[3])


class LatexDataloader(torch.utils.data.IterableDataset):
    def __init__(self, name_dir_images: str,
                 dir_path: str, batch_size: int,
                 maxlen: int = 200,
                 type_data: str = 'train',
                 image_size: tuple = (64, 256),
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
            print(f'Loaded {self.type_data} datasets')
            with open(path_pkl, 'rb') as f:
                total_list_data = pickle.load(f)
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
                    if len(indices) <= self.maxlen:
                        list_data.append((img_name, image_path, indices))
            if (len(datasets) % self.batch_size):
                list_data.append(random.sample(total_list_data[random.randint(0, len(total_list_data) - 1)],
                                               self.batch_size - len(list_data)))
                total_list_data.append(list_data)

            with open(self.dir_path + self.type_data + '.pkl', 'wb') as f:
                pickle.dump(total_list_data, f)
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

    def resize_aspect_ratio(self, max_h, max_w):
        height_default, width_default = self.image_size[0], self.image_size[1]
        ratio = width_default/height_default
        new_h, new_w = min(max_h, max_w/ratio), min(max_w, max_h*ratio)
        return new_h, new_w

    def compute_batch(self, batch_data):
        imgs_name, images, seqs_y = [], [], []
        for img_name, image_path, seq_y in batch_data:
            image = cv2.imread(image_path, 0)
            image = transforms.ToTensor()(image)
            images.append(image)
            imgs_name.append(img_name)
            seqs_y.append(seq_y)

        heights_x = [s.size(1) for s in images]
        widths_x = [s.size(2) for s in images]

        n_samples = len(heights_x)
        max_height_x = max(heights_x)
        max_width_x = max(widths_x)

        max_height_x, max_width_x = self.resize_aspect_ratio(max_height_x, max_width_x)

        x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
        x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)

        for index, image in enumerate(images):
            h_x_img, w_x_img = self.resize_aspect_ratio(heights_x[index], widths_x[index])
            x[index, (max_height_x-h_x_img)//2:(max_height_x-h_x_img)//2 + h_x_img,
                     (max_width_x-w_x_img)//2:(max_width_x-w_x_img)//2 + w_x_img]\
                     = cv2.resize(image, (w_x_img, h_x_img), interpolation=cv2.INTER_NEAREST)
            x_mask[index, : h_x_img, : w_x_img] = 0
        return imgs_name, x, x_mask, seqs_y

    def __iter__(self):
        if self.isShuffle:
            random.shuffle(self.data)
        for batch_data in self.data:
            imgs_name, x, x_mask, seqs_y = self.compute_batch(batch_data)
            yield imgs_name, torch.FloatTensor(x), x_mask.long(), seqs_y

    def __len__(self):
        return len(self.data)


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
    data = CROHMEDatamodule()
