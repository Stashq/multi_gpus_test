import pytorch_lightning as pl
from torch.data.utils import Dataset, DataLoader
import torch


class RandomImgs(Dataset):
    def __init__(
        self,
        img_shape: tuple[int, int] = (640, 640),
        n_examples: int = 20
    ) -> None:
        self.img_shape = img_shape
        self.n_examples = n_examples

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx >= self.n_examples:
            raise IndexError()

        return torch.randn(self.img_shape)

    def __len__(self,) -> int:
        return self.n_examples


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_shape: tuple[int, int] = (640, 640),
        n_examples: int = 20
    ) -> None:
        super().__init__()
        self.img_shape = img_shape
        self.n_examples = n_examples

    def train_dataloader(self):
        return DataLoader(RandomImgs(
            img_shape=self.img_shape, n_examples=self.n_examples))

    def val_dataloader(self):
        return DataLoader(RandomImgs(
            img_shape=self.img_shape, n_examples=self.n_examples))

    def test_dataloader(self):
        return DataLoader(RandomImgs(
            img_shape=self.img_shape, n_examples=self.n_examples))
