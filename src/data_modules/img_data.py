import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class NoiseImgDataset(Dataset[Tensor]):
    def __init__(
        self, img_shape: tuple[int, int] = (640, 640), n_examples: int = 20
    ) -> None:
        self.img_shape = img_shape
        self.n_examples = n_examples

    def __getitem__(self, idx: int) -> Tensor:
        if idx >= self.n_examples:
            raise IndexError()

        return torch.randn((3, *self.img_shape))

    def __len__(
        self,
    ) -> int:
        return self.n_examples


class ImgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_shape: tuple[int, int] = (640, 640),
        n_examples: int = 20,
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        self.img_shape = img_shape
        self.n_examples = n_examples
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader[Tensor]:
        return DataLoader(
            NoiseImgDataset(
                img_shape=self.img_shape, n_examples=self.n_examples
            ),
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader[Tensor]:
        return DataLoader(
            NoiseImgDataset(
                img_shape=self.img_shape, n_examples=self.n_examples
            ),
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader[Tensor]:
        return DataLoader(
            NoiseImgDataset(
                img_shape=self.img_shape, n_examples=self.n_examples
            ),
            batch_size=self.batch_size,
        )
