import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class VectorDataset(Dataset[Tensor]):
    def __init__(
        self, vector_len: int = 256, n_examples: int = 20
    ) -> None:
        self.vector_len = vector_len
        self.n_examples = n_examples

    def __getitem__(self, idx: int) -> Tensor:
        if idx >= self.n_examples:
            raise IndexError()

        return torch.randn(self.vector_len)

    def __len__(
        self,
    ) -> int:
        return self.n_examples


class VectorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        vector_len: int = 256,
        n_examples: int = 20,
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        self.vector_len = vector_len
        self.n_examples = n_examples
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader[Tensor]:
        return DataLoader(
            VectorDataset(
                vector_len=self.vector_len, n_examples=self.n_examples
            ),
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader[Tensor]:
        return DataLoader(
            VectorDataset(
                vector_len=self.vector_len, n_examples=self.n_examples
            ),
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader[Tensor]:
        return DataLoader(
            VectorDataset(
                vector_len=self.vector_len, n_examples=self.n_examples
            ),
            batch_size=self.batch_size,
        )
