from typing import Literal

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from src.data_modules.vector_data import VectorDataModule
from src.models import MLPModel

_1GB = 1073741824

# from pytorch_lightning.strategies import (
#     # DDPStrategy,
#     DDPFullyShardedStrategy
# )


# def train_cnn(
#     img_h: int, img_w: int, batch_size: int,
#     dataset_size: int, conv_kernels: list[int],
#     num_nodes: int,
#     devices: int = 1,
#     accelerator: str = "cpu",
#     strategy: str = "fsdp_native",
# ) -> None:
#     dm = ImgDataModule(img_shape=(img_h, img_w), batch_size=batch_size)
#     model = CNNModel(conv_kernels=conv_kernels)
#     trainer = Trainer(
#         num_nodes=num_nodes,
#         devices=devices,
#         accelerator=accelerator,
#         auto_select_gpus=True,
#         strategy=strategy
#     )
#     trainer.fit(
#         model,
#         datamodule=dm,
#     )


def train_mlp(
    vector_len: int,
    batch_size: int,
    dataset_size: int,
    n_features: list[int],
    num_nodes: int,
    devices: int = 1,
    accelerator: str = "cpu",
    strategy: str | None = None,
) -> None:
    dm = VectorDataModule(
        vector_len=vector_len,
        n_examples=dataset_size,
        batch_size=batch_size,
    )
    model = MLPModel(n_features=n_features)
    trainer = Trainer(
        num_nodes=num_nodes,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        enable_checkpointing=False,
        logger=CSVLogger("lightning_logs/"),
        max_epochs=2,
    )
    trainer.fit(
        model,
        datamodule=dm,
    )


def _calculate_hidden_dim(
    memory_gb: int | float, input_len: int, optim: Literal["adam", "sgd"]
) -> int:
    """Calculates number of hidden dimention neurons in 4 layer MLP with
    following architecture: in_len | h_dim | h_dim | in_len.

    Parameters
    ----------
    memory_gb : int
        Memory size that model should have in gigabytes.
    input_len : int
        Number of input neurons.
    optim : Literal[&quot;adam&quot;, &quot;sgd&quot;]
        Optimizer type. Allowed only adam or sgd.

    Returns
    -------
    int
        Number of neurons in single hidden layer.
    """
    if optim == "sgd":
        divider = 2
    elif optim == "adam":
        divider = 4
    else:
        raise ValueError(f'Unknown optimizer type "{optim}".')
    return int(
        np.sqrt(memory_gb * _1GB / divider + input_len**2 + input_len + 1)
        - input_len
        + 1
    )


if __name__ == "__main__":
    in_features = 1
    h_dim = _calculate_hidden_dim(
        memory_gb=40, input_len=in_features, optim="adam"
    )
    train_mlp(
        vector_len=in_features,
        batch_size=16,
        dataset_size=32,
        n_features=[in_features, h_dim, h_dim, in_features],
        num_nodes=1,
        devices=3,
        accelerator="gpu",  # "cpu"  # DDPStrategy(),
        strategy="fsdp_native",
    )
