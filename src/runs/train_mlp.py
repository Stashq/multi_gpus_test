from typing import Literal

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

# from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.strategies.fsdp import FSDPStrategy
from pytorch_lightning.strategies.strategy import Strategy

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
    devices: list[int] | int | str = "auto",
    accelerator: str = "cpu",
    strategy: Strategy | str = "auto",
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


def _calculate_n_params(
    memory_gb: int | float,
    optim: Literal["adam", "sgd"],
    strategy_coef: int | float = 0,
) -> int:
    """Calculates number of parameters that fits into memory.

    Parameters
    ----------
    memory_gb : int | float
        Memory in gigabytes.
    optim : Literal["adam", "sgd"]
        Optimizer type. Allowed only adam or sgd.
    strategy_coef : int | float
        Coef added to divider, resulting from chosen strategy.

    Returns
    -------
    int
        Number of total model parameters.

    Raises
    ------
    ValueError
        If unknown optimizer types.
    """
    params = 4
    gradients = 4
    signal_state = 0

    divider = params + gradients + signal_state + strategy_coef
    if optim == "adam":
        param_copy = 4
        momentum = 4
        variance = 4
        divider += param_copy + momentum + variance
    else:
        raise ValueError(f'Unknown optimizer type "{optim}".')
    return int(memory_gb * _1GB / divider)


def _calculate_hidden_dim(n_params: int, input_len: int) -> int:
    """Calculates number of hidden dimention neurons in 4 layer MLP with
    following architecture: in_len | h_dim | h_dim | in_len.

    Parameters
    ----------
    n_params : int
        Total number of model params.
    input_len : int
        Number of input neurons.

    Returns
    -------
    int
        Number of neurons in single hidden layer.
    """
    return int(np.sqrt(n_params + input_len**2 + input_len + 1) - input_len - 1)


if __name__ == "__main__":
    in_features = 1
    n_params = _calculate_n_params(memory_gb=40, optim="adam")
    h_dim = _calculate_hidden_dim(n_params=n_params, input_len=in_features)
    train_mlp(
        vector_len=in_features,
        batch_size=2,
        dataset_size=4,
        n_features=[in_features, h_dim, h_dim, in_features],
        num_nodes=1,
        devices=[0, 1],
        accelerator="gpu",  # "cpu"
        strategy=FSDPStrategy(
            auto_wrap_policy="size_based",  # type: ignore[arg-type]
            min_num_params=1e6,
        ),
        # DeepSpeedStrategy(),  # "fsdp_native",
    )
