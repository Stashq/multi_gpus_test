from src.data import DataModule
from src.model import Model
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy


def train(
    img_h: int, img_w: int, batch_size: int,
    dataset_size: int, conv_kernels: list[int],
    gpus: int, num_nodes: int, accelerator: str = "ddp"
) -> None:
    dm = DataModule(img_shape=(img_h, img_w))
    model = Model(conv_kernels=conv_kernels)
    trainer = Trainer()
    trainer.fit(
        model,
        datamodule=dm,
        gpus=gpus, num_nodes=num_nodes, accelerator=accelerator
    )


if __name__ == "__main__":
    train(
        img_h=2640, img_w=2640, batch_size=16,
        dataset_size=1024, conv_kernels=[2048, 2048],
        gpus=2, num_nodes=1, accelerator=DDPStrategy()
    )
