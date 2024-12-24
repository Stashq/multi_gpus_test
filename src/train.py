from src.data import DataModule
from src.model import Model
from pytorch_lightning import Trainer


def train(
    img_h: int, img_w: int, batch_size: int,
    dataset_size: int, conv_kernels: list[int]
) -> None:
    dm = DataModule(img_shape=(img_h, img_w))
    model = Model(conv_kernels=conv_kernels)
    trainer = Trainer()
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train(
        img_h=2640, img_w=2640, batch_size=16,
        dataset_size=1024, conv_kernels=[2048, 2048]
    )
