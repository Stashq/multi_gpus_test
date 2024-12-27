import pytorch_lightning as pl
from torch import Tensor, nn, optim
from torch.optim import Optimizer


class CNNModel(pl.LightningModule):
    def __init__(
        self, conv_kernels: list[int], kernel_size: int = 5
    ) -> None:
        assert len(conv_kernels) > 1
        super().__init__()
        self.conv_kernels = conv_kernels
        self.kernel_size = kernel_size
        self.loss_function = nn.MSELoss()
        self.setup_()

    def setup_(self) -> None:
        layers = [
            nn.Conv2d(
                3, self.conv_kernels[0], kernel_size=self.kernel_size
            )
        ]
        layers += [
            nn.Conv2d(
                self.conv_kernels[i - 1],
                self.conv_kernels[i],
                kernel_size=self.kernel_size,
            )
            for i in range(1, len(self.conv_kernels))
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for conv_layer in self.layers:
            x = conv_layer(x)
        return x

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x = batch
        x_prim = self(x)
        loss: Tensor = self.loss_function(x, x_prim)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validational_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, _ = batch
        x_prim = self(x)
        loss: Tensor = self.loss_function(x, x_prim)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self) -> Optimizer:
        assert self.trainer.model is not None
        optimizer = optim.Adam(self.trainer.model.parameters(), lr=1e-3)
        return optimizer


class MLPModel(pl.LightningModule):
    def __init__(self, n_features: list[int]):
        assert len(n_features) > 1
        super().__init__()
        self.n_features = n_features
        self.loss_function = nn.MSELoss()
        self.setup_()

    def setup_(self) -> None:
        self.layers = nn.Sequential(
            *[
                nn.Linear(self.n_features[i], self.n_features[i + 1])
                for i in range(len(self.n_features) - 1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x = batch
        x_prim = self(x)
        loss: Tensor = self.loss_function(x, x_prim)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validational_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, _ = batch
        x_prim = self(x)
        loss: Tensor = self.loss_function(x, x_prim)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self) -> Optimizer:
        assert self.trainer.model is not None
        optimizer = optim.Adam(self.trainer.model.parameters(), lr=1e-3)
        return optimizer
