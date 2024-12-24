import pytorch_lightning as pl
from torch import nn, optim


class Model(pl.LightningModule):
    def __init__(self, conv_kernels: list[int]):
        assert len(conv_kernels) > 1
        self.conv_kernels = conv_kernels
        self.loss_function = nn.MSELoss()
        self.setup()

    def setup(self):
        layers = [nn.Conv2d(3, self.conv_kernels[0])]
        layers += [
            nn.Conv2d(self.conv_kernels[i], self.conv_kernels[i+1])
            for i in range(1, len(self.conv_kernels))
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for conv_layer in self.layers:
            x = conv_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_prim = self(x)
        loss = self.loss(x, x_prim)
        self.log("train_loss", loss)
        return loss

    def validational_step(self, batch, batch_idx):
        x, _ = batch
        x_prim = self(x)
        loss = self.loss(x, x_prim)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
