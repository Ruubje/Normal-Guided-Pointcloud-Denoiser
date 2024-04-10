from dataclasses import (
    dataclass
)
from torch import (
    cat as torch_cat,
    int64 as torch_int64,
    tensor as torch_tensor,
    zeros as torch_zeros
)
from torch.optim import Adam as torch_Adam
from torch.nn.functional import (
    cosine_embedding_loss as torch_cos_loss,
    mse_loss as torch_val_loss,
    normalize as torch_normalize
)
from torch.nn import (
    BatchNorm1d as torch_BatchNorm1d,
    Dropout as torch_Dropout,
    LeakyReLU as torch_LeakyReLU,
    Linear as torch_Linear,
    Sequential as torch_Sequential
)
from torch_geometric.nn.conv import (
    DynamicEdgeConv as tg_DynamicEdgeConv,
    EdgeConv as tg_EdgeConv
)
from torch_geometric.nn.pool import (
    global_mean_pool as tg_global_mean_pool,
    global_max_pool as tg_global_max_pool
)
import pytorch_lightning as pl
import torchmetrics
import torchvision

@dataclass
class NetworkModelConfiguration():
    k: int = 8
    lr: float = 0.001
    num_edgeconv: int = 3
    num_dynamic_edgeconv: int = 3
    num_prepool_linear: int = 1
    number_postpool_linear_layers: int = 3
    in_channels: int = 8
    feature_channels: torch_tensor = torch_tensor([64, 64, 128, 256, 256, 256, 512, 256, 64], dtype=torch_int64)
    out_channels: int = 3
    dropout: float = 0.5

    def __post_init__(self):
        sum_layers = self.number_postpool_linear_layers + self.num_dynamic_edgeconv + self.num_edgeconv + self.num_prepool_linear
        if sum_layers != self.feature_channels.size(0) + 1:
            raise ValueError("The feature channel sizes array does not match the sum of layers. This configuration is invalid.")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("Dropout should be between 0 and 1. It represents a percentage of weights that randomly get set to zero at the end of the network.")
        if self.lr < 0 or self.lr > 1:
            raise ValueError("Learning Rate should be between 0 and 1. It represents the speed of learning and is passed to the optimizer of the model.")

class Patch2NormalModel(pl.LightningModule):

    def __init__(self, config: NetworkModelConfiguration = NetworkModelConfiguration()):
        # Init Lightning Module
        super().__init__()

        # Init Config
        self.config = config

        # Init network layers
        for i in range(config.feature_channels.size(0)):
            if i < config.num_edgeconv:
                in_features = i == 0 and config.in_channels or config.feature_channels[i-1]
                out_features = config.feature_channels[i]
                layer = tg_EdgeConv(
                    nn=torch_Sequential(
                        torch_Linear(
                            in_features=2*in_features,
                            out_features=out_features,
                            bias=False
                        ),
                        torch_BatchNorm1d(num_features=out_features),
                        torch_LeakyReLU(negative_slope=0.2)
                    ),
                    aggr="max"
                )
            elif i < config.num_dynamic_edgeconv + config.num_edgeconv:
                in_features = config.feature_channels[i-1]
                out_features = config.feature_channels[i]
                layer = tg_DynamicEdgeConv(
                    nn=torch_Sequential(
                        torch_Linear(
                            in_features=2*in_features,
                            out_features=out_features,
                            bias=False
                        ),
                        torch_BatchNorm1d(num_features=out_features),
                        torch_LeakyReLU(negative_slope=0.2)
                    ),
                    k=config.k,
                    aggr="max"
                )
            elif i < config.num_edgeconv + config.num_dynamic_edgeconv + config.num_prepool_linear:
                in_features = config.feature_channels[i-1] if i > config.num_edgeconv + config.num_dynamic_edgeconv else config.feature_channels[:i].sum()
                out_features = config.feature_channels[i]
                layer = torch_Sequential(
                    torch_Linear(
                        in_features=in_features,
                        out_features=out_features,
                        bias=False
                    ),
                    torch_BatchNorm1d(num_features=out_features),
                    torch_LeakyReLU(negative_slope=0.2)
                )
            else:
                first_after_pool = i == config.num_edgeconv + config.num_dynamic_edgeconv + config.num_prepool_linear
                in_features = config.feature_channels[i-1] * (2 if first_after_pool else 1)
                out_features = config.feature_channels[i]
                layer = torch_Sequential(
                    torch_Linear(
                        in_features=in_features,
                        out_features=out_features
                    ),
                    torch_BatchNorm1d(num_features=out_features),
                    torch_Dropout(p=config.dropout)
                )
            setattr(self, f"layer{i}", layer)
        setattr(self, f"lastLayer", torch_Linear(
            in_features=config.feature_channels[-1],
            out_features=config.out_channels
        ))

    def forward(self, x, edge_index, batch):
        _config = self.config
        num_convs = _config.num_edgeconv + _config.num_dynamic_edgeconv
        pool_sizes = torch_cat([torch_zeros(1, dtype=torch_int64), _config.feature_channels[:num_convs].cumsum(dim=0)])
        x_cat = torch_zeros(x.size(0), pool_sizes[-1])
        for i in range(_config.feature_channels.size(0)):
            if i < _config.num_edgeconv:
                layer = getattr(self, f"layer{i}")
                x = layer(x, edge_index)
                x_cat[:, pool_sizes[i]:pool_sizes[i+1]] = x
            elif i < num_convs:
                layer = getattr(self, f"layer{i}")
                x = layer(x, batch)
                x_cat[:, pool_sizes[i]:pool_sizes[i+1]] = x
            elif i < num_convs + _config.num_prepool_linear:
                # Set last_x to be x_cat for the first prepool linear layer.
                if i == num_convs:
                    x = x_cat
                layer = getattr(self, f"layer{i}")
                x = layer(x)
            else:
                # Do pooling only first time.
                if i == num_convs + _config.num_prepool_linear:
                    x1 = tg_global_max_pool(x, batch)
                    x2 = tg_global_mean_pool(x, batch)
                    x = torch_cat((x1, x2), dim=-1)
                layer = getattr(self, f"layer{i}")
                x = layer(x)
        lastLayer = getattr(self, f"lastLayer")
        return lastLayer(x)
    
    def training_step(self, batch, batch_idx):
        val_loss, cos_loss, normals, _y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_val_loss": val_loss,
                "train_cos_loss": cos_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # if batch_idx % 100 == 0:
        #     x = x[:8]
        #     grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
        #     self.logger.experiment.add_image("mnist_images", grid, self.global_step)
        return {"loss": val_loss, "val_loss": val_loss, "cos_loss": cos_loss, "output": normals, "y": _y}

    def validation_step(self, batch, batch_idx):
        val_loss, cos_loss, normals, y = self._common_step(batch, batch_idx)
        self.log("val_val_loss", val_loss)
        self.log("val_cos_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        val_loss, cos_loss, normals, y = self._common_step(batch, batch_idx)
        self.log("test_val_loss", val_loss)
        self.log("test_cos_loss", val_loss)
        return val_loss

    def _common_step(self, batch, batch_idx):
        _x = batch.x
        _edge_index = batch.edge_index
        _batch = batch.batch
        normals = self.forward(_x, _edge_index, _batch)
        _y = batch.y
        val_loss = torch_val_loss(normals, _y)
        cos_loss = torch_cos_loss(normals, _y)
        return val_loss, cos_loss, normals, _y

    def predict_step(self, batch, batch_idx):
        _x = batch.x
        _edge_index = batch.edge_index
        _batch = batch.batch
        normals = self.forward(_x, _edge_index, _batch)
        preds = torch_normalize(normals, dim=-1)
        return preds

    def configure_optimizers(self):
        return torch_Adam(self.parameters(), lr=self.config.lr)



class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes, learning_rate):
        super().__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("mnist_images", grid, self.global_step)
        return {"loss": loss, "scores": scores, "y": y}

    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        scores = torch.cat([x["scores"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        self.log_dict(
            {
                "train_loss": avg_loss,
                "train_acc": self.accuracy(scores, y),
                "train_f1": self.f1_score(scores, y),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
