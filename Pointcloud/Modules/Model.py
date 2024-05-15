from . import Config as config

from dataclasses import (
    dataclass
)
from torch import (
    cat as torch_cat,
    long as torch_long,
    minimum as torch_minimum,
    ones as torch_ones,
    tensor as torch_tensor,
    Tensor as torch_Tensor,
    zeros as torch_zeros
)
from torch import cuda as torch_cuda
from torch.nn import (
    BatchNorm1d as torch_BatchNorm1d,
    Dropout as torch_Dropout,
    LeakyReLU as torch_LeakyReLU,
    Linear as torch_Linear,
    Sequential as torch_Sequential
)
from torch.nn.functional import (
    cosine_embedding_loss as torch_cos_loss,
    cosine_similarity as torch_nn_f_cosine_similarity,
    mse_loss as torch_val_loss,
    normalize as torch_normalize
)
from torch.optim import Adam as torch_Adam
from torch_geometric.nn.conv import (
    DynamicEdgeConv as tg_DynamicEdgeConv,
    EdgeConv as tg_EdgeConv
)
from torch_geometric.nn.pool import (
    global_mean_pool as tg_global_mean_pool,
    global_max_pool as tg_global_max_pool
)
import pytorch_lightning as pl
def printGPUStats():
    t = torch_cuda.get_device_properties(0).total_memory
    r = torch_cuda.memory_reserved(0)
    a = torch_cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"t: {t / 1024 ** 3} GB\nr: {r / 1024 ** 2} MB\na: {a / 1024 ** 2} MB\nf: {f / 1024 ** 2} MB")


def custom_val_loss(input: torch_Tensor, target: torch_Tensor) -> torch_Tensor:
    loss1 = (input + target).square().mean(dim=1)
    loss2 = (input - target).square().mean(dim=1)
    return torch_minimum(loss1, loss2).mean(dim=0)

def custom_cos_loss(input: torch_Tensor, target: torch_Tensor) -> torch_Tensor:
    sim = torch_nn_f_cosine_similarity(input, target)
    return torch_minimum(1 - sim, 1 + sim).mean(dim=0)

@dataclass
class NetworkModelConfiguration():
    k: int = config.DYNAMIC_EDGECONV_K
    lr: float = config.LEARNING_RATE
    num_edgeconv: int = config.NUM_EDGECONV
    num_dynamic_edgeconv: int = config.NUM_DYNAMIC_EDGECONV
    num_prepool_linear: int = config.NUM_PREPOOL
    number_postpool_linear_layers: int = config.NUM_POSTPOOL
    in_channels: int = config.INPUT_SIZE
    out_channels: int = config.OUTPUT_SIZE
    feature_channels: torch_Tensor = torch_tensor(config.HIDDEN, dtype=torch_long)
    dropout: float = config.DROPOUT_RATE

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
        _device = x.device
        _config = self.config
        num_convs = _config.num_edgeconv + _config.num_dynamic_edgeconv
        pool_sizes = torch_cat([torch_zeros(1, dtype=torch_long), _config.feature_channels[:num_convs].cumsum(dim=0)])
        x_cat = torch_zeros(x.size(0), pool_sizes[-1], device=_device)
        for i in range(_config.feature_channels.size(0)):
            layer = getattr(self, f"layer{i}")
            if i < _config.num_edgeconv:
                x = layer(x, edge_index)
                x_cat[:, pool_sizes[i]:pool_sizes[i+1]] = x
            elif i < num_convs:
                x = layer(x, batch)
                x_cat[:, pool_sizes[i]:pool_sizes[i+1]] = x
            elif i < num_convs + _config.num_prepool_linear:
                # Set last_x to be x_cat for the first prepool linear layer.
                if i == num_convs:
                    x = x_cat
                x = layer(x)
            else:
                # Do pooling only first time.
                if i == num_convs + _config.num_prepool_linear:
                    x1 = tg_global_max_pool(x, batch)
                    x2 = tg_global_mean_pool(x, batch)
                    print(x.size(), x1.size(), x2.size(), batch.unique(return_counts=True))
                    x = torch_cat((x1, x2), dim=-1)
                printGPUStats()
                x = layer(x)
            print(x.size())
        lastLayer = getattr(self, f"lastLayer")
        return lastLayer(x)
    
    def training_step(self, batch, batch_idx):
        val_loss, cos_loss, c_val_loss, c_cos_loss, normals, _y = self._common_step(batch, batch_idx)
        batch_size = len(batch)
        self.log_dict(
            {
                "train_val_loss": val_loss,
                "train_cos_loss": cos_loss,
                "train_custom_val_loss": c_val_loss,
                "train_custom_cos_loss": c_cos_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size
        )
        # if batch_idx % 100 == 0:
        #     x = x[:8]
        #     grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
        #     self.logger.experiment.add_image("mnist_images", grid, self.global_step)
        return {"loss": c_val_loss, "val_loss": val_loss, "cos_loss": cos_loss, "output": normals, "y": _y}

    def validation_step(self, batch, batch_idx):
        val_loss, cos_loss, c_val_loss, c_cos_loss, normals, y = self._common_step(batch, batch_idx)
        batch_size = len(batch)
        self.log_dict(
            {
                "val_val_loss": val_loss,
                "val_cos_loss": cos_loss,
                "val_custom_val_loss": c_val_loss,
                "val_custom_cos_loss": c_cos_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size
        )
        return val_loss

    def test_step(self, batch, batch_idx):
        val_loss, cos_loss, c_val_loss, c_cos_loss, normals, y = self._common_step(batch, batch_idx)
        batch_size = len(batch)
        self.log_dict(
            {
                "test_val_loss": val_loss,
                "test_cos_loss": cos_loss,
                "test_custom_val_loss": c_val_loss,
                "test_custom_cos_loss": c_cos_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size
        )
        return val_loss

    def _common_step(self, batch, batch_idx):
        print(f"Do common step! --> X Size: {batch.x.size()}")
        _x = batch.x
        _edge_index = batch.edge_index
        print(f"batch.batch: {batch.batch.unique(return_counts=True)}\nbatch_idx: {batch_idx}")
        _batch = batch.batch
        normals = self.forward(_x, _edge_index, _batch)
        _y = batch.y
        val_loss = torch_val_loss(normals, _y.repeat(normals.size(0), 1))
        cos_loss = torch_cos_loss(normals, _y, torch_ones((_y.size(0),), device=_y.device))
        c_val_loss = custom_val_loss(normals, _y)
        c_cos_loss = custom_cos_loss(normals, _y)
        return val_loss, cos_loss, c_val_loss, c_cos_loss, normals, _y

    def predict_step(self, batch, batch_idx):
        _x = batch.x
        _edge_index = batch.edge_index
        _batch = batch.batch
        normals = self.forward(_x, _edge_index, _batch)
        preds = torch_normalize(normals, dim=-1)
        return preds

    def configure_optimizers(self):
        return torch_Adam(self.parameters(), lr=self.config.lr)

# class NN(pl.LightningModule):
#     def __init__(self, input_size, num_classes, learning_rate):
#         super().__init__()
#         self.lr = learning_rate
#         self.fc1 = nn.Linear(input_size, 50)
#         self.fc2 = nn.Linear(50, num_classes)
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.accuracy = torchmetrics.Accuracy(
#             task="multiclass", num_classes=num_classes
#         )
#         self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return x

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         loss, scores, y = self._common_step(batch, batch_idx)
#         self.log_dict(
#             {
#                 "train_loss": loss,
#             },
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#         )
#         if batch_idx % 100 == 0:
#             x = x[:8]
#             grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
#             self.logger.experiment.add_image("mnist_images", grid, self.global_step)
#         return {"loss": loss, "scores": scores, "y": y}

#     def train_epoch_end(self, outputs):
#         avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
#         scores = torch.cat([x["scores"] for x in outputs])
#         y = torch.cat([x["y"] for x in outputs])
#         self.log_dict(
#             {
#                 "train_loss": avg_loss,
#                 "train_acc": self.accuracy(scores, y),
#                 "train_f1": self.f1_score(scores, y),
#             },
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#         )

#     def validation_step(self, batch, batch_idx):
#         loss, scores, y = self._common_step(batch, batch_idx)
#         self.log("val_loss", loss)
#         return loss

#     def test_step(self, batch, batch_idx):
#         loss, scores, y = self._common_step(batch, batch_idx)
#         self.log("test_loss", loss)
#         return loss

#     def _common_step(self, batch, batch_idx):
#         x, y = batch
#         x = x.reshape(x.size(0), -1)
#         scores = self.forward(x)
#         loss = self.loss_fn(scores, y)
#         return loss, scores, y

#     def predict_step(self, batch, batch_idx):
#         x, y = batch
#         x = x.reshape(x.size(0), -1)
#         scores = self.forward(x)
#         preds = torch.argmax(scores, dim=1)
#         return preds

#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.lr)
