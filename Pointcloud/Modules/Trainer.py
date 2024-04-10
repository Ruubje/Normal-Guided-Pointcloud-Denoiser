import torch
import pytorch_lightning as pl
import Config as config
from .Callbacks import MyPrintingCallback, EarlyStopping
from .FileDataset import FileDataset
from .Model import Patch2NormalModel
from torch_geometric.data.lightning import LightningDataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import (
    PyTorchProfiler,
    SimpleProfiler
)


def main():
    torch.set_float32_matmul_precision("medium")

    logger = TensorBoardLogger("tb_logs", name="patchmodel_v0")
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    #     trace_memory=True,
    #     schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    # )
    profiler = SimpleProfiler()

    model = Patch2NormalModel()
    dm = FileDataset(config.DATA_DIR)

    trainer = pl.Trainer(
        profiler=profiler,
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_val_loss")],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
