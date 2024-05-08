import torch
import pytorch_lightning as pl

from .Object import Pointcloud
from . import Config as config
from .Alignor import Alignor
from .Callbacks import (
    MyPrintingCallback,
    EarlyStopping,
    ModelCheckpoint
)
from .FileDataset import FileDataset
from .Model import Patch2NormalModel
from .Preprocessor import Preprocessor

from pathlib import Path

from torch_geometric.data import (
    Batch as tg_data_Batch,
    DataLoader as tg_data_DataLoader
)

from torch_geometric.profile import get_data_size as tg_profile_get_data_size

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

def getModelSize(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return param_size + buffer_size

class Manager:
    def __init__(self):
        torch.set_float32_matmul_precision("medium")

        logger = TensorBoardLogger(config.LOG_DIR, name=config.MODEL_NAME)
        profiler = PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                Path(config.LOG_DIR) / f"{config.MODEL_NAME}_profilers" / f"{logger.version}"
            ),
            trace_memory=True,
            schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
        )

        self.model = Patch2NormalModel()

        dm = FileDataset(config.DATA_DIR, split_name=config.SPLIT_NAME, split=config.SPLIT)
        self.train_dl = dm.train_dataloader(config.BATCH_SIZE, config.NUM_WORKERS)
        self.val_dl = dm.val_dataloader(config.BATCH_SIZE, config.NUM_WORKERS)
        self.test_dl = dm.test_dataloader(config.BATCH_SIZE, config.NUM_WORKERS)

        monitor_loss = "val_custom_val_loss"
        self.trainer = pl.Trainer(
            profiler=profiler,
            logger=logger,
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            min_epochs=1,
            max_epochs=config.NUM_EPOCHS,
            precision=config.PRECISION,
            callbacks=[
                ModelCheckpoint(
                    monitor=monitor_loss,
                    filename='{config.MODEL_NAME}-epoch{epoch:02d}-{monitor_loss}{val_val_loss:.2f}',
                    auto_insert_metric_name=False,
                    save_top_k=5,
                ),
                MyPrintingCallback(),
                EarlyStopping(monitor=monitor_loss, patience=10)
            ],
        )
        torch.cuda.empty_cache()
        ms = getModelSize(self.model) / 1024 ** 2
        tds = sum([x.numel() * x.element_size() for x in self.train_dl.dataset.to_dict().values()]) / 1024 ** 2
        print(f"Model size: {ms} MB\nTrain Dataset size: {tds} MB")

    def assertCheckpointFile(self, checkpoint: str):
        if checkpoint is not None:
            path = Path(checkpoint)
            assert path.exists() and path.suffix == ".ckpt", f"Exists: {path.exists()}\nsuffix: {path.suffix}\nFull path: {str(path)}"
    
    def train(self, from_checkpoint: str=None):
        self.assertCheckpointFile(from_checkpoint)
        self.trainer.fit(self.model, self.train_dl, self.val_dl, ckpt_path=from_checkpoint)
    
    def validate(self, from_checkpoint: str=None):
        self.assertCheckpointFile(from_checkpoint)
        self.trainer.validate(self.model, self.val_dl, ckpt_path=from_checkpoint)

    def test(self, from_checkpoint: str=None):
        self.assertCheckpointFile(from_checkpoint)
        self.trainer.test(self.model, self.test_dl, ckpt_path=from_checkpoint)

    def predict(self, preprocessor: Preprocessor, from_checkpoint: str=None):
        self.assertCheckpointFile(from_checkpoint)
        graphs, RInv = preprocessor.getPatches()
        ds = tg_data_Batch.from_data_list(graphs)
        dl = tg_data_DataLoader(
            dataset=ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            persistent_workers=True,
        )
        batchedOutput = self.trainer.predict(self.model, dl, ckpt_path=from_checkpoint)
        output = torch.cat(batchedOutput, dim=0)
        predictions = Alignor.applyRInv(RInv, output)
        return predictions

if __name__ == "__main__":
    manager = Manager()
    manager.train()
