from torch import (
    long as torch_long,
    tensor as torch_tensor
)

# Model Hyperparameters
DYNAMIC_EDGECONV_K = 8
LEARNING_RATE = 0.001
NUM_EDGECONV = 6
NUM_DYNAMIC_EDGECONV = 0 
NUM_PREPOOL = 1
NUM_POSTPOOL = 3
INPUT_SIZE = 8
OUTPUT_SIZE = 3
DROPOUT_RATE = 0.5
HIDDEN = torch_tensor([64, 64, 128, 256, 256, 256, 512, 256, 64], dtype=torch_long)
# HIDDEN = torch_tensor([64, 128, 256, 512, 256, 64], dtype=torch_long)

# Training Hyperparameters
BATCH_SIZE = 64
MIN_EPOCHS = 20
NUM_EPOCHS = 100
MODEL_NAME = "testmodel"
MONITOR_LOSS = "val_custom_val_loss"

# Dataset
DATA_DIR = "PatchDataset"
PROCESS_ACCELERATOR = "cuda"
SPLIT_NAME = "Super_Cool_Split_Bro"
SPLIT = (0.6, 0.2, 0.2)
# GAUSSIAN_NOISE_LEVELS = [0.1, 0.2, 0.3]
# IMPULSIVE_NOISE_LEVELS = [0.1, 0.2, 0.3]
GAUSSIAN_NOISE_LEVELS = [0.01, 0.02, 0.03]
IMPULSIVE_NOISE_LEVELS = [0.01, 0.02, 0.03]
# NUM_WORKERS should be 4 * Num_GPU as a general rule.
# https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
NUM_WORKERS = 4

# Logs
LOG_DIR = "tb_logs"

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
# ACCELERATOR = "cpu"
# DEVICES = "auto"
PRECISION = "16-mixed"

# Preprocessing
K_PATCH_RADIUS = 4