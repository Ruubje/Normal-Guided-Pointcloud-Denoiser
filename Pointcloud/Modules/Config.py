# Model Hyperparameters
DYNAMIC_EDGECONV_K = 8
LEARNING_RATE = 0.001
NUM_EDGECONV = 3
NUM_DYNAMIC_EDGECONV = 0 # 3
NUM_PREPOOL = 1
NUM_POSTPOOL = 3
INPUT_SIZE = 8
OUTPUT_SIZE = 3
DROPOUT_RATE = 0.5
# HIDDEN = [64, 64, 128, 256, 256, 256, 512, 256, 64]
HIDDEN = [64, 128, 256, 512, 256, 64]

# Training Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
MODEL_NAME = "testmodel"

# Dataset
DATA_DIR = "PatchDataset"
SPLIT_NAME = "Super_Cool_Split_Bro"
SPLIT = (0.6, 0.2, 0.2)
# NUM_WORKERS should be 4 * Num_GPU as a general rule.
# https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
NUM_WORKERS = 4

# Logs
LOG_DIR = "tb_logs"

# Compute related
ACCELERATOR = "cuda"
DEVICES = [0]
PRECISION = "16-mixed"

# Preprocessing
K_PATCH_RADIUS = 4