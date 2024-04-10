# Training Hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_EPOCHS = 1

# Dataset
DATA_DIR = "PatchDataset"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "cuda"
DEVICES = [0]
PRECISION = "16-mixed"

# Preprocessing
K_PATCH_RADIUS = 4