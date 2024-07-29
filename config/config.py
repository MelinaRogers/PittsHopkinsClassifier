import os

# Data paths
DATA_DIR = os.path.join(os.getcwd(), 'output')
X_FEATURE_PATH = os.path.join(DATA_DIR, 'X_feature_f.npy')
Y_TARGET_PATH = os.path.join(DATA_DIR, 'y_target_c.npy')
FEATURE_NAMES_PATH = os.path.join(DATA_DIR, 'feature_names_c.json')

# Model parameters
HIDDEN_DIMS = [64, 128, 256]
DROPOUT_RATES = [0.3, 0.5, 0.7]
NUM_FEATURES = 100
RANDOM_STATE = 42

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 100
TEST_SIZE = 0.2

# HPO terms
KEY_HPO_TERMS = [
    "HP:0000750",  # Delayed speech and language development
    "HP:0002079",  # Breathing abnormality
    "HP:0000322",  # Short stature
    "HP:0004322",  # Short philtrum
    "HP:0010509",  # Severe intellectual disability
    "HP:0008082"   # Abnormal gait
]