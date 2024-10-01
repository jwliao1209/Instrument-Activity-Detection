PROJECT_NAME = 'DeepMIR_HW1'
DATA_DIR = 'dataset'
RESULT_DIR = 'results'
CHECKPOINT_DIR = 'checkpoints'
CONFIG_FILE = 'config.json'
CKPT_FILE = 'checkpoint.pth'

CATEGORIES = [
    'Piano',
    'Percussion',
    'Organ',
    'Guitar',
    'Bass', 
    'Strings',
    'Voice',
    'Wind Instruments',
    'Synth',
]
SAMPLE_RATE = 24000
NUM_CLASSES = len(CATEGORIES)
# THRERSHOLDS = [0.5, 0.08, 0.2, 0.5, 0.5, 0.5, 0.3, 0.5, 0.3]
THRERSHOLDS = [0.5] * NUM_CLASSES
