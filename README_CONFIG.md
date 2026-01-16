# Configuration System
{
  "paths": {
    "base_dir": "/content/",
    "rgb_frames_dir": "/content/preprocessing",
    "landmarks_dir": "/content/landmarks",
    "checkpoint_dir": "/content/",
    "best_model_path": "/content/checkpoints\\best_model.pth"
  }
}

## Overview

The configuration system has been refactored to separate **paths** and **hyperparameters**:

- **config.json**: Stores all file paths and directory locations
- **args_config.py**: Provides `get_args()` function using argument parser for hyperparameters

## Files

### config.json

JSON file containing only path configurations:

```json
{
  "paths": {
    "base_dir": "path/to/base",
    "rgb_frames_dir": "path/to/rgb/frames",
    "landmarks_dir": "path/to/landmarks",
    "checkpoint_dir": "path/to/checkpoints",
    "best_model_path": "path/to/best_model.pth"
  }
}
```

### args_config.py

Python module with two main functions:

1. **`load_paths(config_path='config.json')`**: Loads paths from config.json
2. **`get_args()`**: Parses command-line arguments for all hyperparameters

## Usage

### Training

```python
from args_config import get_args, load_paths

# Parse arguments
args = get_args()
paths = load_paths()

# Use in your code
print(args.batch_size)  # hyperparameter
print(paths['base_dir'])  # path
```

### Command Line

You can override any hyperparameter from command line:

```bash
# Use default values
python train.py

# Override specific parameters
python train.py --batch-size 4 --learning-rate 0.0001 --num-epochs 50

# Disable augmentation
python train.py --no-train-augment

# Use different optimizer
python train.py --optimizer sgd --momentum 0.95

# Change model architecture
python train.py --num-transformer-layers 4 --num-heads 16
```

### Available Arguments

#### Dataset Settings
- `--num-classes`: Number of sign language classes (default: 100)
- `--img-size`: Input image size (default: 224)
- `--num-frames`: Number of frames per video sequence (default: 8)

#### Model Architecture
- `--rgb-feature-dim`: RGB feature dimension (default: 512)
- `--pose-feature-dim`: Pose feature dimension (default: 256)
- `--fusion-dim`: Fusion layer dimension (default: 512)
- `--num-heads`: Number of attention heads (default: 8)
- `--num-transformer-layers`: Number of transformer layers (default: 2)
- `--dropout`: Dropout rate (default: 0.1)

#### Training Settings
- `--batch-size`: Batch size for training (default: 2)
- `--num-epochs`: Number of training epochs (default: 100)
- `--learning-rate`, `--lr`: Learning rate (default: 0.0001)
- `--weight-decay`: Weight decay (default: 0.00001)
- `--num-workers`: Number of data loader workers (default: 4)

#### Optimizer
- `--optimizer`: Optimizer type ('adam' or 'sgd', default: 'adam')
- `--momentum`: Momentum for SGD optimizer (default: 0.9)

#### Learning Rate Scheduler
- `--lr-scheduler`: LR scheduler type ('cosine', 'step', or 'none', default: 'cosine')
- `--lr-step-size`: Step size for StepLR scheduler (default: 30)
- `--lr-gamma`: Gamma for StepLR scheduler (default: 0.1)

#### Data Augmentation
- `--train-augment` / `--no-train-augment`: Enable/disable training augmentation (default: enabled)
- `--horizontal-flip-prob`: Probability of horizontal flip (default: 0.5)
- `--color-jitter` / `--no-color-jitter`: Enable/disable color jitter (default: enabled)
- `--temporal-augment` / `--no-temporal-augment`: Enable/disable temporal augmentation (default: enabled)

#### Other
- `--early-stopping-patience`: Patience for early stopping (default: 15)
- `--log-interval`: Log every N batches (default: 10)
- `--val-interval`: Validate every N epochs (default: 1)
- `--save-interval`: Save checkpoint every N epochs (default: 5)
- `--device`: Device to use ('cuda' or 'cpu', default: 'cuda')
- `--seed`: Random seed for reproducibility (default: 42)

## Migration from Old Config

The old `config.py` file has been replaced. If you need to update paths, edit `config.json`. If you need to change hyperparameters, use command-line arguments or modify the defaults in `args_config.py`.

### Old Way (config.py)
```python
from config import Config

config = Config()
print(config.BATCH_SIZE)
print(config.RGB_FRAMES_DIR)
```

### New Way (args_config.py + config.json)
```python
from args_config import get_args, load_paths

args = get_args()
paths = load_paths()

print(args.batch_size)
print(paths['rgb_frames_dir'])
```

## Benefits

1. **Separation of Concerns**: Paths and hyperparameters are stored separately
2. **Easy Configuration**: JSON format for paths is easy to edit
3. **Command-Line Flexibility**: Override any hyperparameter without editing code
4. **Version Control Friendly**: JSON paths can be easily templated or ignored
5. **Better Documentation**: All arguments are documented with help text
