# COL-8628 Assignment: Prompt Learning and Semi-Supervised Learning

This repository contains the implementation for the COL-8628 assignment focusing on prompt learning techniques (CoOp, CoCoOp) and semi-supervised learning with G-DINO models.

## Project Structure

```
COL-8628/
├── scripts/
│   ├── task1/
│   │   └── task1.py                 # Zero-shot evaluation
│   ├── task2/
│   │   ├── train_task2_1.py         # CoOp training
│   │   ├── test_task2_1.py          # CoOp testing
│   │   ├── train_task2_2.py         # CoCoOp training
│   │   └── test_task2_2.py          # CoCoOp testing
│   └── task3/
│       ├── train_task3.py           # Semi-supervised CoOp training
│       └── test_task3.py            # Semi-supervised CoOp testing
├── models/                          # Saved model weights
├── data/                           # Dataset storage
├── utils/                          # Utility functions
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup Instructions

### 1. Environment Setup

Create a Python virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Data Preparation

Place your datasets in the `data/` directory with the following structure:
```
data/
├── train.json              # Training dataset
├── test.json               # Test dataset
├── labeled.json            # Fully labeled dataset (Task 3)
└── unlabeled.json          # Unlabeled dataset (Task 3)
```

## Usage Instructions

### Task 1: Zero-Shot Evaluation

Evaluate a model using zero-shot prompting on a test dataset.

```bash
python scripts/task1/task1.py --test_path data/test.json --prompt "A photo of a {}" --output_dir results/zero_shot/
```

**Parameters:**
- `--test_path`: Path to test dataset (required)
- `--prompt`: Text prompt template for evaluation (required)
- `--model_path`: Path to pre-trained model weights (optional)
- `--output_dir`: Directory to save results (default: results/)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--device`: Device to use (default: cuda)

### Task 2: Supervised Prompt Learning

#### Task 2.1: CoOp (Context Optimization)

**Training:**
```bash
python scripts/task2/train_task2_1.py --train_path data/train.json --save_path models/coop_model.pth --epochs 50 --lr 0.002
```

**Testing:**
```bash
python scripts/task2/test_task2_1.py --test_path data/test.json --model_path models/coop_model.pth --output_dir results/coop/
```

#### Task 2.2: CoCoOp (Conditional Context Optimization)

**Training:**
```bash
python scripts/task2/train_task2_2.py --train_path data/train.json --save_path models/cocoop_model.pth --epochs 50 --lr 0.002 --meta_net_hidden_dim 128
```

**Testing:**
```bash
python scripts/task2/test_task2_2.py --test_path data/test.json --model_path models/cocoop_model.pth --output_dir results/cocoop/
```

**Training Parameters:**
- `--train_path`: Path to training dataset (required)
- `--save_path`: Path to save trained model (required)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.002)
- `--batch_size`: Batch size (default: 32)
- `--context_length`: Length of context vectors (default: 16)
- `--device`: Device to use (default: cuda)
- `--seed`: Random seed (default: 42)
- `--meta_net_hidden_dim`: Hidden dimension for meta-network (CoCoOp only, default: 128)

**Testing Parameters:**
- `--test_path`: Path to test dataset (required)
- `--model_path`: Path to trained model (required)
- `--output_dir`: Directory to save results (default: results/method/)
- `--batch_size`: Batch size (default: 32)
- `--device`: Device to use (default: cuda)

### Task 3: Semi-Supervised CoOp Prompt Tuning

**Training:**
```bash
python scripts/task3/train_task3.py --labeled_path data/labeled.json --unlabeled_path data/unlabeled.json --save_path models/semi_coop_model.pth --epochs 100 --lambda_u 1.0 --threshold 0.95
```

**Testing:**
```bash
python scripts/task3/test_task3.py --test_path data/test.json --model_path models/semi_coop_model.pth --output_dir results/semi_coop/
```

**Training Parameters:**
- `--labeled_path`: Path to fully labeled dataset (required)
- `--unlabeled_path`: Path to unlabeled dataset (required)
- `--save_path`: Path to save trained model (required)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.002)
- `--batch_size`: Batch size (default: 32)
- `--context_length`: Length of context vectors (default: 16)
- `--device`: Device to use (default: cuda)
- `--seed`: Random seed (default: 42)
- `--lambda_u`: Weight for unlabeled loss (default: 1.0)
- `--threshold`: Confidence threshold for pseudo-labeling (default: 0.95)
- `--ema_decay`: EMA decay rate for teacher model (default: 0.999)

## Output Files

Each script generates the following outputs:

### Evaluation Results
- `metrics.json`: Evaluation metrics (accuracy, precision, recall, F1-score)
- `predictions.json`: Model predictions on test data

### Model Weights
- Trained model weights are saved in the `models/` directory
- Only learned prompts and one copy of G-DINO weights are included

## Implementation Notes

### Key Features Implemented:
1. **Zero-shot evaluation** with customizable text prompts
2. **CoOp training** with learnable context optimization
3. **CoCoOp training** with conditional context generation
4. **Semi-supervised learning** using FixMatch-style consistency training
5. **Comprehensive evaluation** with multiple metrics

### External Dependencies:
- **G-DINO**: Used as the base vision-language model
- **CLIP**: For vision-language understanding
- **MMDetection**: For G-DINO implementation
- **PyTorch**: Deep learning framework

### Data Format:
The scripts expect JSON files with the following structure:
```json
{
  "images": [
    {
      "id": "image_001",
      "path": "path/to/image.jpg",
      "label": "class_name",
      "bbox": [x, y, width, height]  // optional
    }
  ],
  "classes": ["class1", "class2", "class3"]
}
```

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size using `--batch_size`
2. **Missing dependencies**: Ensure all packages in `requirements.txt` are installed
3. **Data loading errors**: Check dataset paths and format

### Performance Tips:

1. Use mixed precision training for faster convergence
2. Adjust learning rate based on dataset size
3. Monitor training logs for convergence

## References

- CoOp: Learning to Prompt for Vision-Language Models
- CoCoOp: Conditional Prompt Learning for Vision-Language Models
- FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence
- G-DINO: Grounding DINO for Open-Set Object Detection

## Contact

For questions regarding implementation details, please refer to the assignment documentation or contact the course instructors.