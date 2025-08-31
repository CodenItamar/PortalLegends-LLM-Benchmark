# Portal Legends Move Classification

This project tests whether Large Language Models (LLMs) can infer rule-based reasoning from text alone, by classifying Portal Legends moves as **legal** or **illegal** from natural-language descriptions.

---

## Project Structure (minimal)

```
ANLP-Project-Pipeline/
├── configs/                  # Model configs (BERT / DistilBERT / RoBERTa)
├── data/                     # CSV data (see Setup)
├── models/                   # Trained model checkpoints (created by train.py)
├── results/                  # Evaluation reports & predictions
├── src/                      # Code (trainer, predict, registry, etc.)
├── train.py                  # Train all three models
├── predict.py                # Run predictions / reports
└── README.md
```

---

## Setup

1. **Install dependencies**
```bash
pip install torch transformers pandas scikit-learn
```

2. **Dataset**:
Delete any pre-split files from `data/`:
  - `data/train_data.csv`  
  - `data/val_data.csv`  
  - `data/test_data.csv`  

The training script will (re)create the splits automatically.

---

## How to Run

### 1) Train all three models
Runs BERT, DistilBERT, and RoBERTa end-to-end (train → validate → save best).
```bash
python train.py
```
- Checkpoints and hyperparam files are saved in `models/`.
- Logs/metrics are saved in `results/`.

### 2) Predict
Run the prediction script and choose **option 1** to generate predictions from **all** trained models.
```bash
python predict.py
# When prompted in the console, type:
1
```
The script will scan `models/` for the latest checkpoints and produce per-model predictions and a comparison summary in `results/`.


---

## Tips & Troubleshooting

- **“No trained models found”**  
  Ensure you ran `python train.py` and that `models/` contains files like `bert_model_*.pth` and matching `*_hyperparams_*.json`.  
  If you run from an IDE, confirm the **working directory** is the project root.

- **Windows path issues (slashes)**  
  If you edited code, prefer `pathlib.Path` over raw strings and normalize paths when matching files.

- **Reproducibility**  
  The split is recreated when the pre-split CSVs are absent. To keep a split fixed, do **not** delete `train_data.csv` / `val_data.csv` / `test_data.csv`.

---

## What the Scripts Do

- **`train.py`**
  1. Loads `data/PortalLegendsMovesTagged2.csv`
  2. Creates train/val/test splits (if not already present)
  3. Tokenizes (max length 128)
  4. Trains with AdamW (`lr=2e-5`, batch size 8, early stopping)
  5. Saves best checkpoint + hyperparams to `models/`
  6. Writes metrics to `results/`

- **`predict.py`**
  1. Lists available checkpoints in `models/`
  2. On **option 1**, runs predictions with **all** models
  3. Saves per-model outputs + a comparison table to `results/`

---

## Extending (optional)

- To add a new model, copy `configs/new_model_template.py`, implement your config, and register it in `src/model_registry.py`.

---