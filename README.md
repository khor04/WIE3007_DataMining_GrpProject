# WIE3007 Data Mining Group Project
## AI-Enhanced Customer Complaint Escalation Prediction Using Machine Learning and Large Language Models

Given a customer complaint at the time it is submitted, we predict whether it will later be
escalated to a supervisor or higher-level support team. We use structured customer and complaint
data plus features that an LLM extracts from the complaint text.

**Target variable `escalation_status`:** 0 = not escalated, 1 = escalated (positive class = 1).

## 1. Team and branches

| Member | Main responsibility | Model | Branch |
|---|---|---|---|
| M1 (Team Leader) | Dataset simulation, cleaning, EDA, GitHub management | Decision Tree | `member1-dataset-decisionTree` |
| M2 | GenAI complaint text, LLM feature extraction, merge AI features into the dataset | Random Forest | `member2-AIFeatures-randomForest` |
| M3 | Preprocessing, class imbalance, feature importance | XGBoost | `member3-xgboost` |
| M4 | Neural network, tuning, combined evaluation visuals | MLP | `member4-mlp` |
| M5 | Logistic regression, interpretation, model comparison, business insights, final report | Logistic Regression | `member5-logisticRegression` |

## 2. Setup (do this once)

1. Clone the repo and open the **repo folder** in VS Code (File > Open Folder).
2. In the VS Code terminal, from the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
nbstripout --install          # clears notebook outputs automatically when you commit
```

3. Open `main.ipynb` and select the `.venv` kernel (top-right of the notebook).
4. Check out your own branch (bottom-left corner of VS Code, or `git checkout <your-branch>`).

## 3. Repo structure

```
.gitignore
.vscode/settings.json          notebook working folder = repo root
main.ipynb                     the one main notebook (submission notebook)
utils.py                       shared split, seed, feature lists, save/load results (M3, M4, M5)
data/
  raw/complaints_raw.csv       raw data with injected missing values and duplicates (M1)
  hidden/generation_inputs.csv hidden factors used ONLY to generate complaint text (M1 -> M2)
results/                       one results file per model, saved by each model section
README.md
requirements.txt
```

The report is written outside the repo.

## 4. Workflow

1. Start each session by pulling the latest `main` into your branch.
2. Work only in **your own sections** of the main notebook (section owners are listed in the notebook).
3. Commit small and often, with a clear message about what you added. At least 6 meaningful commits
   between Weeks 5 and 13, spread out and not all in the last week.
4. Open a pull request from your branch into `main` when a section works.
5. The assigned reviewer reads it and leaves a comment or approval on GitHub.
6. M1 merges using **"Create a merge commit"** (not squash), so every commit stays visible.
7. Keep your branch after merging. Never delete it.

**Review rotation (reviewer → author):** M2 → M1, M3 → M2, M4 → M3, M5 → M4, M1 → M5.


## 5. Notebook rules

- Run **Restart and Run All** before every PR. The notebook must run from top to bottom without errors.
- Do not edit other people's cells. Message the owner instead.
- Outputs are cleared on commit by `nbstripout`. For the final submission commit, M1 runs
  `nbstripout --uninstall`, runs the whole notebook and commits it with outputs.
- Use `#` for main sections and `##` for steps, so the VS Code Outline works.


## 6. Shared technical rules

| Rule | Agreed approach |
|---|---|
| Dataset | 1,500 raw rows, at least 1,000 unique after cleaning; about 19% escalated |
| Train/test split | One fixed split for all models: 80:20, stratified on `escalation_status` |
| Random seed | 42 |
| Features | All structured columns + the five LLM features. Exclude `complaint_id`, `complaint_text` and `escalation_status` |
| Imbalance handling | Agreed strategy (M3), applied to the training data only |
| Tuning | On the training data only (cross-validation). The test set is used once |
| Metrics | Accuracy, precision, recall, F1-score, ROC-AUC (confusion matrix and ROC curve per model) |
| Threshold | Combined charts use the default 0.5. M5 analyses other thresholds separately |
| Evaluation | Shared functions in `utils.py` |

### Results format
Every model section ends by saving its results with the same format:

```python
results = {
    "model_name": "Random Forest",       # Decision Tree, Random Forest, XGBoost, MLP, Logistic Regression
    "test_ids": test_ids,                # complaint_id of each test row
    "y_test": np.array(y_test),
    "y_pred": np.array(y_pred),
    "y_proba": np.array(y_proba),        # probability of escalation only: model.predict_proba(X_test)[:, 1]
    "best_params": best_params,          # optional, useful for the report
}
save_results(results)                    # writes results/<model_name>.pkl
```

M4 loads all files in `results/` and checks that every model used the same test rows before
building the combined ROC curves, metrics table and charts.

## 7. Data files

- `data/complaints_clean.csv` is produced by M1 and then **frozen**. Do not regenerate or edit it.
- M2 adds `complaint_text` and the LLM features, and saves `data/complaints_features.csv`.
- `data/hidden/generation_inputs.csv` holds the hidden severity, urgency and frustration used to write the complaint text. **Never load it in a modelling section.**
- M2 builds the text-generation inputs by merging `data/complaints_clean.csv` with `data/hidden/generation_inputs.csv` on `complaint_id`. The prompts may use the structured fields and the hidden severity, urgency and frustration, but never `escalation_status`.
- Only the owner commits a data or results file. If Source Control shows someone else's file as modified, discard that change.
