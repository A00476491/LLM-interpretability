# Project File Descriptions

## 📦 Pre-generated Data &  Model

All required datasets and trained models can be downloaded from the following Google Drive link:  
🔗 [Download here](https://drive.google.com/drive/folders/1kxUiiE06ZV_dKm-MTSR7QBiIf1BaSSVH?usp=drive_link)

### `dataset.py`
- **Purpose**: Generates a dataset related to the topic of transportation using prompt and hook techniques.
- **Output**: `./data/dataset.json`
- **Note**: Dataset generation is time-consuming. You can use the pre-generated version via the provided link.

---

### `build_token_feature_table.py`
- **Purpose**: Constructs feature vectors based for each token.
- **Output**: `./data/token_feature_table.json`
- **Requirements**: Requires a SAE model and `dataset.json`.
- **Run**:
  ```bash
  python build_token_feature_table.py
  ```
- **Note**: A pre-generated version is available via the provided link.

---

### `intervention.py`
- **Purpose**: Here are two cases for intervention. They modifie the activations of llm to observe how these interventions affect the llm outputs, helping to explore internal mechanisms.
- **Requirements**: Requires `token_feature_table.json` and a SAE model.
- **Run**:
  ```bash
  python intervention.py
  ```

---

### `model.py`
- **Purpose**: Defines the architecture of the sparse autoencoder.

---

### `train.py`
- **Purpose**: Handles model training, including hyperparameter setup, data loading, model initialization, and training loop execution.
- **Output**: Trained model and logs saved in `./model/`.
- **Requirements**: Requires `dataset.json`.
- **Run**: Refer to `exp_train.bat` for example usage.
- **Note**: A pre-trained model is available via the provided link.

