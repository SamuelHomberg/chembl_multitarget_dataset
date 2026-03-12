# ChEMBL Multi-Target Dataset Generator

Small python project to process the ChEMBL database (SQLite) and generate a clean, multi-target bioactivity dataset suitable for machine learning tasks:


*   **Chemical Standardization**: Uses RDKit to neutralize and desalt molecules, generating standardized SMILES strings. 
*   **Conflict Resolution**: Identifies compound-target pairs with multiple bioactivity entries. If entries for the same pair conflict (i.e., one is active and another inactive based on the cutoff), they are excluded.
*   **Output**: Pickled matrix of Compounds x Targets `complete_matrix.pkl` containing three arrays:
    *   `matrix`: A numpy array (`int8`) where rows correspond to compounds and columns to targets. Values are `1` (active), `0` (inactive), or `-1` (no data).
    *   `smiles_rows`: Array of unique standardized SMILES corresponding to the matrix rows.
    *   `target_columns`: Array of target ChEMBL IDs corresponding to the matrix columns.


## Create the dataset

```bash
uv run main.py --db /path/to/chembl_XX.db 
```

The pchembl cutoff for activity can be set with the `--cutoff` flag:
```bash
# Note that this will overwrite a previously generated complete_matrix.pkl file.
uv run main.py --db /path/to/chembl_XX.db --cutoff 5 # default is 6
```

If you have already run the script, you can skip re-querying the database by supplying the helper `queried_chembl.csv` file:
```bash
uv run main.py --csv queried_chembl.csv 
```

## Using the dataset

```py
with open('complete_matrix.pkl', 'rb') as f:
    arr, unique_rdkit_smiles, unique_targets = pickle.load(f)
# declare unknown data points as inactive
arr[arr == -1] = 0
```
For more options see [usage.py](usage.py).
