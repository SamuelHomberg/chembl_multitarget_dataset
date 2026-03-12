# %%
import pickle, numpy as np
path = 'complete_matrix.pkl'
with open(path, 'rb') as f:
    arr, unique_rdkit_smiles, unique_targets = pickle.load(f)
arr.shape
# assume no data is inactive
arr[arr == -1] = 0
# %%
# declare unknown data points as inactive
arr[arr == -1] = 0

# actives per target
tarr = np.sum(arr, 0)
print(f"actives per target: {tarr}, shape: {tarr.shape}")

# indices of targets with > x actives
min_actives_per_target = 150
tidx = np.where(tarr > min_actives_per_target)[0]
# filter for this
arr = arr[:,tidx]
unique_targets = unique_targets[tidx]
assert arr.shape[1] == unique_targets.shape[0]

# %%
# get some stats on smiles
print("SMILES stats")
act = np.sum(arr, 1)
print(f"{arr.shape} SMILES / targets")
print(f"{np.mean(act):>9.4f} mean num targets a smiles is active on")
print(f"{np.std(act):>9.4f} std" )
print(f"{np.max(act):>9} max")
print(f"{np.min(act):>9} min")  
print()
print(f"{sum(np.isin(act, 0)):>9} true negatives (active on zero targets)") # true negatives
print(f"{sum(np.isin(act, 1)):>9} true positives (active on one target)") # true positives on one target
print(f"{sum(np.isin(act, 2)):>9} true positives (active on two targets)") # true positives on two target
print(f"{sum(np.isin(act, 3)):>9} true positives (active on three targets)") # true positives on three target
print(f"{sum(np.isin(act, 4)):>9} true positives (active on four targets)") # true positives on four target
print(f"{sum(~np.isin(act, 0)):>9} total true positives")# true positives
print(f"{np.sum(act):>9} total actives (on any number of target)") # total activities 
