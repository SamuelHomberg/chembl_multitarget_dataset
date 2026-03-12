import os
import sys
import argparse
import pandas as pd
import sqlite3
from tqdm import tqdm
import numpy as np
import time
from rdkit import Chem
from rdkit import RDLogger
from joblib import Parallel, delayed
import hashlib
import pickle

def process_database(db_path):
    print(f"Processing chembl database at {db_path}...")
    query = """
SELECT md.chembl_id AS compound_chembl_id,
cs.canonical_smiles,
act.standard_type,
act.standard_value,
act.standard_units,
act.pchembl_value,
td.chembl_id AS target_chembl_id,
td.organism AS target_organism,
td.pref_name AS target_name,
td.target_type,
pc.pref_name AS protein_class,
pc.protein_class_desc,
a.assay_type
FROM target_dictionary td
    JOIN assays a ON td.tid = a.tid
    JOIN activities act ON a.assay_id = act.assay_id
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    JOIN compound_structures cs ON md.molregno = cs.molregno
    JOIN target_components ON td.tid = target_components.tid
    JOIN component_sequences ON target_components.component_id = component_sequences.component_id
    JOIN component_class ON component_sequences.component_id = component_class.component_id
    JOIN protein_classification pc ON component_class.protein_class_id = pc.protein_class_id
WHERE
    act.pchembl_value IS NOT NULL
    AND td.chembl_id <> 'CHEMBL612545';
    -- AND td.target_type = 'SINGLE PROTEIN';
"""

    # Check available memory to decide whether to load DB into RAM
    db_size = os.path.getsize(db_path)
    available_memory = None
    try:
        import psutil
        available_memory = psutil.virtual_memory().available
    except ImportError:
        if os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        available_memory = int(line.split()[1]) * 1024
                        break

    # Load into RAM if we have enough memory (DB size + 4GB buffer)
    if available_memory and available_memory > (db_size + 4 * 1024**3):
        print(f"Loading database into RAM ({db_size / 1024**3:.2f} GB)...")
        source_conn = sqlite3.connect(db_path)
        dest_conn = sqlite3.connect(':memory:')
        source_conn.backup(dest_conn)
        source_conn.close()
        df = pd.read_sql_query(query, dest_conn)
        dest_conn.close()
    else:
        with sqlite3.connect(db_path) as connection:
            df = pd.read_sql_query(query, connection)
    return df

def add_neutral_rdkit_smiles(df, disable_warnings=True):
    print("Neutralizing, desalting and adding rdkit smiles.")
    if 'rdkit_smiles' in df.columns:
        print("Found column `rdkit_smiles`, skipping neutralization and desalting.")
        return df

    def neutralize_atoms(mol):
        # https://www.rdkit.org/docs/Cookbook.html # Neutralizing Molecules
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return mol

    def neutralize_molecule(smiles):
        if disable_warnings:
            RDLogger.DisableLog('rdApp.*')
        try:
            components = Chem.MolFromSmiles(smiles)
            fragments = Chem.GetMolFrags(components, asMols=True, sanitizeFrags=False)
            main_molecule = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
            main_molecule = neutralize_atoms(main_molecule)
            neutral_smiles = Chem.MolToSmiles(main_molecule)
            return neutral_smiles
        except:
            return ''
        
    usmiles, inv_smiles = np.unique(np.array(df["canonical_smiles"], dtype=str),
                                              return_inverse=True)
    rdkitsmiles = Parallel(n_jobs=-2)(delayed(neutralize_molecule)(smi) for smi in usmiles)
    rdkitsmiles = np.array(rdkitsmiles)
    rdkitsmiles = rdkitsmiles[inv_smiles]
    df['rdkit_smiles'] = rdkitsmiles
    print("updating csv at `queried_chembl.csv`")
    df.to_csv('queried_chembl.csv')
    return df

def _hash_string(s):
    """Hash a string to an integer of maxsize 2**64."""
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % 2**64

def find_conflicting_pchembl_values(df, cutoff=6):

    df['rdkit_smiles'] = df['rdkit_smiles'].fillna('')
    pvals =  np.array(df["pchembl_value"], dtype=float)
    target_smiles_pair = np.array([';'.join([tup.target_chembl_id, tup.rdkit_smiles]) for tup in df.itertuples()])
    hashed_ts_pair = np.array([_hash_string(x) for x in target_smiles_pair])

    unique_hpair, counts_hpair = np.unique(hashed_ts_pair, return_counts=True)
    duplicate_hpair = unique_hpair[counts_hpair > 1]

    print("Calculate target smiles pairs.")
    start = time.time()
    tenth_idx = np.linspace(0, len(duplicate_hpair), 10, dtype=int)
    def dup_pvals(i, x):
        if i in tenth_idx: print(f"Assessed {i}/{tenth_idx[-1]} pairs.")
        return pvals[np.where(np.isin(hashed_ts_pair, x))[0]]
    print(f"Finding duplicate target-smiles pairs")
    result = Parallel(n_jobs=-2)(delayed(dup_pvals)(i, x) for i, x in enumerate(duplicate_hpair))
    print(f"Calculated for {time.time() - start:.2f} s.")

    with open('duplicate-target-smiles-pairs.pkl', 'wb') as f:
        pickle.dump(result, f)

    print("Writing target smiles pairs to `duplicate-target-smiles-pairs.pkl`.")

    def is_conflicting(pchembl_arr):
        return min(pchembl_arr) < cutoff and max(pchembl_arr) >= cutoff

    conflicts = np.array([is_conflicting(x) for x in result])
    duplicate_hpair_indices_to_exclude = np.where(conflicts)[0] 

    exclude = duplicate_hpair[duplicate_hpair_indices_to_exclude]
    exclude = np.where(np.isin(hashed_ts_pair, exclude))[0]
    print(f"Found {len(exclude)} contradicting entries to exclude for cutoff {cutoff}.")
    return exclude

def build_complete_dataset_matrix(df, exclude, cutoff = 6):
    print("Building dataset matrix...")
    # Create mask for exclusion (conflicts and empty SMILES)
    mask = np.ones(len(df), dtype=bool)
    if len(exclude) > 0:
        mask[exclude] = False
    
    # Also exclude empty smiles (failed standardizations)
    all_smiles = df['rdkit_smiles'].values.astype(str)
    mask = mask & (all_smiles != '')

    # Filter data to valid entries only
    valid_smiles = all_smiles[mask]
    valid_targets = df['target_chembl_id'].values[mask].astype(str)
    valid_pvals = df['pchembl_value'].values[mask]

    # Identify unique sorted values from VALID data
    unique_rdkit_smiles = np.unique(valid_smiles)
    unique_targets = np.unique(valid_targets)

    # Map values to indices
    smiles_indices = np.searchsorted(unique_rdkit_smiles, valid_smiles)
    target_indices = np.searchsorted(unique_targets, valid_targets)

    # Initialize matrix with -1
    arr = np.full((unique_rdkit_smiles.shape[0], unique_targets.shape[0]), -1, dtype=np.int8)

    # Calculate activity (0 or 1)
    activities = np.where(valid_pvals < cutoff, 0, 1).astype(np.int8)

    # Assign to matrix
    arr[smiles_indices, target_indices] = activities
    
    return arr, unique_rdkit_smiles, unique_targets



def main():
    parser = argparse.ArgumentParser(description="Process the Chembl Database.")
    parser.add_argument('--db', type=str, help='Path to the SQLite chembl database file.')
    parser.add_argument('--csv', type=str, help='Path to the CSV file (if database was already queried).')
    parser.add_argument('--cutoff', type=float, default=6, help='Cutoff value to devide actives from inactives')
    parser.add_argument('--enable-warnings', action='store_false', dest='disable_warnings', help='Enable rdkit warnings.')
    parser.set_defaults(disable_warnings=True)
    
    args = parser.parse_args()

    if args.disable_warnings:
        RDLogger.DisableLog('rdApp.*')
    
    if args.csv:
        if not os.path.isfile(args.csv):
            print(f"Error: CSV file '{args.csv}' does not exist.")
            return
        df = pd.read_csv(args.csv, index_col=0)
        print('saving loaded csv to `queried_chembl.csv`')
        df.to_csv('queried_chembl.csv')
    elif args.db:
        if not os.path.isfile(args.db):
            print(f"Error: Database file '{args.db}' does not exist.")
            return
        if os.path.isfile('queried_chembl.csv'):
            print("Found `queried_chembl.csv`, skipping database querying.")
            df = pd.read_csv('queried_chembl.csv', index_col=0)
        else:
            df = process_database(args.db)
            print('saving to `queried_chembl.csv`')
            df.to_csv('queried_chembl.csv')
    else:
        parser.print_help()
        sys.exit()
    
    df = add_neutral_rdkit_smiles(df, disable_warnings=args.disable_warnings)
    exclude_contradicting_pchembl = find_conflicting_pchembl_values(df, cutoff=args.cutoff)
    complete_matrix, smiles_rows, target_columns = build_complete_dataset_matrix(df, exclude_contradicting_pchembl, cutoff=args.cutoff)
    print(f"Saving `complete_matrix.pkl`")
    with open("complete_matrix.pkl", 'wb') as f:
        pickle.dump((complete_matrix, smiles_rows, target_columns), f)
    print("Done.")
    
if __name__ == "__main__":
    main()