import os
import random 
import glob
import shutil

PATH_TO_JSONS = "/scratch/reproduce_scenario_dreamer/gpudrive_training_set_jsons"
NUM_SCENARIOS = 10

PATH_TO_MINI_DATASETS = f"data/processed/scenario_dreamer_{NUM_SCENARIOS}_scenarios"

os.makedirs(PATH_TO_MINI_DATASETS, exist_ok=True)

json_fnames = glob.glob(os.path.join(PATH_TO_JSONS, "*.json"))
random.shuffle(json_fnames)
json_fnames = json_fnames[:NUM_SCENARIOS]

# Copy selected files to mini datasets directory
for json_fname in json_fnames:
    dest_path = os.path.join(PATH_TO_MINI_DATASETS, os.path.basename(json_fname))
    shutil.copy(json_fname, dest_path)
    print(f"Copied {os.path.basename(json_fname)} to {PATH_TO_MINI_DATASETS}")

print(f"\nSuccessfully created mini dataset with {len(json_fnames)} scenarios in {PATH_TO_MINI_DATASETS}")
