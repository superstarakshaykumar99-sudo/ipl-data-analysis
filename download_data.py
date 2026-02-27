import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("patrickb1912/ipl-complete-dataset-20082020")
print("Path to dataset files:", path)

# Copy CSVs into the project's data/ directory
dest = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(dest, exist_ok=True)

for fname in os.listdir(path):
    if fname.endswith(".csv"):
        src = os.path.join(path, fname)
        dst = os.path.join(dest, fname)
        shutil.copy(src, dst)
        print(f"Copied {fname} → data/")

print("Done! CSVs are ready in data/")
