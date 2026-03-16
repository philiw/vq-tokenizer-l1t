import pyarrow.parquet as pq
import glob

DATA_DIR = "/eos/project/f/foundational-model-dataset/samples/production_final/VBFHbb/"
TARGET_COLS = {"L1T_JetPuppiAK8_PT", "L1T_JetPuppiAK8_Eta", "L1T_JetPuppiAK8_Phi"}

files = sorted(glob.glob(DATA_DIR + "*.parquet"))
print(f"Found {len(files)} parquet files")

total_bytes = 0
for path in files:
    file_bytes = 0
    meta = pq.read_metadata(path)
    for rg_idx in range(meta.num_row_groups):
        rg = meta.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            if col.path_in_schema in TARGET_COLS:
                file_bytes += col.total_compressed_size
    total_bytes += file_bytes
    print(f"  {path.split('/')[-1]}: {file_bytes / 1e6:.2f} MB")

print(f"\nTotal compressed on-disk size for 3 columns: {total_bytes / 1e9:.4f} GB ({total_bytes / 1e6:.2f} MB)")
