# Try reading the file with automatic delimiter detection
import pandas as pd
import csv

file_path = "/Users/evanquan/Downloads/Blake/FullPilotBlake_BL/FullPilotBlake_BL_forces_2025_01_17_120728.csv"  # path to your .csv

# Detect delimiter automatically
with open(file_path, 'r', newline='', encoding='utf-8') as f:
    dialect = csv.Sniffer().sniff(f.read(2048))
    f.seek(0)
    delimiter = dialect.delimiter
    print(f"Detected delimiter: '{delimiter}'")
    df = pd.read_csv(f, delimiter=delimiter)

print("Columns detected:", df.columns.tolist())
print(df.head())
