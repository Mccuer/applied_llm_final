import pandas as pd

df = pd.read_parquet(r"C:\Users\robbi\Downloads\testfinalllm\train-00000-of-00206.parquet", engine = "fastparquet")

first_entry_code = df.iloc[0]["content"]
second_entry_code = df.iloc[100]["content"]
third_entry_code = df.iloc[1000]["content"]

print(df)
print(first_entry_code)
print(second_entry_code)
print(third_entry_code)