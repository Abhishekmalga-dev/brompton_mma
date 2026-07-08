import pandas as pd

# Replace this with your parquet filename
input_file = "customer_sat.parquet"

# Output Excel filename
output_file = "customer_sat.xlsx"

# Read the parquet file
df = pd.read_parquet(input_file)

# Write to Excel
df.to_excel(output_file, index=False)

print("Conversion completed!")
print(f"Rows: {len(df)}")
print(f"Output file: {output_file}")