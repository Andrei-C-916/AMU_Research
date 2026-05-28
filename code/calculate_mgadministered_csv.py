import pandas as pd
import json

# Load the CSV file
df = pd.read_csv('data/single_and_multi_cols_2021_labels.csv')

# Result list to store processed records
result = []

# Process each row
for index, row in df.iterrows():
    # Check if required base columns exist
    if pd.isna(row.get('code')) or pd.isna(row.get('BWkg')):
        continue
    
    record = {'code': row['code']}
    
    # Process up to 7 drugs
    for i in range(1, 8):
        # Define column names based on index
        drug_col = f'ABdrug{i}'
        
        if i == 1:
            # First drug uses Dosemgkg and Durationd (no suffix)
            dose_col = 'Dosemgkg'
            duration_col = 'Durationd'
        else:
            # Subsequent drugs use numbered suffixes (i-1)
            dose_col = f'Dosemgkg{i-1}'
            duration_col = f'Durationd{i-1}'
        
        # Check if drug exists and all required values are present
        if (drug_col in row and not pd.isna(row[drug_col]) and
            dose_col in row and not pd.isna(row[dose_col]) and
            duration_col in row and not pd.isna(row[duration_col])):

            # Calculate total mg administered
            try:
                total_mg = float(row['BWkg']) * float(row[dose_col]) * float(row[duration_col])
            except ValueError:
                continue

            # Add to record
            record[drug_col] = row[drug_col]
            record[f'totalmgadministered{i}'] = total_mg
    
    # Only add record if it has at least one drug
    if len(record) > 1:
        result.append(record)

# Convert to JSON
json_output = json.dumps(result, indent=4)
print(json_output)

# Optionally save to file
with open('./data/mgadministered_gold_data.json', 'w') as f:
    f.write(json_output)