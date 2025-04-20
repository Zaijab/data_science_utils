import json
import pandas as pd

# Convert dictionary in 'keywords' to a string.
converted_records = []
for rec in records:
    rec_copy = rec.copy()
    # Convert keywords to a JSON string (ensuring sorted keys for consistency)
    rec_copy['keywords'] = json.dumps(rec_copy['keywords'], sort_keys=True)
    # Convert rmse to list if needed
    rmse_vals = rec_copy['rmse']
    if hasattr(rmse_vals, "tolist"):
        rmse_vals = rmse_vals.tolist()
    rec_copy['rmse'] = rmse_vals
    converted_records.append(rec_copy)

df = pd.DataFrame(converted_records)

# Now, assuming you have already exploded 'rmse' into individual rows with a 'random' column,
# you can group by everything except 'rmse' and 'random'.
metadata_cols = [col for col in df.columns if col not in ['rmse', 'random']]
grouped_df = df.groupby(metadata_cols)

for group_name, group_df in grouped_df:
    break

print(df)

# df.to_csv('enkf_1_4_cov.csv')
