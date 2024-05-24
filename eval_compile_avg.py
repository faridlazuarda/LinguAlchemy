import os
import pandas as pd
import json

# Define the directory containing the JSON files
json_directory = '/home/alham.fikri/farid/lingualchemy/outputs/massive'
models = ['bert-base-multilingual-cased_scale10', 'xlm-roberta-base_scale10']

# Initialize a DataFrame to hold all the data
df = pd.DataFrame()

# Loop through each model directory
for model in models:
    model_directory = os.path.join(json_directory, model)
    if not os.path.exists(model_directory):
        print(f"Directory {model_directory} does not exist")
        continue

    json_files = [f for f in os.listdir(model_directory) if f.endswith('.json')]
    
    # Process each JSON file in the model's directory
    for json_file in json_files:
        file_path = os.path.join(model_directory, json_file)
        with open(file_path, 'r') as file:
            json_data = json.load(file)

            # Extract all metrics from all languages and aggregate
            for language, metrics in json_data.items():
                metrics['Model'] = model
                metrics['File'] = json_file[:-5]  # Remove '.json' from the file name
                df_temp = pd.DataFrame([metrics])
                df = pd.concat([df, df_temp], ignore_index=True)

# Compute the average of each metric for each model-file combination
metric_columns = ['precision_macro', 'precision_micro', 'recall_macro', 'recall_micro', 'f1_macro', 'f1_micro', 'accuracy']
df_avg = df.groupby(['Model', 'File'])[metric_columns].mean()

# Convert the scores to percent and format to 2 decimal places for all metrics
df_avg = df_avg.applymap(lambda x: round(x * 100, 2))

# Optional: Save the DataFrame to an Excel file
df_avg.to_excel('output_avg_metrics.xlsx')

print(df_avg)
