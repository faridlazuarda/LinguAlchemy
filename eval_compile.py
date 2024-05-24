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

            # Extract only the accuracy scores
            for language, metrics in json_data.items():
                accuracy = metrics.get('accuracy', None)
                # Create a temporary DataFrame and append it to the main DataFrame using pd.concat
                df_temp = pd.DataFrame([{
                    'Language': language,
                    'Model': model,
                    'File': json_file[:-5],  # Remove '.json' from the file name
                    'Accuracy': accuracy
                }])
                df = pd.concat([df, df_temp], ignore_index=True)

# Convert accuracy to percentage and format it to 2 decimal places
df['Accuracy'] = df['Accuracy'].apply(lambda x: round(x * 100, 2))

# Pivot the DataFrame to get the desired table format with multi-level columns
df_pivot = df.pivot_table(index='Language', columns=['Model', 'File'], values='Accuracy')

# Optional: Save the DataFrame to an Excel file
df_pivot.to_excel('output_accuracy_scores.xlsx')

print(df_pivot.head())
