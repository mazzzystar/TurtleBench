import csv
import json
import os
import re
from datetime import datetime


def stats():
    logs_folder = 'logs'

    # Prepare a list to store the collected data
    data_list = []

    # Iterate over subfolders in logs/
    subfolders = [f.path for f in os.scandir(logs_folder) if f.is_dir()]

    for subfolder in subfolders:
        # Get the language and shot_type from the subfolder name
        # Assuming subfolder name is like 'zh_with_0shots' or 'en_with_2shots'
        subfolder_name = os.path.basename(subfolder)
        match = re.match(r'(\w+)_with_(\d+)shots', subfolder_name)
        if not match:
            continue
        language, shot_type = match.groups()

        # Now process the JSON files in this subfolder
        json_files = [f for f in os.listdir(subfolder) if f.endswith('.json')]

        # Build a dictionary to keep track of the latest file for each model
        model_files = {}
        for json_file in json_files:
            # The filenames are like 'all_cases_{model_name}_{language}_shot{shot_type}_{i}.json'
            pattern = r'all_cases_(.+?)_' + re.escape(language) + \
                r'_shot' + re.escape(shot_type) + r'_(\d+)\.json'
            match = re.match(pattern, json_file)
            if not match:
                continue
            model_name, index = match.groups()
            index = int(index)
            # Keep the JSON file with the highest index for each model
            if model_name not in model_files or index > model_files[model_name][1]:
                model_files[model_name] = (json_file, index)

        # Now, for each model, read the latest JSON file
        for model_name, (json_file, index) in model_files.items():
            json_path = os.path.join(subfolder, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Get the model name and other info
            overall = data.get('overall', {})
            # Collect the data
            row = {
                'model': model_name,
                'language': language,
                'shot_type': shot_type,
                # 'timestamp': info.get('timestamp', ''),
                'total_samples': overall.get('total_samples', 0),
                'correct': overall.get('correct', 0),
                'accuracy': overall.get('accuracy', 0.0),
                'time_usage': overall.get('time_usage', 0.0),
                'total_tokens': overall.get('total_tokens', 0)
            }
            data_list.append(row)

    data_list.sort(key=lambda x: x['accuracy'], reverse=True)

    # Now write the data_list to a CSV file
    stats_folder = 'stats'
    if not os.path.exists(stats_folder):
        os.makedirs(stats_folder)

    filename = f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(os.path.join(stats_folder, filename), mode='w', newline='') as csvfile:
        fieldnames = ['model', 'language', 'shot_type', 'total_samples',
                      'correct', 'accuracy', 'time_usage', 'total_tokens']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_list:
            writer.writerow(row)

    print(f"Stats collected and saved to {filename}")


if __name__ == "__main__":
    stats()
