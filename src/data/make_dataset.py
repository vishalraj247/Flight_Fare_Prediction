import pandas as pd
import os
import zipfile

def load_data(path):
    all_data = []
    # Iterating through each airport folder
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            # Iterating through each zip file in the airport folder
            for file in os.listdir(folder_path):
                if file.endswith(".zip"):
                    file_path = os.path.join(folder_path, file)
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        # Extract all files in the zip file
                        zip_ref.extractall(folder_path)
                        # Iterating through each csv file in the extracted files
                        for csv_file in os.listdir(folder_path):
                            if csv_file.endswith(".csv"):
                                csv_path = os.path.join(folder_path, csv_file)
                                df = pd.read_csv(csv_path)
                                all_data.append(df)
    return pd.concat(all_data, ignore_index=True)