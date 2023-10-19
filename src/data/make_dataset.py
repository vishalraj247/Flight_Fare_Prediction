import pandas as pd
import os
import zipfile

def load_and_save_data_by_folder(path):
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        
        if os.path.isdir(folder_path):
            concatenated_data = pd.DataFrame()
            
            for file in os.listdir(folder_path):
                if file.endswith(".zip"):
                    file_path = os.path.join(folder_path, file)
                    
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        # Extract all files in the zip file
                        zip_ref.extractall(folder_path)
                        csv_file_name = file.replace('.zip', '.csv')
                        csv_path = os.path.join(folder_path, csv_file_name)
                        
                        # Concatenate CSVs
                        data = pd.read_csv(csv_path)
                        concatenated_data = pd.concat([concatenated_data, data], ignore_index=True)
            
            # Save the concatenated data to the 'interim' directory
            interim_folder_path = os.path.join('data/interim', folder)
            os.makedirs(interim_folder_path, exist_ok=True)
            concatenated_data.to_csv(f"{interim_folder_path}/{folder}_concatenated.csv", index=False)