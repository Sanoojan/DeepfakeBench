import os 
import glob

json_path="/egr/research-sprintai/baliahsa/projects/DeepfakeBench/preprocessing/dataset_json"

def get_all_json_files(json_path):
    all_json_files = glob.glob(os.path.join(json_path, "*.json"))
    
    # open json files and change "\\" to "/"
    for json_file in all_json_files:
        with open(json_file, "r") as f:
            data = f.read()
        
        data = data.replace("\\", "/")
        
        with open(json_file, "w") as f:
            f.write(data)