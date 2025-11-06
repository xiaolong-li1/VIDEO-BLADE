import io
import os
import json
import zipfile
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constant import *

def submission(model_name, zip_file):
    os.makedirs(model_name, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(model_name)
    upload_data = {}
    # load your score
    for file in os.listdir(model_name):
        if file.startswith('.') or file.startswith('__'):
            print(f"Skip the file: {file}")
            continue
        cur_file = os.path.join(model_name, file)
        if os.path.isdir(cur_file):
            for subfile in os.listdir(cur_file):
                if subfile.endswith(".json"):
                    with open(os.path.join(cur_file, subfile)) as ff:
                        cur_json = json.load(ff)
                        if isinstance(cur_json, dict):
                            for key in cur_json:
                                upload_data[key.replace('_',' ')] = cur_json[key][0]
        elif cur_file.endswith('json'):
            with open(cur_file) as ff:
                cur_json = json.load(ff)
                if isinstance(cur_json, dict):
                    for key in cur_json:
                        upload_data[key.replace('_',' ')] = cur_json[key][0]
        
        for key in TASK_INFO:
            if key not in upload_data:
                upload_data[key] = 0
    return upload_data

def load_from_directory(directory_path):
    """Load evaluation results directly from a directory containing JSON files"""
    upload_data = {}
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    # Find all eval_results.json files in the directory
    for file in os.listdir(directory_path):
        if file.startswith('.') or file.startswith('__'):
            continue
            
        if file.endswith('_eval_results.json'):
            file_path = os.path.join(directory_path, file)
            try:
                with open(file_path) as f:
                    cur_json = json.load(f)
                    if isinstance(cur_json, dict):
                        for key in cur_json:
                            # Convert underscore format to space format to match TASK_INFO
                            formatted_key = key.replace('_', ' ')
                            upload_data[formatted_key] = cur_json[key][0]
                            print(f"Loaded {formatted_key}: {cur_json[key][0]}")
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Warning: Error reading {file}: {e}")
                continue
    
    # Fill missing dimensions with 0
    for key in TASK_INFO:
        if key not in upload_data:
            upload_data[key] = 0
            print(f"Missing dimension {key}, set to 0")
    
    return upload_data

def get_nomalized_score(upload_data):
    # get the normalize score
    normalized_score = {}
    for key in TASK_INFO:
        min_val = NORMALIZE_DIC[key]['Min']
        max_val = NORMALIZE_DIC[key]['Max']
        normalized_score[key] = (upload_data[key] - min_val) / (max_val - min_val)
        normalized_score[key] = normalized_score[key] * DIM_WEIGHT[key]
    return normalized_score

def get_quality_score(normalized_score):
    quality_score = []
    for key in QUALITY_LIST:
        quality_score.append(normalized_score[key])
    quality_score = sum(quality_score)/sum([DIM_WEIGHT[i] for i in QUALITY_LIST])
    return quality_score

def get_semantic_score(normalized_score):
    semantic_score = []
    for key in SEMANTIC_LIST:
        semantic_score.append(normalized_score[key])
    semantic_score  = sum(semantic_score)/sum([DIM_WEIGHT[i] for i in SEMANTIC_LIST ])
    return semantic_score

def get_final_score(quality_score,semantic_score):
    return (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Calculate final score from evaluation results')
    parser.add_argument('--zip_file', type=str, help='Name of the zip file (legacy mode)')
    parser.add_argument('--model_name', type=str, help='Name of the model (legacy mode)')
    parser.add_argument('--result_dir', type=str, help='Path to directory containing evaluation result JSON files')
    args = parser.parse_args()

    # Determine which mode to use
    if args.result_dir:
        # New directory mode
        upload_dict = load_from_directory(args.result_dir)
        print(f"Loaded evaluation results from: {args.result_dir}")
    elif args.zip_file and args.model_name:
        # Legacy zip file mode
        upload_dict = submission(args.model_name, args.zip_file)
    else:
        print("Error: Please provide either --result_dir for directory mode, or both --zip_file and --model_name for legacy mode")
        parser.print_help()
        exit(1)
    
    print(f"\nEvaluation results: \n{upload_dict} \n")
    normalized_score = get_nomalized_score(upload_dict)
    quality_score = get_quality_score(normalized_score)
    semantic_score = get_semantic_score(normalized_score)
    final_score = get_final_score(quality_score, semantic_score)
    print('+------------------|------------------+')
    print(f'|     quality score|{quality_score:.6f}|')
    print(f'|    semantic score|{semantic_score:.6f}|')
    print(f'|       total score|{final_score:.6f}|')
    print('+------------------|------------------+')
