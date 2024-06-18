import argparse
import os
import json
import numpy as np






def main():
    parser = argparse.ArgumentParser(
        description='prepare validation detesets')
    parser.add_argument(
        '--dataset',
        choices=['mapillary', 'idd', 'bdd', 'wilddash', 'acdc'],
        help='dataset to prepare')
    parser.add_argument(
        '--root',
        help='root directory for the dataset')
    parser.add_argument(
        '--savedir',
        help='directory to save processed data')

    args = parser.parse_args()
    
    dataset_root = args.root
    dataset_name = args.dataset
    save_dir = args.savedir
    with open("label_conversion.json", 'r') as j:
        conv_dict = json.loads(j.read())[dataset_name]
    getattr(__import__('dataset_func'), "process_" + dataset_name)(dataset_root, conv_dict, dataset_name, save_dir)





if __name__ == '__main__':
    main()
