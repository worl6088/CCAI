from __future__ import division
from parser import *
from data_modules.bounding_boxes import *
from train import *
from utils.utils import get_data_list
import os

ROOT_DIR = os.path.dirname(os.path.abspath('CCAI'))
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, 'data','custom')
DATA_FILE_DIR = os.path.join(ROOT_DIR,'config','custom_data')
cfg_path =  os.path.join(ROOT_DIR, 'config','custom_cfg')
model_save_path = os.path.join(ROOT_DIR,'weights', 'custom_weight')

if __name__ == '__main__':
    args = parser_arguments()
    if args.command == 'downloader':
        domain_groups = bounding_boxes_images(args,ROOT_DIR, DEFAULT_DATA_DIR)
        print(domain_groups)
    elif args.command == 'train':
        domain_groups = get_domain_group(DEFAULT_DATA_DIR) #
        # {'Park': [3, 'Person', 'Tree', 'Dog'], 'Highway': [2, 'Car', 'Bus']}
        domain_names = []
        for key in domain_groups:
            domain_names.append(key)

        data_files = get_data_list(DATA_FILE_DIR)

        for key, data in data_files.items():
            data_config = parse_data_config(data)
            print(str(data_config))
            train_path = data_config['train']
            valid_path = data_config["valid"]
            class_names = load_classes(data_config["names"])
            if args.model_def:
                model_cfg = args.model_def
            else:
                model_cfg = get_group_cfg(cfg_path,'yolo', data_config['classes']) # trained with yolov3 model in default
            print('model_cfg_file:'+str(model_cfg))
            train(args, True, train_path, valid_path, class_names, model_cfg, key,model_save_path)


    elif args.command == 'all':

        domain_groups = bounding_boxes_images(args,ROOT_DIR, DEFAULT_DATA_DIR)
        domain_names = []
        for key in domain_groups:
            domain_names.append(key)
        data_files = [os.path.join(DATA_FILE_DIR, x) for x in os.listdir(DATA_FILE_DIR)]

        for index, data in enumerate(data_files):
            data_config = parse_data_config(data)
            train_path = data_config['train']
            valid_path = data_config["valid"]
            class_names = load_classes(data_config["names"])
            if args.model_def:
                model_cfg = args.model_def
            else:
                model_cfg = get_group_cfg(cfg_path, 'yolo', data_config['classes'])
            train(args, True, train_path, valid_path, class_names, model_cfg, domain_names[index],model_save_path)


