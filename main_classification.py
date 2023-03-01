import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm

from utils import vararg_callback_bool, vararg_callback_int
from dataloader import  *

import torch
from engine import classification_engine

sys.setrecursionlimit(40000)


def get_args_parser():
    parser = OptionParser()

    parser.add_option("--GPU", dest="GPU", help="the index of gpu is used", default=None, action="callback",
                      callback=vararg_callback_int)
    parser.add_option("--model", dest="model_name", help="DenseNet121", default="Resnet50", type="string")
    parser.add_option("--init", dest="init",
                      help="Random| ImageNet| ImageNet_1k| ImageNet_21k| SAM| DeiT| BEiT| DINO| MoCo_V3| GMML| MoBY | MAE| SimMIM",
                      default="Random", type="string")
    parser.add_option("--num_class", dest="num_class", help="number of the classes in the downstream task",
                      default=14, type="int")
    parser.add_option("--data_set", dest="data_set", help="ChestXray14|CheXpert", default="ChestXray14", type="string")
    parser.add_option("--normalization", dest="normalization", help="how to normalize data (imagenet|chestx-ray)", default="imagenet",
                      type="string")
    parser.add_option("--img_size", dest="img_size", help="input image resolution", default=224, type="int")
    parser.add_option("--img_depth", dest="img_depth", help="num of image depth", default=3, type="int")
    parser.add_option("--data_dir", dest="data_dir", help="dataset dir",default="data/", type="string")
    parser.add_option("--train_list", dest="train_list", help="file for training list",
                      default=None, type="string")
    parser.add_option("--val_list", dest="val_list", help="file for validating list",
                      default=None, type="string")
    parser.add_option("--test_list", dest="test_list", help="file for test list",
                      default=None, type="string")
    parser.add_option("--mode", dest="mode", help="train | test", default="train", type="string")
    parser.add_option("--batch_size", dest="batch_size", help="batch size", default=32, type="int")
    parser.add_option("--num_epoch", dest="num_epoch", help="num of epoches", default=20, type="int")
    parser.add_option("--optimizer", dest="optimizer", help="Adam | SGD", default="Adam", type="string")
    parser.add_option("--lr", dest="lr", help="learning rate", default=2e-4, type="float")
    parser.add_option("--lr_Scheduler", dest="lr_Scheduler", help="learning schedule", default="ReduceLROnPlateau",
                      type="string")
    parser.add_option("--patience", dest="patience", help="num of patient epoches", default=10, type="int")
    parser.add_option("--early_stop", dest="early_stop", help="whether use early_stop", default=True, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--trial", dest="num_trial", help="number of trials", default=1, type="int")
    parser.add_option("--start_index", dest="start_index", help="the start model index", default=0, type="int")
    parser.add_option("--clean", dest="clean", help="clean the existing data", default=False, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--resume", dest="resume", help="whether latest checkpoint", default=True, action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--best", dest="best", help="whether to use last or best checkpoint", default="last", action="callback",
                      callback=vararg_callback_bool)
    parser.add_option("--workers", dest="workers", help="number of CPU workers", default=8, type="int")
    parser.add_option("--print_freq", dest="print_freq", help="print frequency", default=50, type="int")
    parser.add_option("--test_augment", dest="test_augment", help="whether use test time augmentation",
                      default=True, action="callback", callback=vararg_callback_bool)
    parser.add_option("--proxy_dir", dest="proxy_dir", help="Path to the Pretrained model", default=None, type="string")
    parser.add_option("--label_layers", default=0, type=int, help='For GMML, how many label token layers. 12 means introduce LT at the first layer')
    parser.add_option("--anno_percent", dest="anno_percent", help="data percent", default=100, type="int")
    parser.add_option("--device", dest="device", help="cpu|cuda", default="cuda", type="string")
    parser.add_option("--activate", dest="activate", help="Sigmoid", default="Sigmoid", type="string")
    parser.add_option("--uncertain_label", dest="uncertain_label",
                      help="the label assigned to uncertain data (Ones | Zeros | LSR-Ones | LSR-Zeros)",
                      default="LSR-Ones", type="string")
    parser.add_option("--unknown_label", dest="unknown_label", help="the label assigned to unknown data",
                      default=0, type="int")


    (options, args) = parser.parse_args()

    return options


def main(args):
    print(args)
    assert args.data_dir is not None
    assert args.train_list is not None
    assert args.val_list is not None
    assert args.test_list is not None
    if args.init.lower() != 'imagenet' and args.init.lower() != 'random':
        assert args.proxy_dir is not None

    args.exp_name = args.model_name + "_" + args.init
    model_path = os.path.join("./Models/Classification",args.data_set)
    output_path = os.path.join("./Outputs/Classification",args.data_set)

    if args.data_set == "ChestXray14":
        diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pneumonia", "Pneumothorax", 'No Finding']
        diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        #dataset_train = ChestXray14Dataset_general(images_path=args.data_dir, file_path=args.train_list,augment=build_transform_classification(normalize=args.normalization, mode="train"), possible_labels=diseases)
        #dataset_val = ChestXray14Dataset_general(images_path=args.data_dir, file_path=args.val_list,augment=build_transform_classification(normalize=args.normalization, mode="valid"), possible_labels=diseases)
        #dataset_test = ChestXray14Dataset_general(images_path=args.data_dir, file_path=args.test_list,augment=build_transform_classification(normalize=args.normalization, mode="test"), possible_labels=diseases)
        dataset_train = ChestXray14Dataset(images_path=args.data_dir, file_path=args.train_list,augment=build_transform_classification(normalize=args.normalization, mode="train"), num_class=len(diseases))
        dataset_val = ChestXray14Dataset(images_path=args.data_dir, file_path=args.val_list,augment=build_transform_classification(normalize=args.normalization, mode="valid"), num_class=len(diseases))
        dataset_test = ChestXray14Dataset(images_path=args.data_dir, file_path=args.test_list,augment=build_transform_classification(normalize=args.normalization, mode="test"), num_class=len(diseases))

        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)


    elif args.data_set == "CheXpert":
        diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pneumonia", "Pneumothorax", 'No Finding']
        diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                           'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                           'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        test_diseases_name = diseases
        test_diseases_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        test_diseases = [diseases.index(c) for c in test_diseases_name]
        dataset_train = CheXpertDataset(images_path=args.data_dir, file_path=args.train_list,
                                        augment=build_transform_classification(normalize=args.normalization, mode="train"), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label, annotation_percent=args.anno_percent, num_class=len(diseases))
        dataset_val = CheXpertDataset(images_path=args.data_dir, file_path=args.val_list,
                                      augment=build_transform_classification(normalize=args.normalization, mode="valid"), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label, annotation_percent=args.anno_percent, num_class=len(diseases))
        dataset_test = CheXpertDataset(images_path=args.data_dir, file_path=args.test_list,
                                       augment=build_transform_classification(normalize=args.normalization, mode="test"), uncertain_label=args.uncertain_label, unknown_label=args.unknown_label, annotation_percent=args.anno_percent, num_class=len(diseases))

        print(f"Got dataset {args.data_set}, starting classification_engine.")
        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases)

    elif args.data_set == "Shenzhen":
        diseases = ['TB']
        dataset_train = ShenzhenCXR(images_path=args.data_dir, file_path=args.train_list,
                                    augment=build_transform_classification(normalize=args.normalization, mode="train"), annotation_percent=args.anno_percent)

        dataset_val = ShenzhenCXR(images_path=args.data_dir, file_path=args.val_list,
                                  augment=build_transform_classification(normalize=args.normalization, mode="valid"), annotation_percent=args.anno_percent)

        dataset_test = ShenzhenCXR(images_path=args.data_dir, file_path=args.test_list,
                                   augment=build_transform_classification(normalize=args.normalization, mode="test"), annotation_percent=args.anno_percent)


        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)


if __name__ == '__main__':
    args = get_args_parser()
    main(args)

