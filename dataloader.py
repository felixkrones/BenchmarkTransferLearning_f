import os
import torch
import random
import copy
import csv
import json
import pandas as pd
import pathlib
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from skimage.io import imread
from typing import Callable, Optional
import numpy as np
import pydicom as dicom
import cv2
from skimage import transform, io, img_as_float, exposure
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)



def build_transform_classification(normalize, crop_size=224, resize=224, mode="train", test_augment=False, nc=3):
    transformations_list = []

    if normalize.lower() == "imagenet":
      if nc == 3:
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      elif nc == 1:
        normalize = transforms.Normalize((0.485), (0.229))
      else:
        raise ValueError("nc should be 1 or 3")
    elif normalize.lower() == "chestx-ray":
      if nc == 3:
        normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
      elif nc == 1:
        normalize = transforms.Normalize((0.5056), (0.252))
      else:
        raise ValueError("nc should be 1 or 3")
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

def build_transform_segmentation():
  AUGMENTATIONS_TRAIN = Compose([
    # HorizontalFlip(p=0.5),
    OneOf([
        RandomBrightnessContrast(),
        RandomGamma(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    RandomSizedCrop(min_max_height=(156, 224), height=224, width=224,p=0.25),
    ToFloat(max_value=1)
    ],p=1)

  return AUGMENTATIONS_TRAIN


class PadchestDataset(Dataset):
  def __init__(self, images_path, file_path, augment, diseases_to_test, nc=1):
    self.img_path_col = "ImageID"
    self.label_col = "Labels"
    self.filter_dir = {
       "Projection": ["PA", "AP"],
       "ImageDir": [0],
       "MethodProjection": ["Manual review of DICOM fields"]
    }
    self.transform = augment
    self.annotation_file = pd.read_csv(file_path)
    self.nc = nc

    # Filder the data based on self.filter_dir
    for key, value in self.filter_dir.items():
      self.annotation_file = self.annotation_file[self.annotation_file[key].isin(value)]

    # Get possible labels
    self.possible_labels = np.unique([list for sublist in self.annotation_file['Labels'].fillna('[]').apply(lambda x: eval(x)).values.tolist() for list in sublist] + [d.lower() for d in diseases_to_test])

    # Get labels
    df_aux = pd.concat([self.annotation_file[self.label_col].apply(lambda x: 1 if p in eval(x) else 0).rename(p) for p in self.possible_labels], axis=1)
    self.annotation_file = pd.concat((self.annotation_file, df_aux), axis=1)
    self.img_label = self.annotation_file[self.possible_labels].values
    self.img_list = [os.path.join(images_path, x) for x in self.annotation_file[self.img_path_col].values]

    self.annotation_file.to_csv(file_path.replace(".csv", "_filtered.csv"), index=False)
    self.possible_labels = self.possible_labels.tolist()

  def __len__(self):
    self._length = len(self.img_list)
    print(f"Length of dataset: {self._length}")
    return self._length 
   
  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageLabel = torch.from_numpy(self.img_label[index])
    imageData = Image.open(imagePath)
    if self.nc == 3:
      if imageData.mode != 'RGB':
        imageData = imageData.convert('RGB')
    elif self.nc == 1:
      if imageData.mode != 'L':
        imageData = imageData.convert('L')
    else:
      raise Exception("Invalid number of channels")
    if self.transform != None: 
       imageData = self.transform(imageData)
    return imageData, imageLabel
  

class COCO(Dataset):
  def __init__(self, images_path, file_path, augment, nc=3, n_samples=None):
     
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.nc = nc
    self.n_samples = n_samples

    self._json = json.load(open(file_path, 'r'))
    df_images = pd.DataFrame(self._json['images'])
    df_annotations = pd.DataFrame(self._json['annotations'])
    self.df_images_annotations = pd.merge(df_images, df_annotations, how="left", left_on="id", right_on="image_id", suffixes=("_image", "_annotation"))

    if self.n_samples is None:
        self.n_samples = len(self.df_images_annotations)
    if self.n_samples < len(self.df_images_annotations):
        self.df_images_annotations = self.df_images_annotations.sample(n=self.n_samples, random_state=42)
    else:
        raise ValueError(f"n_samples must be less than or equal to the number of images in the dataset, which is {len(self.metadata)}")

    self.img_list = list(self.df_images_annotations["file_name"].apply(lambda x: os.path.join(images_path, x)))
    self.img_label = list(self.df_images_annotations["category_id"].values)

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel


class MIMIC_Dataset(Dataset):

  def __init__(self, images_path, file_path, augment, possible_labels=None, n_samples=1000, annotation_file="mimic-cxr-2.0.0-chexpert.csv"):
    
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.datapath = os.path.join(images_path, 'physionet.org', 'files', 'mimic-cxr-jpg', '2.0.0')
    self._annotation_file = pd.read_csv(pathlib.Path(self.datapath) / annotation_file)
    self._labels = [c for c in self._annotation_file.columns if c not in ["subject_id", "study_id", "dicom_id", "split", "view", "img_path"]]
    self.metadata = pd.read_csv(file_path)
    self.possible_labels = possible_labels
    self.n_samples = n_samples

    # Create paths to images
    self.metadata = pd.read_csv(os.path.join(self.datapath, 'mimic-cxr-2.0.0-metadata.csv'))
    self.metadata["subject_category"] = self.metadata["subject_id"].apply(lambda x: int(str(x)[:2]))
    self.metadata["img_path"] = self.metadata[["subject_category", "subject_id", "study_id", "dicom_id"]].apply(lambda x: os.path.join(self.datapath, "files",  f"p{x[0]}/p{x[1]}/s{x[2]}/{x[3]}.jpg"), axis=1)
    
    # Filter data to only include images from the possible patient ids
    #self.metadata = self.metadata[self.metadata["subject_category"].isin(possible_patient_ids)]

    # Filter data to only include images with a view position of PA or AP
    self.metadata = self.metadata[self.metadata["ViewPosition"].isin(["PA", "AP"])]

    # Merge columns from self._annotation_file left to self.metadata, based on columns subject_id and study_id and check if number of rows is the same
    old_shape = len(self.metadata)
    self.metadata = pd.merge(self.metadata, self._annotation_file, how="left", on=["subject_id", "study_id"])
    assert len(self.metadata) == old_shape, "Number of rows in self.metadata changed, prob. because the label file contained duplicates."

    # Remove observations where no column from self._labels has any of the values -1, 0, 1
    old_shape = len(self.metadata)
    self.metadata = self.metadata[self.metadata[self._labels].isin([-1, 0, 1]).any(axis=1)]
    print(f"Removed {old_shape - len(self.metadata)} observations where no column from self._labels had any of the values -1, 0, 1.")

    # Randomly sample n_samples images
    if self.n_samples is None:
        self.n_samples = len(self.metadata)
    if self.n_samples < len(self.metadata):
        self.metadata = self.metadata.sample(n=self.n_samples, random_state=42)
    else:
        raise ValueError(f"n_samples must be less than or equal to the number of images in the dataset, which is {len(self.metadata)}")

    # Get labels
    if self.possible_labels is None:
      self.possible_labels = self._labels

    self.img_list = []
    self.img_label = []

    for index, row in self.metadata.iterrows():
      self.img_list.append(row["img_path"])
      self.img_label.append([1 if row[label] == 1 else 0 for label in self.possible_labels])

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel


class ChestXray14Dataset_general(Dataset):

  def __init__(self, images_path, file_path, augment, possible_labels, annotaion_percent=100, annotation_file="Data_Entry_2017_v2020.csv"):
    
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self._annotation_file = pd.read_csv(pathlib.Path('/'.join(images_path.split('/')[:-2])) / annotation_file)
    self.possible_labels = possible_labels

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])

          labels = self._annotation_file[self._annotation_file["Image Index"]==lineItems[0]]["Finding Labels"].values[0]
          imageLabel = [1 if label in labels else 0 for label in possible_labels]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotaion_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotaion_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)


class ChestXray14Dataset(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14, annotaion_percent=100, nc=1):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.nc = nc

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotaion_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotaion_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    if self.nc == 3:
      imageData = Image.open(imagePath).convert('RGB')
    elif self.nc == 1:
      imageData = Image.open(imagePath).convert('L')
    else:
       raise ValueError(f"args.nc must be 1 or 3, not {self.nc}")
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)


# ---------------------------------------------Downstream CheXpert------------------------------------------
class CheXpertDataset(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100, nc=3):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.nc = nc
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              label[i] = -1
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        imageLabel = [int(i) for i in label]
        self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath)
    if self.nc == 3:
      if imageData.mode != 'RGB':
        imageData = imageData.convert('RGB')
    elif self.nc == 1:
      if imageData.mode != 'L':
        imageData = imageData.convert('L')
    else:
      raise Exception("Invalid number of channels")

    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream Shenzhen------------------------------------------
class ShenzhenCXR(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=1, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)


#__________________________________________Lung Segmentation, Montgomery dataset --------------------------------------------------
class MontgomeryDataset(Dataset):
    """NIH dataset."""

    def __init__(self, pathImageDirectory, pathMaskDirectory,transforms,dim=(224, 224, 3), anno_percent=100,num_class=1,normalization=None):
        self.transforms = transforms
        self.dim = dim
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        self.normalization = normalization
        self.img_list= os.listdir(pathImageDirectory)

        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list= copy.deepcopy(self.img_list)
            self.img_list = []

            for i in indexes:
                self.img_list.append(_img_list[i])

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name= self.img_list[idx]
        image = Image.open(os.path.join(self.pathImageDirectory,image_name))
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')
        mask = Image.open(os.path.join(self.pathMaskDirectory,image_name))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255
        if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(image) / 255.
            mask = np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std

        mask = np.array(mask)
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (im, mask)


#__________________________________________DRIVE dataset --------------------------------------------------

class DriveDataset(Dataset):
    """NIH dataset."""

    def __init__(self, pathImageDirectory, pathMaskDirectory,size=512):

        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory

        files = os.listdir(pathImageDirectory)
        data = []
        labels = []

        for i in files:
            im = Image.open(os.path.join(pathImageDirectory,i))
            im = im.convert('RGB')
            im = (np.array(im)).astype('uint8')
            label = Image.open(os.path.join(pathMaskDirectory, i.split('_')[0] + '_manual1.png'))
            label = label.convert('L')
            label = (np.array(label)).astype('uint8')
            data.append(cv2.resize(im, (size, size)))
            temp = cv2.resize(label, (size, size))
            _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
            labels.append(temp)

        self.data = np.array(data)
        self.label = np.array(labels)

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.data = self.data.astype('float32') / 255.
        self.label = self.label.astype('float32') / 255.

        for i in range(3):
            self.data[:, :, :, i] = (self.data[:, :, :, i] - mean[i]) / std[i]

        self.data = np.reshape(self.data, (
            len(self.data), size, size, 3))  # adapt this if using `channels_first` image data format
        self.label = np.reshape(self.label,
                             (len(self.label), size, size, 1))  # adapt this if using `channels_first` im

    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        image = self.data[idx]
        mask = self.label[idx]

        image = image.transpose(2, 0, 1).astype('float32')
        mask = mask.transpose(2, 0, 1).astype('float32')

        return (image, mask)

#__________________________________________SIIM Pneumothorax segmentation dataset --------------------------------------------------
class PNEDataset(Dataset):
    """NIH dataset."""

    def __init__(self, pathImageDirectory, pathMaskDirectory,transforms,dim=(224, 224, 3),normalization=None):
        self.pathImageDirectory = pathImageDirectory
        self.pathMaskDirectory = pathMaskDirectory
        self.transforms = transforms
        self.dim = dim
        self.normalization = normalization
        self.img_list = os.listdir(pathImageDirectory)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name= self.img_list[idx]
        ds = dicom.dcmread(os.path.join(self.pathImageDirectory,image_name))
        img = np.array(ds.pixel_array)
        im = cv2.resize(img, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        im = (np.array(im)).astype('uint8')
        if len(im.shape) == 2:
            im = np.repeat(im[..., None], 3, 2)
        mask = Image.open(os.path.join(self.pathMaskDirectory,image_name))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255
        mask = (np.array(mask)).astype('uint8')

        if self.transforms:
                augmented = self.transforms(image=im, mask=mask)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(im) / 255.
            mask = np.array(mask) / 255.

        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std

        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0)
        return (im, mask)


class VinDrCXR(Dataset):
    def __init__(self, images_path, file_path, augment, nc=1):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.nc = nc
        annotation_file = pd.read_csv(os.path.join(images_path, "physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv"))

        self.possible_labels = annotation_file.columns[1:].tolist()

        with open(file_path, "r") as fr:
            line = fr.readline().strip()
            while line:
                lineItems = line.split()
                imagePath = os.path.join(images_path, lineItems[0]+".jpg")
                image_id = lineItems[0].split("/")[-1]
                imageLabel = np.array([int(i) for i in annotation_file[annotation_file["image_id"] == image_id].values[0][1:]])
                self.img_list.append(imagePath)
                self.img_label.append(imageLabel)
                line = fr.readline()

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.from_numpy(self.img_label[index])
        imageData = Image.open(imagePath)
        if self.nc == 3:
          if imageData.mode != 'RGB':
            imageData = imageData.convert('RGB')
        elif self.nc == 1:
          if imageData.mode != 'L':
            imageData = imageData.convert('L')
        else:
          raise Exception("Invalid number of channels")
        if self.augment != None: 
           imageData = self.augment(imageData)
        return imageData, imageLabel
    
    def __len__(self):
        return len(self.img_list)