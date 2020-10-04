r"""Convert raw Stanford Cars dataset to TFRecord for object_detection.

Example usage:
    python create_tf_record.py \
        --data_dir=/data/StanfordCars \
        --output_path=stanford_cars_train.tfrecord \
        --set=train \
        --label_map_path=stanford_cars_label_map.pbtxt

    python create_tf_record.py \
        --data_dir=/data/StanfordCars \
        --output_path=stanford_cars_test.tfrecord \
        --set=test \
        --label_map_path=stanford_cars_label_map.pbtxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import io
import logging
import csv
from tqdm import tqdm

import PIL.Image

import numpy as np

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

class_to_id = {
  'car': 1,
  'human': 2,
  'obstacle': 3,
  'sign': 4
}

flags = tf.app.flags

flags.DEFINE_string('data_dir','','Root directory to Stanford Cars dataset. (car_ims is a subfolder)')
flags.DEFINE_string('output_path','','Path to output TFRecord.')
flags.DEFINE_string('label_map_path','','Path to label map proto.')
flags.DEFINE_string('set','','Convert training set, test set, or merged set.')
flags.DEFINE_string('csv_conv','','Converted CSV labels file')
flags.DEFINE_string('csv','','raw CSV labels file')

FLAGS = flags.FLAGS

SETS = ['train', 'test', 'merged']

def read_class_weights(path):
  """
  Read weights for classes
  :param path:
  :return:
  """
  weight_dict = {}
  with open(path, 'r') as file:
    lines = file.readlines()
    lines = [line.split('\n')[0] for line in lines]
    lines = [line.split(',') for line in lines]

    for line in lines:
      weight_dict[int(line[0])]= float(line[1])

  return weight_dict

def dict_to_img_example(img_data):
  """Convert python dictionary formath data of one image to tf.Example proto.
  Args:
      img_data: infomation of one image, inclue bounding box, labels of bounding box,\
          height, width, encoded pixel data.
  Returns:
      example: The converted tf.Example
  """
  bboxes = img_data['bboxes']
  width = int(img_data['width'])
  height = int(img_data['height'])
  xmin, xmax, ymin, ymax = [], [], [], []
  for bbox in bboxes:

    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    # calculate box using original image coordinates
    xmin.append(max(0, x1 / width))
    xmax.append(min(1.0, x2 / width))
    ymin.append(max(0, y1 / height))
    ymax.append(min(1.0, y2 / height))

    # check negativity
    if xmin[-1] < 0 or xmax[-1] < 0 or ymin[-1] < 0 or ymax[-1] < 0:
      print(f'!!!!!{img_data}')

    if (xmax[-1] - xmin[-1]) <= 0 or (ymax[-1] - ymin[-1]) <= 0:
      print(f'\nWrong coordinates: xmax: {x1}, xmin: {x2}, ymax: {y1}, ymin: {y2}\npath: {img_data}')
      return None

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(int(img_data['height'])),
    'image/width': dataset_util.int64_feature(int(img_data['width'])),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
    'image/object/class/label': dataset_util.int64_list_feature(img_data['labels']),
    'image/encoded': dataset_util.bytes_feature(img_data['pixel_data']),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
  }))
  return example


def dict_to_tf_example(annotation, dataset_directory, label_map_dict, class_weight = {}):
  im_path = str(annotation['relative_im_path'])
  cls = int(class_to_id[annotation['class']])
  x1 = int(annotation['bbox_x1'])
  y1 = int(annotation['bbox_y1'])
  x2 = int(annotation['bbox_x2'])
  y2 = int(annotation['bbox_y2'])

  # read image
  full_img_path = os.path.join(dataset_directory, im_path)

  # read in the image and make a thumbnail of it
  max_size = 500, 500
  big_image = PIL.Image.open(full_img_path)
  width,height = big_image.size
  big_image.thumbnail(max_size, PIL.Image.ANTIALIAS)
  full_thumbnail_path = os.path.splitext(full_img_path)[0] + '_thumbnail.jpg'
  big_image.save(full_thumbnail_path)

  with tf.gfile.GFile(full_thumbnail_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  xmin = []
  xmax = []
  ymin = []
  ymax = []

  # check negativity
  x1 = 0 if x1 < 0 else x1
  x2 = 0 if x2 < 0 else x2
  y1 = 0 if y1 < 0 else y1
  y2 = 0 if y2 < 0 else y2

  # calculate box using original image coordinates
  xmin.append(max(0, x1 / width))
  xmax.append(min(1.0, x2 / width))
  ymin.append(max(0, y1 / height))
  ymax.append(min(1.0, y2 / height))

  # check negativity
  if xmin[-1] < 0 or xmax[-1] < 0 or ymin[-1] < 0 or ymax[-1] < 0:
    print(f'!!!!!{full_img_path}')

  if (xmax[-1]-xmin[-1]) <= 0 or (ymax[-1]-ymin[-1]) <= 0:
    print(f'\nWrong coordinates: xmax: {x1}, xmin: {x2}, ymax: {y1}, ymin: {y2}\npath: {full_img_path}')
    return None

  # set width and height to thumbnail size for tfrecord ingest
  width,height = image.size

  classes = []
  classes_text = []

  label=''
  for name, val in label_map_dict.items():
    if val == cls: 
      label = name
      break

  # weights = []
  # weights.append(class_weight[label_map_dict[label]])

  classes_text.append(label.encode('utf8'))
  classes.append(label_map_dict[label])
  
  example = tf.train.Example(features=tf.train.Features(feature={
	'image/height': dataset_util.int64_feature(height),
	'image/width': dataset_util.int64_feature(width),
	'image/filename': dataset_util.bytes_feature(full_img_path.encode('utf8')),
	'image/source_id': dataset_util.bytes_feature(full_img_path.encode('utf8')),
	'image/encoded': dataset_util.bytes_feature(encoded_jpg),
	'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
	'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
	'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
	'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
	'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
	'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
	'image/object/class/label': dataset_util.int64_list_feature(classes),
    #'image/object/weight': dataset_util.float_list_feature(weights)
  }))
  return example 

def distribute_data(val_p, dst_csv):
  csv_file = FLAGS.csv
  object_counter = {}
  result_dict = []
  csv_dict = []

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  with open(csv_file) as f:
    csv_reader = csv.DictReader(f)

    for row in csv_reader:
      cls = label_map_dict[row['class']]
      if cls not in object_counter.keys():
        object_counter[cls] = 0
      object_counter[cls] += 1

      # correct bbox coordinates
      row['bbox_x2'] = int(row['bbox_x2']) + int(row['bbox_x1'])
      row['bbox_y2'] = int(row['bbox_y2']) + int(row['bbox_y1'])

      csv_dict.append(row)

    for (k,v) in object_counter.items():
      object_counter[k] = int(v * (1-val_p))

  actual_counter = {}
  train_img = []
  val_img = []
  train_obj = 0
  val_obj = 0
  for row in tqdm(csv_dict):
    cls = label_map_dict[row['class']]
    if cls not in actual_counter.keys():
      actual_counter[cls] = 0

    if actual_counter[cls] < object_counter[cls]:
      actual_counter[cls] += 1
      if row['relative_im_path'] not in val_img:
        row['test'] = 0
        result_dict.append(row)
        train_obj += 1
        if row['relative_im_path'] not in train_img:
          train_img.append(row['relative_im_path'])

    else:
      if row['relative_im_path'] not in train_img:
        row['test'] = 1
        result_dict.append(row)
        val_obj += 1
        if row['relative_im_path'] not in val_img:
          val_img.append(row['relative_im_path'])

  print(f'train image number: {len(train_img)}, validation image number: {len(val_img)}\n'
        f'train object number: {train_obj}, validation object number: {val_obj}\n')

  with open(dst_csv, 'w') as dst_file:
    fieldnames = ['relative_im_path', 'class', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'width', 'height', 'test']

    writer = csv.DictWriter(dst_file, fieldnames=fieldnames)
    writer.writeheader()
    for object in result_dict:
      writer.writerow(object)

def main(_):

  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.formats(SETS))

  train = FLAGS.set
  data_dir = FLAGS.data_dir
  csv_file = FLAGS.csv_conv

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  with open(csv_file) as f:
    csv_reader = csv.DictReader(f)

    img_dict = {}
    for row in tqdm(csv_reader):

      img_path = os.path.join('C:/Users/gaals/PycharmProjects/onlab3D/images/SSD_images', row['relative_im_path'])

      if img_path not in img_dict.keys():
        img_dict[img_path] = {}
        img_bytes = tf.gfile.FastGFile(img_path, 'rb').read()
        img_dict[img_path]['pixel_data'] = img_bytes
        img_dict[img_path]['height'] = row['height']
        img_dict[img_path]['width'] = row['width']
        img_dict[img_path]['test'] = row['test']
        img_dict[img_path]['bboxes'] = []
        img_dict[img_path]['labels'] = []

      img_dict[img_path]['bboxes'].append([int(row['bbox_x1']), int(row['bbox_y1']), int(row['bbox_x2']), int(row['bbox_y2'])])
      img_dict[img_path]['labels'].append(int(class_to_id[row['class']]))

  cnt_tr = 0
  cnt_te = 0
  cnt_given = 0
  images_found = []
  images_given = []

  for (path,row) in tqdm(img_dict.items()):
    if row['test'] == '':
      continue

    test = int(row['test'])
    if test:
      testset = 'test'
      cnt_te += 1
    else:
      testset = 'train'
      cnt_tr += 1

    if train == 'merged' or train == testset:

      if path not in images_found:
        images_found.append(path)

      tf_example = dict_to_img_example(row)
      if tf_example is not None:
        if path not in images_given:
          images_given.append(path)
        cnt_given += 1
        writer.write(tf_example.SerializeToString())

  writer.close()

  cnt = cnt_te if train =='test' else cnt_tr
  print(f'{len(images_found)} images found, {len(images_given)} images given to tf record\n'
        f'{cnt_given} object given to tfrecord')

if __name__ == '__main__':
  tf.app.run()

  # distribute_data(val_p=0.1, dst_csv='images.csv')
