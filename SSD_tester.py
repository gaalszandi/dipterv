import tensorflow as tf
import csv
import sys
from PIL import Image
import glob
import numpy as np
import os
import shutil as sh
import glob

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

flags = tf.app.flags
flags.DEFINE_string('model', '', 'Frozen graph file')
flags.DEFINE_string('labels', '', 'pbtxt labels file')
flags.DEFINE_string('image', '', 'Image or image dir to run prediction on')
flags.DEFINE_string('output_dir', '', 'Image to run prediction on')
flags.DEFINE_string('gt_csv', '', 'CSV file with groundtruth')
flags.DEFINE_string('class_num', '','number of classes')
FLAGS = flags.FLAGS

NUM_CLASSES = int(FLAGS.class_num)


def configure_cpu():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    # TODO grayscale images?
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_groundtruth_into_dict(filepath):
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        result_dict = {}
        for row in reader:
            if row['test'] == '':
                continue
            if int(row['test']) == 1:
                result_dict[row['relative_im_path'].rsplit('/',1)[1]] = int(row['class'])

    return result_dict


def run_inference_for_single_image(image, graph):
    # TODO load_image_into_numpy_array
    with graph.as_default():
        configure_cpu()
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image
                # size.
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict

def match_all_with_gt(best_result, results, gt_name):

    correct = 0
    correct_in_detections = 0
    if best_result == gt_name:
        correct = 1

    if gt_name in results:
        correct_in_detections = 1
    return correct, correct_in_detections

def match_make_type_vp(best_result, gt_id, category_index):
    gt_name = category_index[gt_id]['name']
    [gt_make, gt_type, gt_vp] = gt_name.split('_')
    result_name = category_index[best_result]['name']
    [res_make, res_type, res_vp] = result_name.split('_')

    correct = {}

    correct['make'] = 0 if gt_make =='UNK'else (gt_make == res_make)
    correct['type'] = (gt_type == res_type)
    correct['vp'] = (gt_vp == res_vp)

    return correct


def create_test_dir(path, dst_dir):

    files = glob.glob(f"{path}/**/*.jp*g", recursive=True)

    labels = []
    for file in files:
        label = file.rsplit('/', 2)[1]
        if label not in labels:
            labels.append(label)
            sh.copyfile(file,f"{dst_dir}/{file.rsplit('/', 1)[1]}")


def main(_):
    if not FLAGS.image:
        print("No image path provided to predict on")
        print("Expected --image <image_path> as argument")
        sys.exit(1)
    elif not FLAGS.gt_csv:
        print("No groundtruth provided")
        sys.exit(1)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    else:
        sh.rmtree(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # label map processing
    label_map = label_map_util.load_labelmap(FLAGS.labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # groundtruth csv
    gt_dict = load_groundtruth_into_dict(FLAGS.gt_csv)

    # only one image
    if FLAGS.image.find('.jpg') > 0 or FLAGS.image.find('jpeg') > 0:
        images = [FLAGS.image]
    else:
        files = glob.glob(f"{FLAGS.image}/**/*.jp*g", recursive=True)
        images = []
        if files is not None:
            images = files

    total = 0
    correct = 0
    correct_in_detections = 0
    correct_make = 0
    correct_type = 0
    correct_vp = 0

    for image_path in images:
        img_name = image_path.rsplit('/', 1)[1]
        if img_name in gt_dict.keys():
            gt_name = gt_dict[img_name]
        else:
            print(f'{img_name} image not in groundtruth list')
            continue

        print(image_path)

        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        if image.mode is not 'RGB':
            continue
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)

        total += 1
        # check with groundtruth
        if len(output_dict['detection_classes']) > 0:
            max_idx = np.argmax(output_dict['detection_scores'])

            correct_dict = match_make_type_vp(output_dict['detection_classes'][max_idx], gt_name, category_index)
            correct_make += correct_dict['make']
            correct_type += correct_dict['type']
            correct_vp += correct_dict['vp']

            # correct_in_detections += 1

        print(output_dict['detection_scores'][0], output_dict['detection_classes'][0])
        print(category_index[output_dict['detection_classes'][0]]['name'])

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        myim = Image.fromarray(image_np)
        myim_basename = os.path.basename(image_path)
        myim.save(os.path.join(FLAGS.output_dir, myim_basename))

    # calculate accuracy

    acc ={
        'make': correct_make/total *100,
        'type': correct_type/total * 100,
        'viewpoint': correct_vp/total * 100
    }

    print (acc)

    # acc_first = correct/total * 100
    # acc_contain = correct_in_detections/total * 100
    # print(f"Overall accuracy: {acc_first} % from {total} images")
    # print(f"Accuracy for detections contain the correct result: {acc_contain} % from {total} images")


if __name__ == '__main__':
    tf.app.run()

    # create_test_dir('/home/galexandra/workspace/CompCars/test_images_even', '/home/galexandra/workspace/CompCars/test_MTV')

