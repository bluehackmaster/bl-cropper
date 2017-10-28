from __future__ import print_function
import uuid

import os
from google.cloud import bigquery

import numpy as np
import argparse
import multiprocessing
import tensorflow as tf
from PIL import Image
import urllib
import redis
import json
import time
import stylelens_index
from stylelens_index.rest import ApiException
from pprint import pprint
from util import label_map_util
from object_detection.utils import visualization_utils as vis_util
from util import s3

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

AWS_BUCKET = 'bluelens-style-object'

TMP_CROP_IMG_FILE = './tmp.jpg'

CWD_PATH = os.getcwd()

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY'].replace('"', '')
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY'].replace('"', '')

OD_MODEL = os.environ['OD_MODEL']
OD_LABELS = os.environ['OD_LABELS']

NUM_CLASSES = 89

HOST_URL = 'host_url'
TAG = 'tag'
SUB_CATEGORY = 'sub_category'
PRODUCT_NAME = 'product_name'
IMAGE_URL = 'image_url'
PRODUCT_PRICE = 'product_price'
CURRENCY_UNIT = 'currency_unit'
PRODUCT_URL = 'product_url'
PRODUCT_NO = 'product_no'
MAIN = 'main'
NATION = 'nation'

REDIS_KEY_IMAGE_QUEUE = 'image_queue'
REDIS_KEY_IMAGE_INFO = 'image_info'

# Loading label map
label_map = label_map_util.load_labelmap(OD_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

api_instance = stylelens_index.ImageApi()

REDIS_SERVER = os.environ['REDIS_SERVER']
rconn = redis.StrictRedis(REDIS_SERVER)

def query():
    client = bigquery.Client.from_service_account_json(
        'BlueLens-d8117bd9e6b1.json')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(OD_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # query = 'SELECT * FROM stylelens.8seconds LIMIT 100'
    query = 'SELECT * FROM stylelens.8seconds'

    query_job = client.run_async_query(str(uuid.uuid4()), query)

    query_job.begin()
    query_job.result()  # Wait for job to complete.

    # Print the results.
    destination_table = query_job.destination
    destination_table.reload()
    for row in destination_table.fetch_data():
        image_info = stylelens_index.Image()
        image_info.host_url = str(row[0])
        image_info.tags = str(row[1]).split(',')
        image_info.product_name = str(row[3])
        image_info.image_url = str(row[4])
        image_info.product_price = str(row[5])
        image_info.currency_unit = str(row[6])
        image_info.product_url = str(row[7])
        image_info.product_no = str(row[8])
        image_info.main = int(row[9])
        image_info.nation = str(row[10])
        image_info.bucket = AWS_BUCKET
        f = urllib.request.urlopen(image_info.image_url)
        img = Image.open(f)
        image_np = load_image_into_numpy_array(img)

        show_box = False
        out_image, boxes, scores, classes, num_detections = detect_objects(image_np, sess, detection_graph, show_box)

        take_object(image_info,
                    out_image,
                    np.squeeze(boxes),
                    np.squeeze(scores),
                    np.squeeze(classes).astype(np.int32))

        if show_box:
          img = Image.fromarray(out_image, 'RGB')
          img.show()


    redis_pub()
    sess.close()

def redis_pub():
  rconn.publish('crop/done', 'DONE')


def take_object(image_info, image_np, boxes, scores, classes):
  max_boxes_to_save = 10
  min_score_thresh = .7
  if not max_boxes_to_save:
    max_boxes_to_save = boxes.shape[0]
  for i in range(min(max_boxes_to_save, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      if classes[i] in category_index.keys():
        class_name = category_index[classes[i]]['name']
        # class_code = category_index[classes[i]]['code']
        class_code = 'n0100000'
      else:
        class_name = 'na'
        class_code = 'na'
      ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())

      image_info.format = 'jpg'
      image_info.class_code = class_code

      id = crop_bounding_box(
        image_info,
        image_np,
        ymin,
        xmin,
        ymax,
        xmax,
        use_normalized_coordinates=True)
      image_info.name = id
      # print(image_info)
      save_to_storage(image_info)
      save_to_redis(image_info)

def save_to_redis(image_info):
  image = image_class_to_json(image_info)
  rconn.hset(REDIS_KEY_IMAGE_INFO, image_info.name, image)
  rconn.lpush(REDIS_KEY_IMAGE_QUEUE, image_info.name)

def image_class_to_json(image):
  image_dic = {}
  image_dic['name'] = image.name
  image_dic['host_url'] = image.host_url
  image_dic['host_code'] = image.host_code
  image_dic['tags'] = image.tags
  image_dic['format'] = image.format
  image_dic['product_name'] = image.product_name
  image_dic['parent_image_raw'] = image.parent_image_raw
  image_dic['parent_image_mobile'] = image.parent_image_mobile
  image_dic['parent_image_mobile_thumb'] = image.parent_image_mobile_thumb
  image_dic['image'] = image.image
  image_dic['class_code'] = image.class_code
  image_dic['bucket'] = image.bucket
  image_dic['storage'] = image.storage
  image_dic['product_price'] = image.product_price
  image_dic['currency_unit'] = image.currency_unit
  image_dic['product_url'] = image.product_url
  image_dic['product_no'] = image.product_no
  image_dic['main'] = image.main
  image_dic['nation'] = image.nation

  s = json.dumps(image_dic)
  print(s)
  return s

def save_to_db(image):
  try:
      api_response = api_instance.add_image(image)
      pprint(api_response)
  except ApiException as e:
      print("Exception when calling ImageApi->add_image: %s\n" % e)
  return api_response.data._id

def save_to_storage(image_info):
    print('save_to_storage')
    storage = s3.S3(AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY)
    key = os.path.join(image_info.class_code, image_info.name + '.' + image_info.format)
    is_public = False
    if image_info.main == 1:
      is_public = True
    storage.upload_file_to_bucket(AWS_BUCKET, TMP_CROP_IMG_FILE, key, is_public=is_public)
    print('save_to_storage done')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def crop_bounding_box(image_info,
                      image,
                       ymin,
                       xmin,
                       ymax,
                       xmax,
                       use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box in normalized coordinates (same below).
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    name: classname
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  im_width, im_height = image_pil.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

  # print(image_pil)
  area = (left, top, left + abs(left-right), top + abs(bottom-top))
  cropped_img = image_pil.crop(area)
  size = 300, 300
  cropped_img.thumbnail(size, Image.ANTIALIAS)
  cropped_img.save(TMP_CROP_IMG_FILE)
  # cropped_img.show()
  id = save_to_db(image_info)

  # save_image_to_file(image_pil, ymin, xmin, ymax, xmax,
  #                            use_normalized_coordinates)
  # np.copyto(image, np.array(image_pil))
  return id

def detect_objects(image_np, sess, detection_graph, show_box=True):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    if show_box:
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
    # print(image_np)
    return image_np, boxes, scores, classes, num_detections

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    query()
    while True:
      time.sleep(10)

