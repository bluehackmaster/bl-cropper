from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from PIL import Image
import urllib
import redis
import json
import logging
from threading import Timer
import time
import signal
import stylelens_index
from stylelens_index.rest import ApiException
from bluelens_spawning_pool import spawning_pool
from pprint import pprint
from util import label_map_util
from object_detection.utils import visualization_utils as vis_util
from util import s3

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

AWS_BUCKET = 'bluelens-style-object'

REDIS_IMAGE_INDEX_QUEUE = 'bl:image:index:queue'

TMP_CROP_IMG_FILE = './tmp.jpg'

CWD_PATH = os.getcwd()

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY'].replace('"', '')
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY'].replace('"', '')

OD_MODEL = os.environ['OD_MODEL']
OD_LABELS = os.environ['OD_LABELS']
SPAWN_ID = os.environ['SPAWN_ID']

NUM_CLASSES = 89

HOST_URL = 'host_url'
TAGS = 'tags'
SUB_CATEGORY = 'sub_category'
PRODUCT_NAME = 'product_name'
IMAGE = 'image'
PRODUCT_PRICE = 'product_price'
CURRENCY_UNIT = 'currency_unit'
PRODUCT_URL = 'product_url'
PRODUCT_NO = 'product_no'
MAIN = 'main'
NATION = 'nation'

REDIS_IMAGE_CROP_QUEUE = 'bl:image:crop:queue'

STR_BUCKET = "bucket"
STR_STORAGE = "storage"
STR_CLASS_CODE = "class_code"
STR_NAME = "name"
STR_FORMAT = "format"

# Loading label map
label_map = label_map_util.load_labelmap(OD_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

api_instance = stylelens_index.ImageApi()

REDIS_SERVER = os.environ['REDIS_SERVER']
REDIS_PASSWORD = os.environ['REDIS_PASSWORD']

rconn = redis.StrictRedis(REDIS_SERVER, port=6379, password=REDIS_PASSWORD)

logging.basicConfig(filename='./log/main.log', level=logging.DEBUG)
heart_bit = True

def job():
  detection_graph = tf.Graph()
  with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(OD_MODEL, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')

      sess = tf.Session(graph=detection_graph)

  redis_pub('START')
  def items():
    while True:
      yield rconn.blpop([REDIS_IMAGE_CROP_QUEUE])


  def request_stop(signum, frame):
    print('stopping')
    stop_requested = True
    rconn.connection_pool.disconnect()
    print('connection closed')

  signal.signal(signal.SIGINT, request_stop)
  signal.signal(signal.SIGTERM, request_stop)

  for item in items():
    key, image_data = item
    print(item)
    image_info = {}
    if type(image_data) is str:
      image_info = json.loads(image_data)
    elif type(image_data) is bytes:
      image_info = json.loads(image_data.decode('utf-8'))

    print('1')
    image = stylelens_index.Image()
    print('2')

    image.name = image_info['name']
    image.host_url = image_info['host_url']
    image.host_code = image_info['host_code']
    image.tags = image_info['tags']
    image.format = image_info['format']
    image.product_name = image_info['product_name']
    image.parent_image_raw = image_info['parent_image_raw']
    image.parent_image_mobile = image_info['parent_image_mobile']
    image.parent_image_mobile_thumb = image_info['parent_image_mobile_thumb']
    image.image = image_info['image']
    image.class_code = image_info['class_code']
    image.bucket = image_info['bucket']
    image.storage = image_info['storage']
    image.product_price = image_info['product_price']
    image.currency_unit = image_info['currency_unit']
    image.product_url = image_info['product_url']
    image.product_no = image_info['product_no']
    image.main = image_info['main']
    image.nation = image_info['nation']

    print('3')
    f = urllib.request.urlopen(image.image)
    img = Image.open(f)
    image_np = load_image_into_numpy_array(img)
    # print(image_np)

    show_box = False
    out_image, boxes, scores, classes, num_detections = detect_objects(image_np, sess, detection_graph, show_box)

    print('4')
    take_object(image,
                out_image,
                np.squeeze(boxes),
                np.squeeze(scores),
                np.squeeze(classes).astype(np.int32))

    print('5')
    if show_box:
      img = Image.fromarray(out_image, 'RGB')
      img.show()

    global  heart_bit
    heart_bit = True

def check_health():
  global  heart_bit
  print('check_health: ' + str(heart_bit))
  logging.debug('check_health: ' + str(heart_bit))
  if heart_bit == True:
    heart_bit = False
    Timer(100, check_health, ()).start()
  else:
    exit()

def exit():
  print('exit: ' + SPAWN_ID)
  logging.debug('exit: ' + SPAWN_ID)
  data = {}
  data['namespace'] = 'index'
  data['id'] = SPAWN_ID
  spawn = spawning_pool.SpawningPool()
  spawn.setServerUrl(REDIS_SERVER)
  spawn.setServerPassword(REDIS_PASSWORD)
  spawn.delete(data)

def redis_pub(message):
  rconn.publish('crop', message)


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
  rconn.lpush(REDIS_IMAGE_INDEX_QUEUE, image)

def image_class_to_json(image):
  image_info = {}
  image_info['name'] = image.name
  image_info['host_url'] = image.host_url
  image_info['host_code'] = image.host_code
  image_info['tags'] = image.tags
  image_info['format'] = image.format
  image_info['product_name'] = image.product_name
  image_info['parent_image_raw'] = image.parent_image_raw
  image_info['parent_image_mobile'] = image.parent_image_mobile
  image_info['parent_image_mobile_thumb'] = image.parent_image_mobile_thumb
  image_info['image'] = image.image
  image_info['class_code'] = image.class_code
  image_info['bucket'] = image.bucket
  image_info['storage'] = image.storage
  image_info['product_price'] = image.product_price
  image_info['currency_unit'] = image.currency_unit
  image_info['product_url'] = image.product_url
  image_info['product_no'] = image.product_no
  image_info['main'] = image.main
  image_info['nation'] = image.nation

  s = json.dumps(image_info)
  print(s)
  return s

def save_to_db(image):
  print('save_to_db 0')
  try:
      api_response = api_instance.add_image(image)
      pprint(api_response)
  except ApiException as e:
      print("Exception when calling ImageApi->add_image: %s\n" % e)
  print('save_to_db 1')

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
  Timer(60, check_health, ()).start()
  job()

