from object_detection.utils import dataset_util
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
from collections import namedtuple
from PIL import Image
import tensorflow as tf
import os
import io


def convert_classes(classes, path, start=1):
    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=start):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))
    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    with open(path, 'w') as f:
        f.write(text)
    return text


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, label_map_dict):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmn = row['xmin'] / width
        if xmn < 0.0:
            xmn = 0.0
        elif xmn > 1.0:
            xmn = 1.0
        xmins.append(xmn)

        xmx = row['xmax'] / width
        if xmx < 0.0:
            xmx = 0.0
        elif xmx > 1.0:
            xmx = 1.0
        xmaxs.append(xmx)

        ymn = row['ymin'] / height
        if ymn < 0.0:
            ymn = 0.0
        elif ymn > 1.0:
            ymn = 1.0
        ymins.append(ymn)

        ymx = row['ymax'] / height
        if ymx < 0.0:
            ymx = 0.0
        elif ymx > 1.0:
            ymx = 1.0
        ymaxs.append(ymx)

        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map_dict[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example