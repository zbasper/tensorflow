import tensorflow as tf

LABELS = ['labels', 'feature']
TRAIN_FILE = "tianwen_data_zlib.tfrecord"
TEST_FILE = "tianwen_data_test_zlib.tfrecord"


def _parse_function(example_proto):
    features = {"feature": tf.FixedLenFeature([], tf.string, default_value=""),
                "label": tf.FixedLenFeature([], tf.int64, default_value=0)}
    parse_features = tf.parse_single_example(example_proto, features)
    feature = parse_features["feature"]
    feature = tf.reshape(feature, [-1])
    feature = tf.string_split(feature, ',').values
    feature = tf.string_to_number(feature)

    return feature, parse_features["label"]


def load_data():
  
    dataset = tf.data.TFRecordDataset(TRAIN_FILE, "ZLIB")
    dataset = dataset.map(_parse_function)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    feature, label = next_element
    
    return (feature, label)
