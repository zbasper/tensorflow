
import tensorflow as tf
import zipfile as zf
import argparse

data_path = "eeeeee.zip"
label_path = "sefsef.csv"
labels = {'star': 0, 'galaxy': 1, 'qso': 2, 'unknown': 3}




def main(args):
    
    tfRecordOption = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter("xxxxx.tfrecord", tfRecordOption)
    
    with zf.ZipFile(data_path, "r") as myzip:
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Process some parameters.")
    parser.add_argument("--dataNum", type=int)
    args = parser.parse_args()
    
    main(args)
    
