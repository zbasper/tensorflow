
import tensorflow as tf
import zipfile as zf
import pandas as pd
import argparse

data_path = "tianwen_train_data.zip"
label_path = "tianwen_train_index.csv"
labels = {'star': 0, 'galaxy': 1, 'qso': 2, 'unknown': 3}

def main(args):
    
    totalNum = args.dataNum
    i = 0
    j = 0
    
    df = pd.read_csv(label_path)
    df['type'] = df['type'].replace(['star', 'galaxy', 'qso', 'unknown'], [0,1,2,3])
        
    tfRecordOption = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter("tianwen_train_data_zlib.tfrecord", tfRecordOption)
    
    with zf.ZipFile(data_path, "r") as myzip:
        filenames = myzip.namelist()
        for fn in filenames:
            with myzip.open(fn) as myfile:
                for line in myfile:
                    if totalNUm == None:
                        pass
                    elif i > totalNum :
                        j = 1
                        break
                        
                    example = tf.train.Example(features=tf.train.Features(feature={
                            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[df.type[i]])),
                            "feature": tf.train.Feature(bytes_list=tf.train.BytesList(value=[line]))
                    }))
                    writer.write(example.SerializeToString())    
                    i += 1
                    
            if j == 1:
                break
                
    
    writer.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Process some parameters.")
    parser.add_argument("--dataNum", type=int)
    args = parser.parse_args()
    
    main(args)
    
