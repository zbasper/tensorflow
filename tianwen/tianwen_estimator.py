
import tensorflow as tf
import argparse
import tianwen_data


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=100, type=int, help="batch size")
parser.add_argument("--train_steps", default=1000, type=int, help="number of training steps")

TRAIN_FILE = "tianwen_data_zlib.tfrecord"
TEST_FILE = "tianwen_data_eval_zlib.tfrecord"
PREDICT_FILE = "tianwen_data_test_zlib.tfrecord"


def main():
    
    args = parser.parse_args()
    batch_size = args.batch_size
    train_steps = args.train_steps
    
    my_feature_columns = []
    my_feature_columns.append(tf.feature_column.numeric_column(key='feature', shape=2600))
    
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifer = tf.estimator.DNNClassifier (
        feature_columns = my_feature_columns,
        hidden_units = [10, 10],
        n_classes = 3
    )
    
    dataset_train = tianwen_data.load_data(TRAIN_FILE)
    # Train the model
    classifier.train (
        input_fn = lambda: tianwen_data.train_input_fn(dataset_train, batch_size),
        steps = train_steps
    )
    
    dataset_test = tianwen_data.load_data(TEST_FILE)
    #Evaluate the model.
    eval_result = classifier.evaluate (
        input_fn = lambda: tianwen_data.eval_input_fn(dataset_test, batch_size)
    )
    print("Test set accuracy: (accuracy:0.3)\n".format(**eval_result))
    
    dataset_predict = tianwen_data.load_data(PREDICT_FILE)
    #Generate predictions from the model
    predictions = classifier.predict (
        input_fn = lambda: tianwen_data.eval_input_fn(dataset_predict)
    )
    
    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        
if __name__ == "__main__":
    tf.logging.set_verbosity(tf.loggin.INFO)
    tf.app.run(main)
    
    
