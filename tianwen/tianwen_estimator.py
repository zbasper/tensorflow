
import tensorflow as tf
import argparse
import tianwen_data


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=100, type=int, help="batch size")
parser.add_argument("--train_steps", default=1000, type=int, help="number of training steps")


def input_evaluation_set():

def main():
    
    args = parser.parse_args()
    batch_size = args.batch_size
    train_steps = args.train_steps
    
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifer = tf.estimator.DNNClassifier (
        feature_columns = my_feature_columns,
        hidden_units = [10, 10],
        n_classes = 3
    )
    
    # Train the model
    classifier.train (
        input_fn = tianwen_data.train_input_fn(),
        steps = train_steps
    )
    
    #Evaluate the model.
    eval_result = classifier.evaluate (
        input_fn = tianwen_data.eval_input_fn()
    )
    
    print("Test set accuracy: (accuracy:0.3)\n".format(**eval_result))
    
    #Generate predictions from the model
    
    predictions = classifier.predict (
        input_fn = tianwen_data.eval_input_fn()
    )
    

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.loggin.INFO)
    tf.app.run(main)
    
    
