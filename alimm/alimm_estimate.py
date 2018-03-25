#!/usr/bin/env python

import tensorflow as tf
import pandas as pd
import argparse
import sys

TRAIN_FILE = "round1_ijcai_18_train_20180301.txt"
TEST_FILE = "round1_ijcai_18_test_a_20180301.txt"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')



#特征函数
def features():
    
    #广告商品品牌编号
    item_brand_id = tf.feature_column.categorical_column_with_vocabulary_file(
        key = "item_brand_id",
        vocabulary_file = "/features/item_brand_id.csv",
        dtype = tf.int64)
    item_brand_id = tf.feature_column.embedding_column(
        categorical_column=item_brand_id,
        dimension=7)
    #广告商品城市编号
    item_city_id = tf.feature_column.categorical_column_with_vocabulary_file(
        key = "item_city_id",
        vocabulary_file = "/features/item_city_id.csv",
        dtype = tf.int64)
    item_city_id = tf.feature_column.embedding_column(
        categorical_column=item_city_id,
        dimension=4)
    
    #广告商品价格等级
    item_price = tf.feature_column.numeric_column("item_price_level", dtype=tf.int8)
    #广告商品销量等级
    item_sales = tf.feature_column.numeric_column("item_sales_level", dtype=tf.int8)
    #广告商品收藏次数等级
    item_collected_level = tf.feature_column.numeric_column("item_collected_level", dtype=tf.int8)
    #广告商品展示次数等级
    item_pv_level = tf.feature_column.numeric_column("item_pv_level", dtype=tf.int8)
    
    #用户性别编号
    user_gender_id = tf.feature_column.numeric_column("user_gender_id", dtype=tf.int8)
    #用户年龄等级
    user_age_level = tf.feature_column.numeric_column("user_age_level", dtype=tf.int16)
    #用户职业编号
    user_occupation_id = tf.feature_column.numeric_column("user_occupation_id", dtype=tf.int16)
    #用户星级编号
    user_star_level = tf.feature_column.numeric_column("user_star_level", dtype=tf.int16)
    
    #广告商品展示时间
    #context_timestamp 
    #广告商品展示页面编号
    context_page_id = tf.feature_column.numeric_column("context_page_id", dtype=tf.int16)
    #查询词类目属性
    #predict_category_property
    
    #店铺评价数量等级
    shop_review_num_level = tf.feature_column.numeric_column("shop_review_num_level", dtype=tf.int8)
    #店铺好评率
    shop_review_positive_rate = tf.feature_column.numeric_column("shop_review_positive_rate")
    #店铺星级编号
    shop_star_level = tf.feature_column.numeric_column("shop_star_level", dtype=tf.int16)
    #店铺服务态度评分
    shop_score_service = tf.feature_column.numeric_column("shop_score_service")
    #店铺物流服务评分
    shop_score_delivery = tf.feature_column.numeric_column("shop_score_delivery")
    #店铺描述相符评分
    shop_score_description = tf.feature_column.numeric_column("shop_score_description")
    
    my_features = [
        item_brand_id,
        item_city_id,
        item_price,
        item_sales,
        item_collected_level,
        item_pv_level,
        user_gender_id,
        user_age_level,
        user_occupation_id,
        user_star_level,
        context_page_id,
        shop_review_num_level,
        shop_review_positive_rate,
        shop_star_level,
        shop_score_service,
        shop_score_delivery,
        shop_score_description
    ]
    
    return my_features

#训练函数
def train_input_fn(_features, _labels, _batch_size):
    
    dataset = tf.data.Dataset.from_tensor_slices((dict(_features), _labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.batch(_batch_size)
    
    return dataset

#执行函数
def eval_input_fn(_features, _labels, _batch_size):
    _features = dict(_features)
    if _labels is None:
        inputs = _features
    else:
        inputs = (_features, _labels)
    
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(_batch_size)
    
    return dataset

#主函数
def main(argv):
    
    args = parser.parse_args()
    batch_size = args.batch_size
    train_steps = args.train_steps
    
    train = pd.read_csv(TRAIN_FILE, sep=' ')
    train = train.drop(['instance_id', 'item_id', 'user_id', 'shop_id', 'context_id', 
                       'predict_category_property', 'item_category_list', 'item_property_list'], axis=1)
    total = train['is_trade'].count()
    train_num = int(total * 0.8)
    train_x, train_y = train.iloc[0:train_num, :-1], train["is_trade"][0:train_num]
    test_x, test_y = train.iloc[train_num:total, :-1], train["is_trade"][train_num:total]
    print(test_x.tail(3))
    
    predict = pd.read_csv(TEST_FILE, sep=' ')
    predict_x = predict.drop(['instance_id', 'item_id', 'user_id', 'shop_id', 'context_id', 
                       'predict_category_property', 'item_category_list', 'item_property_list'], axis=1)
    
    
    classifier = tf.estimator.DNNClassifier (
        feature_columns = features(),
        hidden_units = [10, 10],
        n_classes = 2
    )
    
    # Train the model
    classifier.train (
        input_fn = lambda: train_input_fn(train_x, train_y, batch_size),
        steps = train_steps
    )
    
    #Evaluate the model.
    eval_result = classifier.evaluate (
        input_fn = lambda: eval_input_fn(test_x, test_y, batch_size)
    )
    print("Test set accuracy: (accuracy:0.3)\n".format(**eval_result))
    
    #Generate predictions from the model
    predictions = classifier.predict (
        input_fn = lambda: eval_input_fn(predict_x, _labels=None, _batch_size=batch_size)
    )

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
    
    
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main) 
