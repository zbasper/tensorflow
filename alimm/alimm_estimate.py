import tensorflow as tf
import pandas as pd

TRAIN_FILE = "round1_ijcai_18_train_20180301.txt"
TEST_FILE = "round1_ijcai_18_test_a_20180301.txt"


def features():
    
    #广告商品品牌编号
    item_brand_id = tf.feature_column.categorical_column_with_vocabulary_file(
        key = "item_brand_id",
        vacabulary_file = "../features/item_brand_id.csv")
    #广告商品城市编号
    item_city_id = tf.feature_column.categorical_column_with_vocabulary_file(
        key = "item_city_id",
        vacabulary_file = "../features/item_city_id.csv")
    #广告商品价格等级
    item_price = tf.feature_column.numerical_column("item_price_level", dtype=tf.int8)
    #广告商品销量等级
    item_sales = tf.feature_column.numerical_column("item_sales_level", dtype=tf.int8)
    #广告商品收藏次数等级
    item_collected_level = tf.feature_column.numerical_column("item_collected_level", dtype=tf.int8)
    #广告商品展示次数等级
    item_pv_level = tf.feature_column.numerical_column("item_pv_level", dtype=tf.int8)
    
    #用户性别编号
    user_gender_id = tf.feature_column.numerical_column("user_gender_Id", dtype=tf.int8)
    #用户年龄等级
    user_age_level = tf.feature_column.numerical_column("user_age_level", dtype=tf.int16)
    #用户职业编号
    user_occupation_id = tf.feature_column.numerical_column("user_occupation_id", dtype=tf.int16)
    #用户星级编号
    user_star_level = tf.feature_column.numerical_column("user_star_level", dtype=tf.int16)
    
    #广告商品展示时间
    #context_timestamp 
    #广告商品展示页面编号
    context_page_id = tf.feature.column.numerical_column("context_page_id", dtype=tf.int16)
    #查询词类目属性
    #predict_category_property
    
    #店铺评价数量等级
    shop_review_num_level = tf.feature_column.numerical_column("shop_review_num_level", dtype=tf.int8)
    #店铺好评率
    shop_review_positive_rate = tf.feature.column.numerical_column("shop_review_positive_rate")
    #店铺星级编号
    shop_star_level = tf.fetaure_column.numerical_column("shop_star_level", dtype=tf.int16)
    #店铺服务态度评分
    shop_score_service = tf.feature_column.numerical_column("shop_score_service")
    #店铺物流服务评分
    shop_score_delivery = tf.feature_column.numerical_column("shop_score_delivery")
    #店铺描述相符评分
    shop_score_description = tf.feature_column.numerical_column("shop_score_description")
    
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

def train_input_fn(_features, _labels, _batch_size):
    
    dataset = tf.data.Dataset.from_tensor_slices((dict(_features), _labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.batch(_batch_size)
    
    return dataset

def eval_input_fn(_features, _labels, _batch_size):
    _feature = dict(_features)
    if _labels == None:
        inputs = _features
    else:
        inputs = (_features, _labels)
    
    dateset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(_batch_size)
    
    return dataset



def main(argv):
    
    
    train = pd.read_csv(TRAIN_FILE, sep=' ')
    train = train.pop(['instance_id', 'item_id', 'user_id', 'shop_id', 'context_id', 
                       'predict_category_property', 'item_category_list', 'item_property_list'], axis=1)
    train_x, train_y = train, train.pop("is_trade")
    
    classifier = tf.estimator.DNNClassifier (
        feature_columns = features(),
        hidden_units = [10, 10],
        n_classes = 2
    )
    
    # Train the model
    classifier.train (
        input_fn = lambda: train_input_fn(feature, label, batch_size),
        steps = train_steps
    )
    
    #Evaluate the model.
    eval_result = classifier.evaluate (
        input_fn = lambda: eval_input_fn(dataset_test, batch_size)
    )
    print("Test set accuracy: (accuracy:0.3)\n".format(**eval_result))
    
    #Generate predictions from the model
    predictions = classifier.predict (
        input_fn = lambda: eval_input_fn(dataset_predict)
    )

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
    
    
    
    
