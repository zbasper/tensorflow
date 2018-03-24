import tensorflow as tf


my_features=[]

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
    
    
    
