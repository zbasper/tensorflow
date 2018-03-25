import tensorflow as tf




def features():
    
    my_features = []
    
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


    
