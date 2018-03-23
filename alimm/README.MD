### 数据分析：
训练数据有27列，将其中的基础数据编号字段去掉。保留各项目的具体内容信息。

基础数据部分：
* instance_id：样本编号，Long
* is_trade：是否交易的标记位，Int类型；取值是0或者1，其中1 表示这条样本最终产生交易，0 表示没有交易
* item_id：广告商品编号，Long类型
* user_id：用户的编号，Long类型
* context_id：上下文信息的编号，Long类型
* shop_id：店铺的编号，Long类型

广告商品信息：
* item_category_list：广告商品的的类目列表，String类型；从根类目（最粗略的一级类目）向叶子类目（最精细的类目）依次排列，数据拼接格式为 "category_0;category_1;category_2"，其中 category_1 是 category_0 的子类目，category_2 是 category_1 的子类目
> 使用最后一个类目做为分析对象

* item_property_list：广告商品的属性列表，String类型；数据拼接格式为 "property_0;property_1;property_2"，各个属性没有从属关系
* item_brand_id：广告商品的品牌编号，Long类型
* item_city_id：广告商品的城市编号，Long类型
* item_price_level：广告商品的价格等级，Int类型；取值从0开始，数值越大表示价格越高，价格分布符合正态分布。
* item_sales_level：广告商品的销量等级，Int类型；取值从0开始，数值越大表示销量越大
* item_collected_level：广告商品被收藏次数的等级，Int类型；取值从0开始，数值越大表示被收藏次数越大
* item_pv_level：广告商品被展示次数的等级，Int类型；取值从0开始，数值越大表示被展示次数越大

