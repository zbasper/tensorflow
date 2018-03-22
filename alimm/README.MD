### 数据分析：
训练数据有27列，将其中的基础数据编号字段去掉。保留各项目的具体内容信息。

基础数据部分：
* instance_id：样本编号，Long
* is_trade：是否交易的标记位，Int类型；取值是0或者1，其中1 表示这条样本最终产生交易，0 表示没有交易
* item_id：广告商品编号，Long类型
* user_id：用户的编号，Long类型
* context_id：上下文信息的编号，Long类型
* shop_id：店铺的编号，Long类型



