## 初始化变量

`import tensorflow as tf`

### 常量

`
    scalar = tf.contant(100)
    vector = tf.constant([1,2,3,4,5])
    matrix = tf.constant([[1,2,3],[4,5,6]])
    cube_matrix = tf.constant([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]])
`

### 占位符，最后用feed_dict进行赋值

`
    tf.placeholder(dtype, shape=None, name=None)
    
    x = tf.placeholder(tf.float32, shape=(1024, 1024))
    y = tf.matmul(x, x)
    
    with tf.Session() as sess:
        rand_array = np.random.rand(1024, 1024)
        sess.run(y, feed_dict={x: rand_array})   
`
