
## Dataset.from_tensors和Dataset.from_tensor_slices的用法：

from_tensors将参数生成包含一个tensor的dataset
```
dataset = tf.data.Dataset.from_tensors(({'a': 32, 'b': 3}, 4))
iterator = dataset.make_one_shot_iterator()
feature, label = iterator.get_next()
with tf.Session() as sess:
    print(sess.run([feature, label]))
```
输出：[{'a': 32, 'b': 3}, 4]

```
dataset = tf.data.Dataset.from_tensors(({'a': [32,25], 'b': [3,2]}, [4,5]))
iterator = dataset.make_one_shot_iterator()
feature, label = iterator.get_next()
with tf.Session() as sess:
    print(sess.run([feature, label]))
```
输出：[{'a': array([32, 25], dtype=int32), 'b': array([3, 2], dtype=int32)}, array([4, 5], dtype=int32)]



from_tensor_slices将参数按第一维进行切分，生成包含N行tensor的dataset
```
dataset = tf.data.Dataset.from_tensor_slices(({'a': [32, 23], 'b': [3, 5]}, [4, 7]))
iterator = dataset.make_one_shot_iterator()
feature, label = iterator.get_next()
with tf.Session() as sess:
    print(sess.run([feature, label]))
```
输出：[{'a': 32, 'b': 3}, 4]

