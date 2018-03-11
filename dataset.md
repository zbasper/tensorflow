
## Dataset.from_tensors和Dataset.from_tensor_slices的用法：

```
dataset = tf.data.Dataset.from_tensors(({'a': 32, 'b': 3}, 4))
iterator = dataset.make_one_shot_iterator()
feature, label = iterator.get_next()
with tf.Session() as sess:
    print(sess.run([feature, label]))
```
输出：[{'a': 32, 'b': 3}, 4]


```
dataset = tf.data.Dataset.from_tensor_slices(({'a': [32, 23], 'b': [3, 5]}, [4, 7]))
iterator = dataset.make_one_shot_iterator()
feature, label = iterator.get_next()
with tf.Session() as sess:
    print(sess.run([feature, label]))
```
输出：[{'a': 32, 'b': 3}, 4]

