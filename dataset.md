
dataset = tf.data.Dataset.from_tensors(({'a': 32, 'b': 3}, 4))
iterator = dataset.make_one_shot_iterator()
feature, label = iterator.get_next()
with tf.Session() as sess:
    print(sess.run([feature, label]))

[{'a': 32, 'b': 3}, 4]
