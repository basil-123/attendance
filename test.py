import tensorflow as tf

model = tf.keras.models.load_model('facenet512_weights.h5')

model.summary()