import tensorflow as tf

if __name__ == '__main__':
    model = tf.keras.models.load_model('image_ocr_model.h5')
    model.summary()