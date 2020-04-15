import tensorflow as tf

if __name__ == '__main__':
    model = tf.keras.models.load_model('model/image_ocr_word-01.h5')
    model.summary()