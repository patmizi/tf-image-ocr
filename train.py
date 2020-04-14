import datetime
import os

from keras import backend as K
from keras import Input, Model, losses
from keras.layers import Conv2D, MaxPooling2D, Reshape, Lambda, Activation, concatenate, Dense, GRU, add, \
    BatchNormalization
from keras.optimizers import SGD
from keras.utils import get_file

from lib.constants import OUTPUT_DIR
from lib.image_text_generator import TextImageGenerator
from lib.visualisation_callback import ctc_lambda_func, VizCallback


def train(run_name, start_epoch, stop_epoch, img_w, build_word_count,
          max_string_len, mono_fraction, save_model_path=None):
    # Input Parameters
    img_h = 64
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    # GRU output NAN when rnn_size=512 with my GPU, but CPU or rnn_size=256 is ok.
    # Tensorflow 1.10 appears, but vanishes in 1.12!
    rnn_size = 512
    minibatch_size = 32

    # if start_epoch >= 12:
    #     minibatch_size = 8  # 32 is to large for my poor GPU

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    img_gen = TextImageGenerator(
        minibatch_size=minibatch_size,
        img_w=img_w,
        img_h=img_h,
        downsample_factor=(pool_size ** 2),
        val_split=words_per_epoch - val_words,
        build_word_count=build_word_count,
        max_string_len=max_string_len,
        mono_fraction=mono_fraction)
    act = 'relu'
    kernel_init = 'he_normal'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=None, kernel_initializer=kernel_init,
                   name='conv1')(input_data)
    inner = BatchNormalization(axis=3, scale=False, name='bn1')(inner)
    inner = Activation(activation=act)(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)

    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=None, kernel_initializer=kernel_init,
                   name='conv2')(inner)
    inner = BatchNormalization(axis=3, scale=False, name='bn2')(inner)
    inner = Activation(activation=act)(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2),
                        (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # bidirectional GRU, GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True,
                kernel_initializer=kernel_init, name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True,
                 go_backwards=True, kernel_initializer=kernel_init,
                 name='gru1_b')(inner)
    gru1_merged = concatenate([gru_1, gru_1b])

    # transforms RNN output to character activations:
    inner = Dense(img_gen.get_output_size(), kernel_initializer=kernel_init,
                  name='dense2')(gru1_merged)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels',
                   shape=[img_gen.max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
        optimizer=sgd,
        metrics=['accuracy'])
    if start_epoch > 0:
        weight_file = os.path.join(
            OUTPUT_DIR,
            os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        model.load_weights(weight_file)
        print("load_weight: ", weight_file)
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    viz_cb = VizCallback(
        run_name=run_name,
        test_func=test_func,
        text_img_gen=img_gen.next_val())

    model.fit_generator(
        generator=img_gen.next_train(),
        steps_per_epoch=(words_per_epoch - val_words) // minibatch_size,
        epochs=stop_epoch,
        validation_data=img_gen.next_val(),
        validation_steps=val_words // minibatch_size,
        callbacks=[viz_cb, img_gen],
        initial_epoch=start_epoch,
        verbose=1)

    if save_model_path:
        predict_model = Model(inputs=input_data, outputs=y_pred)
        predict_model.save(save_model_path)

if __name__ == '__main__':
    RUN_NAME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # RUN_NAME = "EnglishWord_GRU"
    SAVE_MODEL_PATH = "./model/image_ocr_word.h5"
    train(run_name=RUN_NAME,
          start_epoch=0,
          stop_epoch=12,
          img_w=128,
          build_word_count=16000,
          max_string_len=5,
          mono_fraction=1)
    # increase to wider images and start at epoch 12.
    # The learned weights are reloaded
    train(run_name=RUN_NAME,
          start_epoch=12,
          stop_epoch=20,
          img_w=512,
          build_word_count=32000,
          max_string_len=25,
          mono_fraction=1,
          save_model_path=SAVE_MODEL_PATH)
