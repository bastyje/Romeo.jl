import tensorflow as tf
import tensorflow_datasets as ds
import time
import psutil
from memory_profiler import profile
import os


class PerformanceLogger(tf.keras.callbacks.Callback):
    __timer_start = None
    __train_timer_start = None
    __memory_usage = None
    __train_memory_usage = None

    def on_epoch_begin(self, epoch, logs=None):
        self.__timer_start = time.time()
        self.__memory_usage = psutil.Process().memory_full_info().pss
        return super().on_epoch_end(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.__timer_start
        memory_usage = psutil.Process(os.getpid()).memory_full_info().pss - self.__memory_usage
        self.__log(memory_usage, elapsed_time)
        return super().on_epoch_end(epoch, logs)

    @staticmethod
    def __log(memory_usage, elapsed_time):
        print('\n\n==== Performance Metrics ====')
        print(f'Epoch took {elapsed_time:.2f}s and used {memory_usage / 1024**3:.2f}GiB of memory\n')


def preprocess(image, label):
    image = tf.reshape(image, (4, 196))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


settings = {
    'batch_size': 100,
    'epochs': 5,
    'learning_rate': 0.015
}

(train_data, test_data), info = ds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

train_data = train_data.map(preprocess).shuffle(len(train_data)).batch(settings['batch_size'])
test_data = test_data.map(preprocess).batch(settings['batch_size'])

net = tf.keras.Sequential([
    tf.keras.layers.RNN(
        tf.keras.layers.SimpleRNNCell(196),
        unroll=True
    ),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

# net = tf.keras.Sequential([
#     tf.keras.layers.Reshape((4, 196), input_shape=(784,)),  # Reshape input to 4 time steps of 196 features each
#     tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=False),  # Vanilla RNN layer
#     tf.keras.layers.Dense(10, activation='softmax')  # Dense layer with softmax activation
# ])

net.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=settings['learning_rate']),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


@profile
def train(model):
    model.fit(
        train_data,
        epochs=settings['epochs'],
        batch_size=settings['batch_size'],
        callbacks=[PerformanceLogger()]
    )


train(net)
net.evaluate(test_data)
