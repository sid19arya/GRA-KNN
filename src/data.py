import tensorflow as tf
from sklearn.model_selection import train_test_split

def give_ten():
    return 10

def import_data():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.95)

    # x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size=0.95)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)

    x_train = x_train.reshape(-1, 28*28)
    x_val = x_val.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    x_val = x_val.astype('float32')
    x_train = x_train.astype('float32')   
    x_test = x_test.astype('float32')

    return (x_train, x_val, x_test, y_train, y_val, y_test)
