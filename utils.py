from sklearn.model_selection import (
    StratifiedKFold, StratifiedShuffleSplit
)

import time
import numpy as np
import pandas as pd
from skimage.transform import sk_rescale
from keras import backend as K  # noqa
import keras.datasets as datasets

from models import PatchedKerasClassifier

from patch._sklearn import cross_val_score

import matplotlib.pyplot as plt

available_datasets = [
    'mnist', 'fashion_mnist', 'cifar',
    'cifar10', 'cifar100', 'imdb', 'reuters'
]


class ANSIColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def deprecated(f):
    def _inner(*args, **kargs):
        print(
            ANSIColors.WARNING +
            '[WARNING] {}: this method is deprecated.'.format(f.__name__) +
            ANSIColors.ENDC
        )
        return f(*args, **kargs)
    return _inner


def concat_sample(*array_pairs, axis=0):
    """
    Concat every array pairs.

    examples:
        x_all = concat_sample((x_train, x_test))
        x_all, y_all = concat_sample((x_train, x_test), (y_train, y_test))
    """

    outs = []
    for arr1, arr2 in array_pairs:
        outs.append(np.concatenate((arr1, arr2), axis=axis))
    return tuple(outs)


def visualize_feature(transformer, X, y, title=None, figsize=(18, 10),
                      ylim=[-1.5, 1.5], xlim=[-1.5, 1.5],
                      alpha=0.8, legend_loc='upper right', markersize=12):

    # hard coding to 2d space
    encoded = transformer.transform(X)
    visual_map = pd.DataFrame({'x': encoded[:, 0], 'y': encoded[:, 1], 'label': y})
    fig = plt.figure(figsize=figsize)
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    ax = fig.gca()

    for key, df in visual_map.groupby('label'):
        ax.scatter(df.x, df.y, color=plt.cm.tab10_r(int(key)), alpha=alpha, label=key, marker='${}$'.format(key), s=markersize)
    ax.legend(loc=legend_loc)
    if title is not None:
        ax.set_title(title)
    plt.close(fig)

    return fig


def down_sampling(X, y, num=1000):
    cv = StratifiedShuffleSplit(n_splits=1, test_size=num, random_state=int(time.time()))

    sampling_index = None
    for _, test_index in cv.split(X, y):
        sampling_index = test_index
        break

    return X[sampling_index].copy(), y[sampling_index].copy()


def load_keras_dataset(name, model_type='mlp', feature_type='image', rescale=None):

    if name not in available_datasets:
        msg = '''
            Can\'t find dataset: {} from keras.datasets,\n
            available datasets are: {}.\n
        '''.format(name, ', '.join(available_datasets))
        raise AttributeError(msg)

    ds = datasets.__getattribute__(name)

    (x_train, y_train), (x_test, y_test) = ds.load_data()

    # NOTE: lazy to check binary labels in (0, 1) ...
    num_classes = len(np.unique(y_train))
    num_classes = 1 if num_classes == 2 else num_classes

    # NOTE: inconsist tensor shape
    num_train, *train_tensor = x_train.shape
    num_test, *_ = x_test.shape

    if len(train_tensor) == 2:
        img_rows, img_cols = train_tensor
        channels = 1
    elif len(train_tensor) == 3:
        if K.image_data_format() == 'channels_last':
            img_rows, img_cols, channels = train_tensor
        else:
            channels, img_rows, img_cols = train_tensor

    if feature_type in ('image',):

        if model_type in ('mlp',):
            input_shape = img_rows * img_cols * channels
            x_train = x_train.reshape(num_train, input_shape)
            x_test = x_test.reshape(num_test, input_shape)

        elif model_type in ('cnn',):
            if K.image_data_format() == 'channels_first':
                x_train = x_train.reshape(num_train, channels, img_rows, img_cols)
                x_test = x_test.reshape(num_test, channels, img_rows, img_cols)
                input_shape = (channels, img_rows, img_cols)
            else:
                x_train = x_train.reshape(num_train, img_rows, img_cols, channels)
                x_test = x_test.reshape(num_test, img_rows, img_cols, channels)
                input_shape = (img_rows, img_cols, channels)

            if isinstance(rescale, float):
                x_train, (img_rows, img_cols) = _batch_rescale(x_train, rescale)
                x_test, (_, _) = _batch_rescale(x_test, rescale)

                input_shape = (
                    (channels, img_rows, img_cols)
                    if K.image_data_format() == 'channels_first'
                    else (img_rows, img_cols, channels)
                )

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.0
        x_test /= 255.0

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return (x_train, y_train), (x_test, y_test), (input_shape, num_classes)


def _batch_rescale(X, scale):
    # X should be 4D tensor
    shape = X.shape
    if len(shape) != 4:
        raise ValueError('Input dimension must be 4, i.e., (batch_size, nrow, ncol, channels)')

    num_samples = shape[0]
    img_rows, img_cols, channels = (
        (shape[2], shape[3], shape[1]) if K.image_data_format() == 'channels_first' else shape[1:]
    )
    new_rows, new_cols, _ = sk_rescale(X[0, :].reshape(img_rows, img_cols, channels), scale, mode='constant')

    new_X = np.zeros(num_samples, new_rows, new_cols, channels). # noqa

    for i, x in enumerate(X):
        new_X[i, :] = sk_rescale(X[i, :].reshape(img_rows, img_cols, channels), scale, mode='constant')

    return (
        new_X.transpose(0, 3, 1, 2)
        if K.image_data_format() == 'channels_first'
        else new_X,
        (new_rows, new_cols)
    )


def decode_from_latent(ax, decoder,
                       grid_x=np.linspace(-1.5, 1.5, 16),
                       grid_y=np.linspace(-1.5, 1.5, 16),
                       spacing=1):
    pass


@deprecated
def evaluate_keras_model(dataset_name, build_fn, X, y,
                         epochs=1, batch_size=32,
                         n_fold=2, shuffle=True,
                         random_seed=None, verbose=1):
    build_fn().summary()
    model = PatchedKerasClassifier(build_fn=build_fn, epochs=epochs,
                                   batch_size=batch_size, verbose=verbose)
    results = evaluate_model(model, X, y)
    print('\n\nMetrics of {} with {}: {}\n\n'.format(model.build_fn.__name__, dataset_name, results))

    return results


@deprecated
def evaluate_model(estimator, X, y,
                   n_fold=2, shuffle=True,
                   random_seed=None, verbose=1):
    # evaluate using 2-fold cross validation
    cv = StratifiedKFold(n_splits=n_fold, shuffle=shuffle, random_state=random_seed)
    results = cross_val_score(estimator, X, y, cv=cv)
    return results
