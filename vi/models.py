import copy
import types

import numpy as np

import keras
from keras.layers import (  # noqa
    Dense, Dropout, Flatten,
    Conv2D, MaxPooling2D,
    Input, Lambda, Reshape
)

from keras.models import (
    Model, Sequential
)

from keras.utils import to_categorical

from keras.wrappers.scikit_learn import (
    BaseWrapper as KWrapper
)

from keras import backend as K  # noqa


class _BaseWrapper(KWrapper):

    def fit(self, x, mixed_y, **kwargs):

        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        # TODO: wrapper patch, key hard coding?
        y = mixed_y['classifier'] if isinstance(mixed_y, dict) else mixed_y
        loss_name = self.model.loss['classifier'] if isinstance(self.model.loss, dict) else self.model.loss

        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

            # TODO: wrapper patch, should I copy? key hard coding?
            if isinstance(mixed_y, dict):
                mixed_y['classifier'] = y
            else:
                mixed_y = y

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)

        history = self.model.fit(x, mixed_y, **fit_args)

        return history


class PatchedKerasClassifier(_BaseWrapper):

    def fit(self, x, mixed_y, **kwargs):
        # wrapper patch
        y = mixed_y['classifier'] if isinstance(mixed_y, dict) else mixed_y

        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)

            # TODO: wrapper patch, should I copy? key hard coding?
            if isinstance(mixed_y, dict):
                mixed_y['classifier'] = y
            else:
                mixed_y = y

        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))

        self.n_classes_ = len(self.classes_)
        return super(PatchedKerasClassifier, self).fit(x, mixed_y, **kwargs)

    def score(self, x, mixed_y, **kwargs):

        # TODO: wrapper patch, key hard coding?
        y = mixed_y['classifier'] if isinstance(mixed_y, dict) else mixed_y
        loss_name = self.model.loss['classifier'] if isinstance(self.model.loss, dict) else self.model.loss

        y = np.searchsorted(self.classes_, y)
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

            # TODO: wrapper patch, should I copy? key hard coding?
            if isinstance(mixed_y, dict):
                mixed_y['classifier'] = y
            else:
                mixed_y = y

        outputs = self.model.evaluate(x, mixed_y, **kwargs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc' or name == 'classifier_acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')


class BaseClassifier(object):
    __name__ = 'base'

    def __init__(self, input_shape, num_classes, activation='sigmoid'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation
        # if self.num_classes < 2:
        #   raise ValueError('Number of classes must greater than 2.')

        self.output_activation = 'sigmoid' if self.num_classes == 1 else 'softmax'

    @property
    def xent_loss(self):
        if self.num_classes >= 2:
            return keras.losses.categorical_crossentropy
        else:
            return keras.losses.binary_crossentropy

    def __call__(self, do_compile=True):
        # TODO: should I cache model ?
        model = self.create()

        if do_compile:
            return self.compile(model)
        else:
            return model

    def getnet(self, name):

        try:
            return self.__getattribute__(name)
        except Exception:
            raise AttributeError('Network \'{}\' has no subnet named \'{}\'.'.format(self.__name__, name))

    def create(self):
        raise NotImplementedError


class VariationalMixin(object):

    def vloss(self, z_mean, z_log_var, penalty=1.0):
        def loss(y_true, y_pred):
            xent_loss = K.sum(K.square(K.batch_flatten(y_pred) - K.batch_flatten(y_true)), axis=-1)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return penalty * (xent_loss + kl_loss)
        return loss

    def sampling(self, x, latent_dim, sigma=1.0):
        z_mean, z_log_var = x
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0.0, stddev=sigma)
        return z_mean + K.exp(z_log_var / 2) * epsilon


class MlpClassifier(BaseClassifier):
    __name__ = 'mlp'

    def __init__(self, input_shape, num_classes, activation='sigmoid',
                 latent_dim=16):
        super().__init__(input_shape, num_classes, activation='sigmoid')
        self.latent_dim = latent_dim

    def create(self):
        x = Input(shape=(self.input_shape,))
        latent = Dense(self.latent_dim, input_dim=self.input_shape, activation=self.activation)(x)
        classifier = Dense(self.num_classes, activation=self.output_activation)(latent)

        encoder = Model(x, latent)
        model = model = Model(x, classifier)

        model.compile(loss=self.xent_loss,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self.encoder = encoder
        self.classifier = model

        return model


class ZMlpClassifier(BaseClassifier, VariationalMixin):
    __name__ = 'mlp_with_z'

    def __init__(self, input_shape, num_classes, activation='sigmoid',
                 latent_dim=16, vi_penalty=1.0, sigma=1.0):
        super().__init__(input_shape, num_classes, activation='sigmoid')
        self.latent_dim = latent_dim
        self.vi_penalty = vi_penalty
        self.sigma = sigma

    def create(self):
        x = Input(shape=(self.input_shape,))
        z_mean = Dense(self.latent_dim, activation=self.activation)(x)
        z_log_var = Dense(self.latent_dim, activation=self.activation)(x)

        z = Lambda(self.sampling, output_shape=(self.latent_dim,),
                   arguments=dict(latent_dim=self.latent_dim, sigma=self.sigma))([z_mean, z_log_var])
        yh = Dense(self.num_classes, activation=self.output_activation, name='classifier')(z)
        xh = Dense(self.input_shape, activation='sigmoid', name='decoder')(z)

        model = Model(inputs=x, outputs=[yh, xh])

        model.compile(loss=dict(classifier=self.xent_loss,
                                decoder=self.vloss(z_mean, z_log_var, penalty=self.vi_penalty)),
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        encoder = Model(x, z)
        decoder = Model(x, xh)
        classifier = Model(x, yh)

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

        return model


class ZCnnClassifier(BaseClassifier, VariationalMixin):
    __name__ = 'cnn_with_z'

    def __init__(self, input_shape, num_classes, activation='sigmoid',
                 latent_dim=16, vi_penalty=1.0, sigma=1.0):
        super().__init__(input_shape, num_classes, activation=activation)
        self.latent_dim = latent_dim
        self.vi_penalty = vi_penalty
        self.sigma = sigma

    def create(self):
        x = Input(shape=self.input_shape)

        ly0 = Conv2D(64, (3, 3), activation=self.activation)(x)
        ly1 = MaxPooling2D(pool_size=(2, 2))(ly0)
        ly2 = Conv2D(64, (3, 3), activation=self.activation)(ly1)
        ly3 = MaxPooling2D(pool_size=(2, 2))(ly2)
        # ly4 = Dropout(0.25)(ly3)
        flatten = Flatten()(ly3)
        # z = Dense(self.latent_dim, activation=self.activation)(flatten)
        z_mean = Dense(self.latent_dim, activation=self.activation)(flatten)
        z_log_var = Dense(self.latent_dim, activation=self.activation)(flatten)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,),
                   arguments=dict(latent_dim=self.latent_dim, sigma=self.sigma))([z_mean, z_log_var])

        # ly6 = Dropout(0.5)(z)

        yh = Dense(self.num_classes, activation=self.output_activation, name='classifier')(z)

        if len(self.input_shape) == 1:
            flatten_dim = self.input_shape
        else:
            flatten_dim = 1
            for val in self.input_shape:
                flatten_dim *= val
        # ly7 = Dropout(0.5)(z)
        ly8 = Dense(flatten_dim, activation='sigmoid')(z)
        xh = Reshape(self.input_shape, name='decoder')(ly8)

        model = Model(inputs=x, outputs=[yh, xh])
        encoder = Model(x, z)

        classifier = Model(x, yh)

        self.model = model
        self.encoder = encoder
        self.classifier = classifier
        # self.loss = self.xent_loss
        self.loss = dict(classifier=self.xent_loss,
                         decoder=self.vloss(z_mean, z_log_var, penalty=self.vi_penalty))
        self.optimizer = 'adam'
        self.metrics = ['accuracy']

        return model

    def compile(self, model, *args, **kwargs):
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics
        )

        return model
