from keras.callbacks import Callback


class DummyHistory(Callback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dummy = []

    def on_epoch_begin(self, epoch, logs={}):
        super().on_epoch_begin(epoch, logs)
        pass

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        pass

    def on_batch_begin(self, batch, logs={}):
        super().on_batch_begin(batch, logs)
        pass

    def on_batch_end(self, batch, logs={}):
        super().on_batch_end(batch, logs)
        pass

    def on_train_begin(self, logs={}):
        super().on_train_begin(logs)
        pass

    def on_train_end(self, logs={}):
        super().on_train_end(logs)
        pass


class FeedforwardHistory(Callback):
    def __init__(self, net, x_test, skip=5):
        self.net = net
        self.x_test = x_test
        self.outs = []
        self.epoch = 0
        self.batch = 0
        self.skip = skip
#        self.train = 0

    def on_batch_end(self, batch, logs={}):
        self.batch = batch
        if batch % self.skip == 0:
            self.outs.append(('epoch:{}, batch:{}'.format(self.epoch, self.batch),
                              self.net.predict(self.x_test)))

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

#    def on_train_begin(self, logs={}):
#        self.train += 1
