import numpy as np
import random


class HMM:
    
    def __init__(self, initial, transition, emission, states=None, observations=None):
        #vrow_init, col_init = initial.shape
        row_tran, col_tran = transition.shape
        row_emis, col_emis = emission.shape

        self._initial = initial
        self._transition = transition
        self._emission = emission

        self._state_size = col_tran
        self._observation_size = col_emis

    def draw(self, n=1):
        init_state = self._weighted_draw(self._initial)
        yield self._weighted_draw(self._emission[init_state, :])
        state = init_state
        for i in range(n-1):
            state = self._weighted_draw(self._transition[state, :])
            yield self._weighted_draw(self._emission[state, :])

    @property
    def state_size(self):
        return self._state_size

    @property
    def observation_size(self):
        return self._observation_size

    def _weighted_draw(self, weighted):
        normalized = self._norm(weighted)
        accum = self._acc(normalized)
        rng = random.random()
        for i, a in enumerate(accum):
            if rng <= a:
                return i

    def _norm(self, array):
        total = sum(array)
        return (v/total for v in array)

    def _acc(self, array):
        a = 0
        for v in array:
            a += v
            yield a

