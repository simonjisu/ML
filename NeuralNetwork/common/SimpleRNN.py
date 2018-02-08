import numpy as np
from collections import defaultdict, OrderedDict
from common.layers import Linear, Tanh, SoftmaxWithLoss
from common.utils_nn import numerical_gradient

class Simple_RNN(object):
    def __init__(self, input_size, hidden_size, output_size, bptt_truncate):
        self.input_size = input_size  # V
        self.hidden_size = hidden_size  # N
        self.output_size = output_size  # M
        self.bptt_truncate = bptt_truncate

        self.params = {}
        self.params_init()

        self.layers = None
        self.last_layers = None
        self.T = None
        self.h0 = None

    def params_init(self):
        scale_xh = np.sqrt(1.0 / self.input_size)
        scale_hh = np.sqrt(1.0 / self.hidden_size)
        self.params['W_xh'] = np.random.randn(self.input_size, self.hidden_size) * scale_xh
        self.params['W_hh'] = np.random.randn(self.hidden_size, self.hidden_size) * scale_hh
        self.params['W_hy'] = np.random.randn(self.hidden_size, self.output_size) * scale_hh
        self.params['b_h'] = np.zeros((1, self.hidden_size))
        self.params['b_y'] = np.zeros((1, self.output_size))

    def _hidden_init(self):
        self.h0 = np.zeros((1, self.hidden_size))
        return self.h0

    def _layers_init(self):
        self.layers = defaultdict(OrderedDict)
        for i in np.arange(self.T):
            self.layers['Affine_xh'][i] = Linear(self.params['W_xh'], bias=False)
            self.layers['Affine_hh'][i] = Linear(self.params['W_hh'], self.params['b_h'])
            self.layers['Tanh'][i] = Tanh()
            self.layers['Affine_hy'][i] = Linear(self.params['W_hy'], self.params['b_y'])

    def _last_layers_init(self):
        self.last_layers = {}
        for i in np.arange(self.T):
            self.last_layers[i] = SoftmaxWithLoss()

    def forward(self, x):
        self.T = len(x)
        if self.layers is None:
            self._layers_init()

        hs = {}
        hs[-1] = self._hidden_init()
        os = {}

        for i in np.arange(self.T):
            a = self.layers['Affine_hh'][i].forward(hs[i - 1]) + \
                self.layers['Affine_xh'][i].forward(x[i:i + 1, :])
            hs[i] = self.layers['Tanh'][i].forward(a)
            os[i] = self.layers['Affine_hy'][i].forward(hs[i])

        return os, hs

    def predict(self, x):
        os, hs = self.forward(x)
        result = []
        for i in np.arange(self.T):
            max_idx = np.argmax(os[i])
            result.append(max_idx)
        return np.array(result)

    def loss(self, x, t):
        total_loss = 0
        os, hs = self.forward(x)
        if self.last_layers is None:
            self._last_layers_init()

        for i in range(self.T):
            loss = self.last_layers[i].forward(os[i], t[i].reshape(1, -1))
            total_loss += loss

        return total_loss / len(t)

    def backward(self):
        # BPTT
        self._params_summation_init()
        dht = np.zeros_like(self.h0)

        for t in np.arange(self.T)[::-1]:
            dout = self.last_layers[t].backward()
            dht_raw = self.layers['Affine_hy'][t].backward(dout)
            dat = self.layers['Tanh'][t].backward(dht_raw) + dht
            dht = self.layers['Affine_hh'][t].backward(dat)
            dx = self.layers['Affine_xh'][t].backward(dat)  # dx
            self._params_summation(t)
            #         self._params_summation(key=False)

    def _params_summation_init(self):
        self.params_summ = {}
        self.summ_count = 0
        self.params_summ['W_xh'] = np.zeros_like(self.params['W_xh'])
        self.params_summ['W_hh'] = np.zeros_like(self.params['W_hh'])
        self.params_summ['W_hy'] = np.zeros_like(self.params['W_hy'])
        self.params_summ['b_h'] = np.zeros_like(self.params['b_h'])
        self.params_summ['b_y'] = np.zeros_like(self.params['b_y'])

    def _params_summation(self, step=None, key=True):
        if key:
            self.summ_count += 1
            self.params_summ['W_xh'] += self.layers['Affine_xh'][step].dW
            self.params_summ['W_hh'] += self.layers['Affine_hh'][step].dW
            self.params_summ['W_hy'] += self.layers['Affine_hy'][step].dW
            self.params_summ['b_h'] += self.layers['Affine_hh'][step].db
            self.params_summ['b_y'] += self.layers['Affine_hy'][step].db
        else:
            for k, v in self.params_summ.items():
                self.params_summ[k] = v / self.summ_count

    def backward_truncate(self):
        # TBPTT
        self._params_summation_init()
        dht = np.zeros_like(self.h0)

        for t in np.arange(self.T)[::-1]:
            dout = self.last_layers[t].backward()
            dht_raw = self.layers['Affine_hy'][t].backward(dout)
            self.params_summ['W_hy'] += self.layers['Affine_hy'][t].dW
            self.params_summ['b_y'] += self.layers['Affine_hy'][t].db

            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dat = self.layers['Tanh'][bptt_step].backward(dht_raw) + dht
                dht = self.layers['Affine_hh'][bptt_step].backward(dat)
                dx = self.layers['Affine_xh'][bptt_step].backward(dat)  # dx
                self.params_summ['W_xh'] += self.layers['Affine_xh'][bptt_step].dW
                self.params_summ['W_hh'] += self.layers['Affine_hh'][bptt_step].dW
                self.params_summ['b_h'] += self.layers['Affine_hh'][bptt_step].db

    def gradient_check(self, x, t, delta=0.001, th_error=0.1, num_check=3):
        ks = list(self.params_summ.keys())
        f = lambda w: self.loss(x, t)
        for name in ks:
            s0 = self.params[name].shape
            s1 = self.params_summ[name].shape
            assert s0 == s1, \
                "[Error] dimensions don't match: ({}) params-{} grads-{}".format(name, s0, s1)
            for i in np.arange(num_check):
                num_grads = numerical_gradient(f, self.params[name])
                back_grads = self.params_summ[name]
                rel_error = np.abs(back_grads - num_grads) / (np.abs(back_grads) + np.abs(num_grads))
                mask = rel_error > th_error
            print(name, mask.sum())