from network import NeuralNetwork

import numpy as np
import unittest

inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3], [-0.2, 0.5, 0.2]])
test_w_h_o = np.array([[0.3, -0.1]])

class TestMethods(unittest.TestCase):
    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        self.assertTrue(np.all(network.activation_function(0.5) == \
                1 / (1 + np.exp(-0.5))))

    def test_train(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.first_layer_weights = test_w_i_h.copy()
        network.second_layer_weights = test_w_h_o.copy()

        network.train(inputs, targets)
        print('Networks H_to_O', network.second_layer_weights)
        print('Networks W_to_H', network.first_layer_weights)
        self.assertTrue(np.allclose(network.second_layer_weights, \
            np.array([[ 0.37275328, -0.03172939]])))
        self.assertTrue(np.allclose(network.first_layer_weights, \
            np.array([[ 0.10562014,  0.39775194, -0.29887597], \
            [-0.20185996,  0.50074398,  0.19962801]])))

    def test_run(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.first_layer_weights = test_w_i_h.copy()
        network.second_layer_weights = test_w_h_o.copy()
        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
    unittest.TextTestRunner().run(suite)
