import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Neural Network with one hidden layer
# 1st activation function as sigmoid
# 2nd activation function as f(x) => x
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.hidden_nodes = hidden_nodes
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.first_layer_weights = np.random.normal(0.0,
            self.hidden_nodes**-0.5, (self.hidden_nodes, self.input_nodes))
        self.second_layer_weights = np.random.normal(0.0,
            self.output_nodes**-0.5, (self.output_nodes, self.hidden_nodes))

    #sigmoid activation into hidden layer
    def activation_function(self, x):
        return 1 / (1.00 + np.exp(-x))

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #Forward Pass
        hidden_inputs = np.dot(self.first_layer_weights, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.second_layer_weights, hidden_outputs)
        final_outputs = final_inputs

        #Backward Pass
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.second_layer_weights.T, output_errors)
        hidden_grad = hidden_outputs * (1 - hidden_outputs)

        # Update weights
        self.second_layer_weights += self.learning_rate * \
            np.dot(output_errors, hidden_outputs.T)
        self.first_layer_weights += self.learning_rate * \
            np.dot(hidden_errors * hidden_grad, inputs.T)

    def run(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        #Forward Pass
        hidden_inputs = np.dot(self.first_layer_weights, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.second_layer_weights, hidden_outputs)
        final_outputs = final_inputs

        return final_outputs

#Mean Squared Error
def MSE(y, Y):
    return np.mean((y - Y) ** 2)

if __name__ == '__main__':
    rides = pd.read_csv('Bike-Sharing-Dataset/hour.csv')

    #remove dummy fields and create binary dummy variables for those
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    drop_fields = ['season', 'weathersit', 'weekday', 'atemp', 'mnth',
        'workingday', 'hr', 'instant', 'dteday']

    for field in dummy_fields:
        dummies = pd.get_dummies(rides[field], prefix=field, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)

    data = rides.drop(drop_fields, axis=1)

    #standardize continuous variables
    features_to_std = ['casual', 'registered', 'cnt', 'temp', 'hum',
        'windspeed']
    scaled_features = {}
    for each in features_to_std:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean) / std

    #split data into test/train, seperate features and targets
    test = data[-21*24:]
    data = data[:-21*24]

    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test.drop(target_fields, axis=1), \
        test[target_fields]

    #create validation set
    train_features, train_targets = features[:-60*24], targets[:-60*24]
    val_features, val_targets = features[-60*24:], targets[-60*24:]

    #create neural net on data
    epochs = 1000
    learning_rate = .1
    hidden_nodes = 30

    network = NeuralNetwork(train_features.shape[1], hidden_nodes, 1, \
        learning_rate)

    losses = {
        'train': [],
        'validation': []
    }

    for e in range(epochs):
        #get random 128 datapoints from set
        batch = np.random.choice(train_features.index, size=128)

        for record, target in zip(train_features.ix[batch].values, \
            train_targets.ix[batch]['cnt']):
            network.train(record, target)

        #Print training progress
        train_loss = MSE(network.run(train_features), \
            train_targets['cnt'].values)
        val_loss = MSE(network.run(val_features), val_targets['cnt'].values)


        ### Runs for a hella long time
        sys.stdout.write('\nProgress: ' + str(100 * e/float(epochs))[:4] \
            + '%\nTraining Loss: ' + str(train_loss)[:5] \
            + 'validation Loss: ' + str(val_loss)[:5])

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)


    #Plot Losses
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    plt.ylim(ymax=0.5)

    ### plt.show()

    #Plot Neural Network generated Regression
    fig, ax = plt.subplots(figsize=(8, 4))

    mean, std = scaled_features['cnt']
    predictions = network.run(test_features) * std + mean

    ax.plot(predictions[0], label='Prediction')
    ax.plot((test_targets['cnt'] * std + mean).values, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()

    dates = pd.to_datetime(rides.ix[test.index]['dteday'])
    dates = dates.apply(lambda d : d.strftime('%b %d'))

    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)

    ### plt.show()
