import gzip
import math
import pickle
import random as rand
import matplotlib.pyplot as plt

import numpy as np

INPUT_SIZE = 784
OUTPUT_SIZE = 10


def get_params(hidden_size):
    weights = np.array([np.array([[np.random.normal(loc=0.0, scale=1/math.sqrt(INPUT_SIZE)) for _ in range(INPUT_SIZE)] for _ in range(hidden_size)]),
                        np.array([[np.random.normal(loc=0.0, scale=1/math.sqrt(hidden_size)) for _ in range(hidden_size)] for _ in range(OUTPUT_SIZE)])],
                       dtype=object)
    biases = np.array([np.array([np.random.normal(loc=0.0, scale=1/math.sqrt(INPUT_SIZE)) for _ in range(hidden_size)]),
                       np.array([np.random.normal(loc=0.0, scale=1/math.sqrt(hidden_size)) for _ in range(OUTPUT_SIZE)])], dtype=object)

    return weights, biases


def sigmoid(values):
    return np.array([1 / (1 + pow(math.e, -_)) for _ in values])


def sigmoid_derivative(outputs):
    return np.array([sigmoid(_) * (1 - sigmoid(_)) for _ in outputs])


def softmax(values):
    inputs_sum = sum(pow(math.e, value) for value in values)
    return np.array([pow(math.e, value) / (inputs_sum + pow(1, 0.00000001)) for value in values])


def forward_prop(weights, biases, instances):
    z0 = np.add(weights[0].dot(instances.transpose()), biases[0].reshape(100, 1))
    y0 = sigmoid(z0)
    z1 = np.add(np.dot(weights[1], y0), biases[1].reshape(10, 1))
    y1 = softmax(z1)
    return z0, y0, z1, y1


def forward_prop_instance(weights, biases, instance):
    z0 = np.add(weights[0].dot(instance), biases[0])
    y0 = sigmoid(z0)
    z1 = np.add(np.dot(weights[1], y0), biases[1])
    y1 = softmax(z1)
    return z0, y0, z1, y1


def get_one_hot_for_label(label):
    label_arr = np.array([0 for _ in range(OUTPUT_SIZE)])
    label_arr[label] = 1
    return label_arr


def get_one_hot(labels):
    result = [np.array(get_one_hot_for_label(labels[0]))]
    for i in range(1, len(labels)):
        result = np.append(result, [get_one_hot_for_label(labels[i])], axis=0)

    return result.transpose()


def get_tuned_outputs(outputs):
    result = np.array([0 for _ in range(len(outputs))])
    result[np.argmax(outputs)] = 1
    return result


def backward_prop(z0, inputs, y1, y2, weights_level_2, targets):
    # dC/dy * dy/dz
    # shape = 10, 50
    dz2 = -1 / len(targets[0]) * np.subtract(targets, y2)
    # dy/dz * (error for every neuron * weights for it)
    # shape = 100, 50 * 100, 50(50, 10 dot 10, 100)  =>  100, 50
    dz1 = sigmoid_derivative(z0) * dz2.transpose().dot(weights_level_2).transpose()
    # (hidden -> output)
    # 10, 50 * 50, 100 = 10, 100
    dw2 = dz2.dot(y1.transpose())
    db2 = np.sum(dz2, axis=1)
    # (input -> hidden)
    # 100, 50 * 50, 784 = 100, 784
    dw1 = dz1.dot(inputs)
    db1 = np.sum(dz1, axis=1)

    return dw1, dw2, db1, db2


def update_params(weights, biases, dw1, dw2, db1, db2, learning_rate):
    weights[0] = weights[0] - learning_rate * dw1
    weights[1] = weights[1] - learning_rate * dw2
    biases[0] = biases[0] - learning_rate * db1
    biases[1] = biases[1] - learning_rate * db2

    return weights, biases


def get_loss(targets, outputs):
    loss = 0
    for i in range(len(targets)):
        loss += get_loss_for_instance(targets[i], outputs[i])
    return -1 / len(targets) * loss


def get_loss_for_instance(target, output):
    return sum([target[i] * math.log(output[i]) for i in range(len(target))])


def train(weights, biases, nr_epochs=60, learning_rate=0.65, batch_size=1000):
    for i in range(nr_epochs):
        for batch_start, batch_end in zip(range(0, TRAINING_SIZE, batch_size), range(batch_size, TRAINING_SIZE, batch_size)):
            batch = [np.array(training_set[0][batch_start:batch_end]), get_one_hot(training_set[1][batch_start:batch_end])]
            z0, y0, z1, y1 = forward_prop(weights, biases, batch[0])
            dw1, dw2, db1, db2 = backward_prop(z0, batch[0], y0, y1, weights_level_2=weights[1], targets=batch[1])
            weights, biases = update_params(weights, biases, dw1, dw2, db1, db2, learning_rate)
        print(f"epoch = {i}")
        if i % 5 == 0:
            print(f"the accuracy for the training set is: {get_accuracy(weights, biases, training_set)}")
            print(f"the accuracy for the validation set is: {get_accuracy(weights, biases, validation_set)}")
    return weights, biases


def get_accuracy(weights, biases, dataset):
    wrong_classified_nr = 0
    dataset_size = len(dataset[0])

    for i in range(dataset_size):
        label = dataset[1][i]
        z0, y0, z1, y1 = forward_prop_instance(weights, biases, dataset[0][i])
        y1 = get_tuned_outputs(y1)
        label_arr = get_one_hot_for_label(label)

        if not np.array_equal(y1, label_arr):
            wrong_classified_nr += 1

    return (dataset_size - wrong_classified_nr) / dataset_size * 100


if __name__ == '__main__':
    HIDDEN_SIZE = 100

    with gzip.open('mnist.pkl.gz', 'rb') as fd:
        training_set, validation_set, test_set = pickle.load(fd, encoding='latin')
        TRAINING_SIZE = len(training_set[1])

    weights_main, biases_main = get_params(hidden_size=HIDDEN_SIZE)
    weights_main, biases_main = train(weights_main, biases_main)
    print(f"the accuracy for the training set is: {get_accuracy(weights_main, biases_main, training_set)}")
    print(f"the accuracy for the test set is: {get_accuracy(weights_main, biases_main, test_set)}")
