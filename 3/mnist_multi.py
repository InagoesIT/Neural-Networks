import gzip
import math
import pickle
import random as rand
import matplotlib.pyplot as plt

import numpy as np

INPUT_SIZE = 784
OUTPUT_SIZE = 10


def get_params(hidden_size):
    weights = np.array([np.array([rand.uniform(-0.1, 0.1) for _ in range(INPUT_SIZE)] for _ in range(hidden_size)),
                        np.array([rand.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(OUTPUT_SIZE))],
                       dtype=object)
    biases = np.array([np.array(rand.uniform(-0.1, 0.1) for _ in range(hidden_size)),
                       np.array(rand.uniform(-0.1, 0.1) for _ in range(OUTPUT_SIZE))], dtype=object)

    print(weights)

    return weights, biases


def sigmoid(values):
    return np.array([1 / (1 + pow(math.e, -_)) for _ in values])


def sigmoid_derivative(outputs):
    return outputs * (1 - outputs)


def softmax(values):
    inputs_sum = sum(pow(math.e, value) for value in values)
    return np.array([pow(math.e, value) / inputs_sum for value in values])


def forward_prop(weights, biases, instance):
    print(weights[0])
    z0 = weights[0].dot(instance) + biases[0]
    y0 = sigmoid(z0)
    z1 = weights[1].dot(y0) + biases[1]
    y1 = softmax(z1)
    return z0, y0, z1, y1


def get_one_hot(label):
    label_arr = np.array([0 for _ in range(OUTPUT_SIZE)])
    label_arr[label] = 1
    return label_arr.transpose()


def get_tuned_outputs(results):
    outputs = np.array([sigmoid(result) for result in results])
    if np.count_nonzero(outputs == 1) == 1:
        return outputs.transpose()

    outputs = np.array([0 for _ in range(OUTPUT_SIZE)])
    outputs[np.argmax(results)] = 1
    return outputs.transpose()


def backward_prop(z1, y0, y1, y2, weights_level_2, targets):
    dz = np.array([[], []])
    dw = np.array([[], []])
    db = np.array([[], []])

    # dC/dy * dy/dz
    # shape = 1, 10
    dz[1] = 1 / len(targets) * np.sum(targets.subtract(y2))
    # dy/dz * (error for every neuron * weights for it)
    # shape = 1, 100
    dz[0] = sigmoid_derivative(z1) * (dz[1].dot(weights_level_2))
    # (hidden -> output)
    # 10, 1 * 1, 100 = 10, 100
    dw[1] = dz[1].transpose().dot(y1)
    db[1] = dz[1]
    # (input -> hidden)
    # 100, 1 * 1, 10 = 100, 10
    dw[0] = dz[0].transpose().dot(y0)
    db[0] = dz[0]

    return dz, db


def update_params(weights, biases, dw, db, learning_rate):
    weights = weights - learning_rate * dw
    biases = biases - learning_rate * db

    return weights, biases


def get_loss(targets, outputs):
    loss = 0
    for i in range(len(targets)):
        loss += get_loss_for_instance(targets[i], outputs[i])
    return -1 / len(targets) * loss


def get_loss_for_instance(target, output):
    return sum([target[i] * math.log(output[i]) for i in range(len(target))])


def train(weights, biases, nr_epochs=6, learning_rate=0.4, batch_size=50):
    batches_nr = TRAINING_SIZE // batch_size + 1

    for i in range(nr_epochs):
        dw_delta, db_delta = np.array([([0] * 100) * 10]), np.array([([0] * 10) * 100])
        for batch_start, batch_end in zip(range(0, batches_nr, batch_size), range(batch_size, batches_nr, batch_size)):
            batch = [training_set[0][batch_start:batch_end], training_set[1][batch_start:batch_end]]
            z0, y0, z1, y1 = forward_prop(weights, biases, batch)
            dw, db = backward_prop(z0, y0, z1, y1, weights_level_2=weights[1], targets=batch[1])
            dw_delta += dw
            db_delta += db
        for j in range(batches_nr):
            weights, biases = update_params(weights, biases, dw_delta, db_delta, learning_rate)

    return weights, biases


def get_accuracy(dataset):
    wrong_classified_nr = 0
    dataset_size = len(dataset[0])

    for i in range(dataset_size):
        image_pixels = dataset[0][i]
        label = dataset[1][i]
        results = np.add(np.dot(weights, image_pixels), biases)
        outputs = get_tuned_outputs(results)
        label_arr = get_one_hot(label)

        if not np.array_equal(outputs, label_arr):
            wrong_classified_nr += 1

    return (dataset_size - wrong_classified_nr) / dataset_size * 100


if __name__ == '__main__':
    HIDDEN_SIZE = 100

    with gzip.open('mnist.pkl.gz', 'rb') as fd:
        training_set, validation_set, test_set = pickle.load(fd, encoding='latin')
        TRAINING_SIZE = len(training_set[1])

    weights_main, biases_main = get_params(hidden_size=HIDDEN_SIZE)
    weights_main, biases_main = train(weights_main, biases_main)
    print(f"the accuracy for the training set is: {get_accuracy(training_set)}")
    print(f"the accuracy for the test set is: {get_accuracy(test_set)}")
