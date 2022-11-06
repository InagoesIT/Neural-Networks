import gzip
import pickle
import random as rand
import matplotlib.pyplot as plt

import numpy as np

INPUT_SIZE = 784
OUTPUT_SIZE = 10

with gzip.open('mnist.pkl.gz', 'rb') as fd:
    training_set, validation_set, test_set = pickle.load(fd, encoding='latin')
    TRAINING_SIZE = len(training_set[1])
    print(training_set)

weights = np.array([[rand.random() for _ in range(INPUT_SIZE)] for _ in range(OUTPUT_SIZE)])
biases = np.array([rand.random() for _ in range(OUTPUT_SIZE)])


def activation(value):
    if value > 0:
        return 1
    return 0


def get_tuned_outputs(results):
    outputs = np.array([activation(result) for result in results])
    if np.count_nonzero(outputs == 1) == 1:
        return outputs.reshape((OUTPUT_SIZE, 1))

    outputs = np.array([0 for _ in range(OUTPUT_SIZE)])
    outputs[np.argmax(results)] = 1
    outputs = outputs.reshape((OUTPUT_SIZE, 1))
    return outputs


def get_label_array(label):
    label_arr = np.array([0 for _ in range(OUTPUT_SIZE)])
    label_arr[label] = 1
    return label_arr.reshape((OUTPUT_SIZE, 1))


def train(nr_epochs=6, learning_rate=0.4):
    global weights, biases
    all_classified = False

    while not all_classified and nr_epochs > 0:
        all_classified = True
        for i in range(TRAINING_SIZE):
            image_pixels = np.array(training_set[0][i])
            label = np.array(training_set[1][i])
            # result = weights * image_pixels + biases
            results = np.add(np.dot(weights, image_pixels), biases)
            outputs = get_tuned_outputs(results)
            label_arr = get_label_array(label)
            image_pixels = image_pixels.reshape((1, INPUT_SIZE))
            # weights = weights + (label_arr - outputs) * image_pixels * learning_rate
            weights = np.array(weights + (np.dot(np.subtract(label_arr, outputs), image_pixels) * learning_rate))
            biases = biases.reshape((OUTPUT_SIZE, 1))
            # biases = biases + (label_arr - outputs) * learning_rate
            biases = np.array(np.add(biases, np.subtract(label_arr, outputs) * learning_rate))
            biases = biases.reshape(-1)

            if not np.array_equal(outputs, label_arr):
                all_classified = False
        nr_epochs -= 1

    return weights, biases


def get_accuracy(dataset):
    wrong_classified_nr = 0
    dataset_size = len(dataset[0])

    for i in range(dataset_size):
        image_pixels = dataset[0][i]
        label = dataset[1][i]
        results = np.add(np.dot(weights, image_pixels), biases)
        outputs = get_tuned_outputs(results)
        label_arr = get_label_array(label)

        if not np.array_equal(outputs, label_arr):
            wrong_classified_nr += 1

    return (dataset_size - wrong_classified_nr) / dataset_size * 100


def get_best_epoch_nr(graph_name):
    training_accuracy = []
    validation_accuracy = []
    nr_epochs = [i for i in range(1, 10)]

    for nr_epoch in nr_epochs:
        train(nr_epochs=nr_epoch)
        training_accuracy.append(get_accuracy(training_set))
        validation_accuracy.append(get_accuracy(validation_set))

    plt.plot(nr_epochs, training_accuracy, label="training accuracy")
    plt.plot(nr_epochs, validation_accuracy, label="validation accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig(graph_name)


def get_best_learning_rate(graph_name):
    training_accuracy = []
    validation_accuracy = []
    learning_rates = [i * 0.1 for i in range(1, 11)]

    for learning_rate in learning_rates:
        train(learning_rate=learning_rate)
        training_accuracy.append(get_accuracy(training_set))
        validation_accuracy.append(get_accuracy(validation_set))

    plt.plot(learning_rates, training_accuracy, label="training accuracy")
    plt.plot(learning_rates, validation_accuracy, label="validation accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("learning_rate")
    plt.legend()
    plt.savefig(graph_name)


train()
print(f"the accuracy for the training set is: {get_accuracy(training_set)}")
print(f"the accuracy for the test set is: {get_accuracy(test_set)}")
