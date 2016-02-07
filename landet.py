# -*- coding: utf-8 -*-

from __future__ import print_function

import codecs
import json
import math
import os
import random
import re
import time
import math
import multiprocessing
import threading
from collections import Counter

from pybrain.datasets.supervised     import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities           import percentError
from pybrain.structure import SigmoidLayer
from pybrain.structure import TanhLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure.modules import BiasUnit

import numpy

import matplotlib.pyplot as plt

#import experiment5

MAX_FEATURE_COUNT = 30 # Max is 4029 right now because everything else is filtered.
MAX_TIME_SECONDS = 60
MAX_SAMPLE_COUNT = 0 #0 # Unlimited 500

SPLIT_CHARS = r"\s,\?/!;:\[\]\(\)"
# Quotes
SPLIT_CHARS = SPLIT_CHARS + u'"\u201d\u2019\u201c\u201a\u2018'
SPLIT_CHARS = SPLIT_CHARS + u"\u2014\u2013\u2026\u2308\u2021"
PATTERN = r"[%s]+|'\s+|\s+'|\.\s|--+" % SPLIT_CHARS
SPLIT_PATTERN = re.compile(PATTERN)

class Features(object):
    def __init__(self, word_list, word_to_index, class_list, class_to_index):
        self.word_list = word_list
        self.word_to_index = word_to_index
        self.class_list = class_list
        self.class_to_index = class_to_index

        self.word_freq_mean = len(word_list) * [0]
        self.word_freq_std = len(word_list) * [0.001]


    @staticmethod
    def make_word_list_of_text(text):
        word_list = []
        # Simple letter frequencies.
        for x in text:
            if not x.isspace():
                word_list.append(x)
        # Bigrams
        chunk_len = 2
        for index in xrange(len(text) - chunk_len + 1):
            bigram = text[index:index + chunk_len - 1]
            if not bigram.isspace():
                pass
#                word_list.append(x)
        # Trigrams
        # Same as above but with chunk_len = 3
        return word_list

    def map_to_frequency(self, text):
        words = Counter()
        text_words = Features.make_word_list_of_text(text)
        count = len(text_words)
        words.update(text_words)

        freq = [0] * len(self.word_list)
        for word, word_count in words.iteritems():
            if word in self.word_to_index:
                # A word we track.
                word_index = self.word_to_index[word]
                freq[word_index] = ((float(word_count) / count) - self.word_freq_mean[word_index]) /  self.word_freq_std[word_index] # To bring the numbers in the -1 -> 1 range.

        return freq

    def map_to_classes(self, text_classes):
        assert len(self.class_list)
        target = [0] * len(self.class_list)
        for text_class in text_classes.split(","):
            if text_class in self.class_to_index:
                target[self.class_to_index[text_class]] = 1

        return target


class StdDevMeanCalculator(object):
    def __init__(self):
        self._mean = 0.0
        self.S = 0.0
        self._value_count = 0

    def updateValueCountWithZeros(self, newTotalCount):
        assert newTotalCount  >= self._value_count
        if self._value_count == 0:
            self._value_count = newTotalCount
            return

        while self._value_count < newTotalCount:
            self.addNumber(0.0)

    def addNumber(self, value):
        old_mean = self._mean
        value_count = self._value_count + 1
        new_mean = old_mean + float(value - old_mean) / value_count
        self.S = self.S + (value - old_mean) * (value - new_mean)
        self._mean = new_mean
        self._value_count = value_count

    def getVariance(self):
        if self._value_count < 2:
            return 0
        return self.S / (self._value_count - 1)

    def getStdDev(self):
        return math.sqrt(self.getVariance())

testStdDevCalc = StdDevMeanCalculator()
testStdDevCalc.addNumber(1.0)
assert testStdDevCalc._value_count == 1
assert testStdDevCalc._mean == 1
assert testStdDevCalc.getVariance() == 0
testStdDevCalc.addNumber(3.0)
assert testStdDevCalc._value_count == 2
assert testStdDevCalc._mean == 2
assert testStdDevCalc.getVariance() == 2
testStdDevCalc.updateValueCountWithZeros(4) # 1, 3, 0, 0
assert testStdDevCalc._value_count == 4
assert testStdDevCalc._mean == 1
assert testStdDevCalc.getVariance() > 1.9 # approximately 2.0
assert testStdDevCalc.getVariance() < 2.1 # approximately 2.0

def get_data_files():
    result = {}
    root_dirs = ("data", "scrape_data")
    for root_dir in root_dirs:
        languages = os.listdir(root_dir)
        for language in languages:
            language_dir = os.path.join(root_dir, language)
            if os.path.isdir(language_dir):
                data_files = os.listdir(language_dir)
                for data_file in data_files:
                    data_file_name = os.path.join(language_dir, data_file)
                    if data_file_name.endswith(".txt") and os.path.isfile(data_file_name):
                        result.setdefault(language, []).append(data_file_name)
    return result

def main():

    print("Looking for data files")
    data_files = get_data_files()

    best_test_f1_algo = (0, 0)
    best_solution = None

    WORD_SETS_TO_TRY = 1
    for word_set_index in range(WORD_SETS_TO_TRY):
        print("Find features to use (%d/%d)" % (word_set_index + 1,
                                                WORD_SETS_TO_TRY))
        features = build_features(data_files)

        files_with_lan = []
        for language, file_list in data_files.iteritems():
            for data_file in file_list:
                files_with_lan.append((data_file, language))

        random.shuffle(files_with_lan)
        SPLIT_PCT = 0.7

        samples = get_samples_from_txt_meta(files_with_lan,
                                            features, 2 * MAX_SAMPLE_COUNT)

        # Current
        BACKPROP_MOMENTUM = 0.001
        BACKPROP_WEIGHTDECAY = 0.001
        BACKPROP_LEARNINGRATE = 0.1
        BACKPROP_LRDECAY = 1.0

        max_error = 0
        max_epochs = 0
        important_features = [] # List of maps "word" -> Percentile (0.0 -> 1.0)
        for ds_size in (
    #        10,
    #        25,
    #        100,
    #        250,
    #        500,
    #        1000,
            len(samples) / 2,
            ):
            (train_ds, test_ds,
             train_ds_labels, test_ds_labels) = make_ds_from_samples(
                samples, SPLIT_PCT, ds_size, features)
            for hidden_layers in [
                [],
                [train_ds.outdim * 3],
    #            [train_ds.outdim * 3],
    #            [train_ds.outdim * 9],
    #            [train_ds.outdim * 9, train_ds.outdim * 3],
        #        [train_ds.outdim * 27, train_ds.outdim * 9],
        #        [train_ds.outdim * 27, train_ds.outdim * 9, train_ds.outdim * 3],
        #        [train_ds.outdim * 10],
        #        [train_ds.outdim * 3],
        #        [train_ds.outdim * 3],
        #        [30 * train_ds.outdim / 4],
                ]:
                for learningrate, linestyle in [
        #            (BACKPROP_LEARNINGRATE * 10, "--"),
                    (BACKPROP_LEARNINGRATE, "-"),
        #            (BACKPROP_LEARNINGRATE / 10, ":"),
                    ]:
                    for momentum in [
            #            BACKPROP_MOMENTUM * 10,
                        BACKPROP_MOMENTUM,
            #            BACKPROP_MOMENTUM / 10,
                        ]:
                        for lrdecay in [
        #                    BACKPROP_LRDECAY / 10,
                            BACKPROP_LRDECAY,
        #                    BACKPROP_LRDECAY * 10,
                            ]:
                            for weightdecay in [
    #                            BACKPROP_WEIGHTDECAY / 100,
    #                            BACKPROP_WEIGHTDECAY / 10,
                                BACKPROP_WEIGHTDECAY,
    #                            BACKPROP_WEIGHTDECAY * 10,
    #                            BACKPROP_WEIGHTDECAY * 100,
                                ]:
                                description = "test_size=%d, layers=%s, lr=%g/%g, mom=%g dec=%g" % (
                                    len(train_ds),
                                    str(hidden_layers),
                                    learningrate, lrdecay,
                                    momentum, weightdecay)
                                print("Trying %s" % description)
                                (fnn, epochs, train_algo_errors, test_algo_errors, train_F1s, testF1s) = trainNetwork(
                                    train_ds, test_ds, train_ds_labels, test_ds_labels, features,
                                    learningrate, lrdecay,
                                    momentum,
                                    weightdecay,
                                    hidden_layers,
                                    MAX_TIME_SECONDS)

                                package_res = (max(testF1s[1:]), 100-min(train_algo_errors[1:]))
                                if package_res > best_test_f1_algo or best_solution is None:
                                    best_solution = { "words": features.word_list,
                                                      "desc": description,
                                                      # Could save fnn.params but that would be spammy in the output.
                                                      "fnn": fnn,
                                                      "festures": features,
                                                      }
                                    best_test_f1_algo = package_res

                        #    print_classifications(TRAIN_TEXTCLASSES + TEST_TEXTCLASSES, fnn,
                        #                          features)
                        #    print_classifications_from_txt_meta(calibre_txt_and_meta_files, fnn,
                        #                                        features)

                                # plt.plot(epochs, train_algo_errors, linestyle,
                                #          label="Train error [%s]" % (description))
                                # plt.plot(epochs, test_algo_errors, "." + linestyle,
                                #          label="Test error [%s]" % (description))

                                for i in range(len(features.class_list)):
                                    plt.plot(epochs,
                                             [x[i] for x in testF1s],
                                             "." + linestyle,
                                             label="Test F1 %s [%s]" %
                                             (features.class_list[i], description))
                                max_error = max(max_error, max(train_algo_errors + test_algo_errors))
                                if epochs and epochs[-1] > max_epochs:
                                    max_epochs = epochs[-1]
                                important_features.append(calc_feature_importance(fnn, features))
        #                        print(fnn.params)

#        print_most_important_features(important_features, features)
    if best_solution is not None:
        pass
        print("best_solution with success %s is %s %s" % (str(best_test_f1_algo),
                                                          best_solution["desc"],
                                                          best_solution["words"]))
        test_text = input("Write something to test with: ")
        print_classification_for_text(best_solution["desc"],
                                      test_text,
                                      best_solution["fnn"],
                                      best_solution["features"])
        

#    plt.ylim(0, min(100, max_error * 1.1))
    plt.xlim(0, max_epochs + 1)
#    plt.plot(train_errors, '-', label="Train data error")
#    plt.plot(test_errors, '-', label="Test data error")
#    plt.plot(train_F1s, '-', label="Train data F1")
#    plt.plot(test_F1s, '-', label="Test data F1")
    plt.legend(loc="upper right")
    print("DONE! Close graph to get the prompt back.")
    plt.show(block=True)


def trainNetwork(train_ds, test_ds,
                 train_ds_labels, test_ds_labels,
                 features,
                 learningrate, lrdecay,
                 momentum, weightdecay,
                 hidden_layers,
                 time_limit_seconds):
    fnn = FeedForwardNetwork()
    inLayer = LinearLayer(train_ds.indim)
    fnn.addInputModule(inLayer)
    lastLayer = inLayer
    connection_number = 0 # connection-0 is the connection from the input layer.
    for hidden_layer_size in hidden_layers:
#        hiddenLayer = SigmoidLayer(hidden_layer_size)
        hiddenLayer = TanhLayer(hidden_layer_size)
        fnn.addModule(hiddenLayer)
        fnn.addConnection(
            FullConnection(lastLayer, hiddenLayer,
                           name="connection-%d" % connection_number))
        connection_number = connection_number + 1
        bias = BiasUnit()
        fnn.addModule(bias)
        fnn.addConnection(FullConnection(bias, hiddenLayer))
        lastLayer = hiddenLayer
    outLayer = SigmoidLayer(train_ds.outdim)
    fnn.addOutputModule(outLayer)
    fnn.addConnection(
        FullConnection(lastLayer, outLayer,
                       name="connection-%d" % connection_number))
    bias = BiasUnit()
    fnn.addModule(bias)
    fnn.addConnection(FullConnection(bias, outLayer))
    fnn.sortModules()

    trainer = BackpropTrainer(fnn, dataset=train_ds,
                              learningrate=learningrate,
                              lrdecay=lrdecay,
                              momentum=momentum,
                              verbose=False,
                              weightdecay=weightdecay)

    # Train
    (initial_train_error, initial_train_F1) = percentClassErrorAndF1(fnn, train_ds, train_ds_labels, features)
    train_errors = [initial_train_error]
    train_F1s = [initial_train_F1]
    (initial_test_error, initial_test_F1) = percentClassErrorAndF1(fnn, test_ds, test_ds_labels, features)
    test_errors = [initial_test_error]
    test_F1s = [initial_test_F1]
    train_algo_errors = [trainer.testOnData(train_ds) * 100]
    test_algo_errors = [trainer.testOnData(test_ds) * 100]
    epochs = [0]
    try:
        start_time = time.time()
        for i in range(200):
            for _ in xrange(50):
                train_algo_error = trainer.train() * 100.0
                if math.isnan(train_algo_error):
                    break
            if math.isnan(train_algo_error):
                break
            (trnresult, trnF1) = percentClassErrorAndF1(fnn, train_ds, train_ds_labels, features)
            (tstresult, tstF1) = percentClassErrorAndF1(fnn, test_ds, test_ds_labels, features)
            test_algo_error = trainer.testOnData(test_ds)* 100
            now_time = time.time()
            time_left = time_limit_seconds - (now_time - start_time)
            print("epoch %3d:" % trainer.totalepochs,
                  "  train error: %6.4f%%" % train_algo_error,
                  "  test error: %6.4f%%" % test_algo_error,
                  "  train F1: %s" % ", ".join([("%.2f" % x) for x in trnF1]),
                  "  test F1: %s" % ", ".join([("%.2f" % x) for x in tstF1]),
                  "  %ds left" % int(round(time_left)))

            epochs.append(trainer.totalepochs)
            train_errors.append(trnresult)
            train_F1s.append(trnF1)
            test_errors.append(tstresult)
            test_F1s.append(tstF1)
            train_algo_errors.append(train_algo_error)
            test_algo_errors.append(test_algo_error)
            if time_left <= 0:
                print("Timeout: Time to report the results.")
                break;
            # if test_algo_errors[-1] < 4:
            #     print("Good enough? Don't want to overtrain")
            #     break;

    except KeyboardInterrupt:
        # Someone pressed Ctrl-C, try to still plot the data.
        print("Aborted training...")
        pass

    return (fnn, epochs, train_algo_errors, test_algo_errors, train_F1s, test_F1s)

def make_ds_from_samples(samples, split_pct,
                         max_count, features):
    split_index = int(len(samples) * split_pct)
    if split_index > max_count:
        split_index = max_count

    def make_ds_with_samples(sample_subset):
        ds = SupervisedDataSet(len(features.word_list),
                               len(features.class_list))
        ds_labels = []
        for sample_features, target, label in sample_subset:
            ds.addSample(sample_features, target)
            ds_labels.append(label)
        return (ds, ds_labels)

    train_ds, train_ds_labels = make_ds_with_samples(samples[0:split_index])
#    print("train_ds:")
#    print(train_ds)

    test_ds, test_ds_labels = make_ds_with_samples(
        samples[split_index:len(samples)])
#    print("test_ds:")
#    print(test_ds)

    return (train_ds, test_ds, train_ds_labels, test_ds_labels)


def calc_feature_importance(fnn, features):
    inputLayer = fnn.inmodules[0]
    connection = fnn.connections[inputLayer][0]
#    print(connection)
    params = connection.params
#    print(params)
    indim = connection.indim
    outdim = connection.outdim
    # this is an array of indim*outdim elements.
    word_and_weight = []
    for i in xrange(indim):
        word = features.word_list[i]
        numbers = []
#        sumsquares = 0
        for o in xrange(outdim):
            number = params[o * indim + i]
#            sumsquares = sumsquares + number * number
            numbers.append(number)
#        print("%s: %g (%r)" % (word, sum(numbers), numbers))
#        print("%s: %g" % (word, sum(numbers)))
        word_and_weight.append((sum([abs(x) for x in numbers]), word))

    weights = [x for (x, y) in word_and_weight]
    max_value = max(weights)
    min_value = min(weights)
    feature_percentiles = {}
    for weight, word in word_and_weight:
        percentile = (weight - min_value) / (max_value - min_value)
        feature_percentiles[word] = percentile

    return feature_percentiles

def print_most_important_features(feature_percentile_list, features):
    avg_percentile_for_word = []
    for word in features.word_list:
        percentiles = []
        for feature_percentiles in feature_percentile_list:
            percentiles.append(feature_percentiles[word])
        mean = numpy.mean(percentiles)
        avg_percentile_for_word.append((mean, word))

    sorted_word_and_pct = sorted(avg_percentile_for_word, reverse=True)
    if len(features.word_list) > 100:
        for weight, word in sorted_word_and_pct[:50]:
            print("%s  %g" % (word, weight))
        print("...")
        for weight, word in sorted_word_and_pct[-50:]:
            print("%s  %g" % (word, weight))
    else:
        for weight, word in sorted_word_and_pct:
            print("%s  %g" % (word, weight))

def print_classification_for_text(desc, text, fnn, features):
    freq = features.map_to_frequency(text)
    machine_res = fnn.activate(freq)
    classes = []
    for i in range(len(features.class_list)):
        if machine_res[i] >= 0.5:
            classes.append(features.class_list[i])
    if not classes:
        classes = ["not of a supported class"]
    print("'%s' is a %s" % (desc, ", ".join(classes)))

def print_classifications(textclasses, fnn, features):
    for text, _ in textclasses:
        print_classification_for_text(text[:40], text, fnn, features)

def print_classifications_from_txt_meta(txt_and_meta_files, fnn, features):

    for txt_file, metadata_file in txt_and_meta_files:
        text = read_text(txt_file)
        print_classification_for_text(os.path.basename(txt_file), text, fnn, features)

def percentClassErrorAndF1(fnn, ds, ds_labels, features, verbose=False):
    count = len(ds) * ds.outdim
    total_error = 0
    total_F1s = []
    machine_results = []
    targets = []
    for data, target in ds:
        machine_res = fnn.activate(data)
        machine_results.append(machine_res)
        targets.append(target)

    for output_bit in range(ds.outdim):
        true_positive_count = 0
        true_negative_count = 0
        false_positive_count = 0
        false_negative_count = 0

        index = -1
        fp_example_index = None
        fn_example_index = None
        for target, machine_res in zip(targets, machine_results):
            index = index + 1
            if target[output_bit] == 0:
                if machine_res[output_bit] >= 0.5:
                    false_positive_count = false_positive_count + 1
                    fp_example_index = index
                else:
                    true_negative_count = true_negative_count + 1
            else:
                if machine_res[output_bit] >= 0.5:
                    true_positive_count = true_positive_count + 1
                else:
                    false_negative_count = false_negative_count + 1
                    fn_example_index = index

        print("tp: %3d, fn: %3d, tn: %3d, fp: %3d - %s" % (true_positive_count, false_negative_count, true_negative_count, false_positive_count, features.class_list[output_bit]))
        if fn_example_index is not None and true_positive_count > 0:
            print("System didn't understand that %s is a %s" % (
                    ds_labels[fn_example_index], features.class_list[output_bit]))
        if fp_example_index is not None and true_negative_count > 0:
            print("System wrongly claimed that %s is a %s" % (
                    ds_labels[fp_example_index], features.class_list[output_bit]))
        error = false_negative_count + false_positive_count
        if true_positive_count + false_positive_count:
            precision = true_positive_count / float(true_positive_count + false_positive_count)
        else:
            # No bad results in the (empty) set of positive data.
            precision = 1.0
        if true_positive_count + false_negative_count:
            recall = true_positive_count / float(true_positive_count + false_negative_count)
        else:
            recall = 1.0
#            raise Exception("There are no positive elements in the dataset (bit %d/%s) so this is meaningless." % (output_bit, features.class_list[output_bit]))
            print("There are no positive elements in the dataset (bit %d/%s) so recall is meaningless." % (output_bit, features.class_list[output_bit]))
        if precision + recall:
            F1 = 2 * precision * recall / (precision + recall)
        else:
            # Only classified negative results as positive. Lousy.
            F1 = 0
        total_error = total_error + error;
        total_F1s.append(F1)
    error_pct = 100 * float(total_error) / count
    return (error_pct, total_F1s)

SUPPORTED_TAGS = set([
        "action & adventure",
        "adult",
        "christian",
        "drama",
        "erotica",
        "fantasy",
        "fiction",
        "history",
        "humor",
        "mystery",
        "non-fiction",
        "romance",
        "science fiction",
        "war",
        "western",
        ])

INTERESTING_CLASSES = [
    "christian",
#    "erotica",
#    "fantasy",
#    "history",
#    "mystery",
#    "romance",
#    "science fiction",
#    "war",
#    "western",
    ]

MEANINGLESS_WORDS = (
    "",
    "-",
    ".",
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "for",
    "from",
    "get",
    "going",
    "got",
    "had",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "one",
    "or",
    "out",
    "s",
    "so",
    "that",
    "the",
    "then",
    "there",
    "t",
    "to",
    "was",
    "were",
    "what",
    "where",
    "which",
    "while",
    "who",
    "with",
    "would",
    "~",


    "he", "i", "his", "you", "her", "she", "my", "they", "up", "not", "him",
    "this", "me",
 "we", "down", "over", "your", "when", "no", "them", "some", "its", "their", "know", "said.", "off", "other", "think", "it.", "than", "two", "him.",

    "monday", "tuesday", "wednesday", "thursday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    )

MEANINGLESS_WORDS = ()

def load_base_classes_data(data_files):
    return data_files.keys()

def read_text(text_file):
    with codecs.open(text_file, "r", "utf-8", errors="ignore") as f:
        return f.read()

def load_base_word_data(data_files):
    words = Counter()
    word_stats = {}
    if (os.path.isfile("landet.words.json") and
        os.path.isfile("landet.word_stats.json")
        ):
        print("Using word cache")
        try:
            with codecs.open("landet.words.json", "r", "utf-8") as f:
                words = Counter(json.load(f))
            with codecs.open("landet.word_stats.json", "r", "utf-8") as f:
                word_stats = json.load(f)
        except ValueError:
            print("Something wrong with the cache.")

    if not words or not word_stats:
        # for text, text_classes in textclasses:
        #     text_words = Features.make_word_list_of_text(text)
        #     words.update(text_words)

        txt_file_count = sum(len(x) for x in data_files.itervalues())
        scanned_count = 0
        words_per_book = [] # Array of dicts
        for language, file_list in data_files.iteritems():
            for txt_file in file_list:
                scanned_count = scanned_count + 1
                print("Scanning [%d/%d] %s" %
                      (scanned_count, txt_file_count, os.path.basename(txt_file)))
                text = read_text(txt_file)
                text_words = Features.make_word_list_of_text(text)
                counter_for_book = Counter(text_words)
                words_per_book.append(counter_for_book)
                words.update(counter_for_book)

        print("Calculating mean and stdev for each word (%d)" % len(words))
        total_book_word_count = []
        for book_counter in words_per_book:
            total_book_word_count.append(sum(book_counter.itervalues()))
        for word in words.iterkeys():
            freq = []
            for (book_counter, book_word_count) in zip(words_per_book, total_book_word_count):
                freq.append(float(book_counter.get(word, 0)) / book_word_count)
            freq_mean = numpy.mean(freq)
            freq_stddev = numpy.std(freq)
            word_stats[word] = [freq_mean, freq_stddev]

        for i in xrange(len(words_per_book)):
            words_per_book[i] = dict(words_per_book[i])
        print("Saving word cache")
        with codecs.open("landet.words.json", "w", "utf-8") as f:
            json.dump(dict(words), f)
        with codecs.open("landet.word_stats.json", "w", "utf-8") as f:
            json.dump(word_stats, f)

    return words, word_stats

def load_base_feature_data(data_files):
    words, word_stats = load_base_word_data(data_files)
    classes = load_base_classes_data(data_files)

    return words, classes, word_stats

def build_features(data_files):
    words, classes, word_stats = load_base_feature_data(data_files)

    print("Number of different words: %d" % len(words))
#    print(words.most_common(1000))

    # Numbers? (TODO)

    print("Number of different words after filtering: %d" % len(words))
    print("100 most commonly used word:")
    print(words.most_common(100))
    print("The classes/tags we support")
    print(classes)

    words_to_use_as_features = set(
        [x for x, y in words.most_common(MAX_FEATURE_COUNT / 2)])
    words_to_use_as_features.update(random.sample(list(words),
                                                  MAX_FEATURE_COUNT/2))
    assert all([x in words for x in words_to_use_as_features])
#    words_to_use_as_features = words_to_use_as_features - set(PROMISING_FANTASY) # Trying to see if it can find other words.
    word_list = list(words_to_use_as_features)
    word_index = range(len(word_list))
    if len(word_list) < 200:
        word_list_for_display = ", ".join(word_list)
    else:
        word_list_for_display = ", ".join(word_list[:100]) + ", ..., " + ", ".join(word_list[-100:])
    verbose_output_string = "words (%d) we care about: %s" % (
        len(word_list), word_list_for_display)
    print(codecs.encode(verbose_output_string, "utf-8", "ignore"))
    word_to_index = dict(zip(word_list, word_index))
#    print(word_to_index)

    class_list = data_files.keys()

    class_index = range(len(class_list))
    class_to_index = dict(zip(class_list, class_index))
    print("Class to index:")
    print(class_to_index)

    features = Features(word_list, word_to_index, class_list, class_to_index)

    for i, word in enumerate(word_list):
        features.word_freq_mean[i] = word_stats[word][0]
        features.word_freq_std[i] = word_stats[word][1]

#    print(features.word_list)
#    print(features.word_freq_mean)
#    print(features.word_freq_std)
    return features

def filter_tags(raw_tags, class_to_index):
    tags = []
    for raw_tag in raw_tags:
        raw_tag = raw_tag.lower()
        if raw_tag in class_to_index:
            tags.append(raw_tag)
    return tags

def convert_text_to_sample(text, language, features):
    freq = features.map_to_frequency(text)
    target = features.map_to_classes(language)

    return (freq, target)


class AnalyzeThread(threading.Thread):
    def __init__(self, file_tuple_list, features):
        threading.Thread.__init__(self)
        print("Thread %s will analyze %d files" % (self, len(file_tuple_list)))
        self._file_tuple_list = file_tuple_list
        self._features = features
        self._samples = []

    def run(self):
        self._samples = analyze_inner_thread_fn(self._file_tuple_list,
                                                self._features)

# def analyze_inner_multiprocess_fn(arg):
#     file_tuple_list, features = arg

def analyze_inner_thread_fn(file_tuple_list, features):
    samples = []
    count = len(file_tuple_list)
    for index in xrange(count):
        txt_file, language = file_tuple_list[index]
        sample_label = "%s [%s]" % (os.path.basename(txt_file), language)
        print("Analyzing [%d/%d] %-50s   \r" %
              (index + 1, count, sample_label), end="")
        text = read_text(txt_file)

        sample = convert_text_to_sample(text, language, features)
        samples.append((sample[0], sample[1], sample_label))
    return samples

def get_samples_from_txt_meta(txt_and_meta_files, features, max_count):
    count = len(txt_and_meta_files)
    if max_count > 0 and count > max_count:
        count = max_count

    samples = []

    threads = []
    thread_count = 2 # Python doesn't really support multithreads due
                     # to the global interpreter lock
    # multiprocess_pool = multiprocessing.Pool(thread_count)
    # multiprocess_args = []
    # start = 0
    # for index in range(thread_count):
    #     end = int(count * (index + 1) / thread_count)
    #     multiprocess_args.append((txt_and_meta_files[start:end], features))
    # res = multiprocess_pool.map(analyze_inner_thread_fn,
    #                             multiprocess_args)
    # print(res)
    start = 0
    for index in range(thread_count):
        end = int(count * (index + 1) / thread_count)
        thread = AnalyzeThread(txt_and_meta_files[start:end], features)
        start = end
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
        samples.extend(thread._samples)
        print("Analyzed [%d/%d]  \r" %
              (len(samples), count), end="")

    return samples

if __name__ == "__main__":
    main()
