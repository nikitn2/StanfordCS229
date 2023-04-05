import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    # Split the message on spaces
    sp = message.split(" ")
    # Convert everything to lower cases
    return [x.lower() for x in sp]
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    count = dict()
    # Create a counts for messages, counting the number of occurences.
    for msg in messages:
        words = set(get_words(msg))  # As we are counting the number of messages
        # ... containing certain words, this prevents double counting.
        for w in words:
            if w in count:
                count[w] += 1
            else:
                count[w] = 1
    # Filter infrequent words
    freq = list()
    for w, num_occur in count.items():
        if num_occur >= 5:
            freq.append(w)
    # Construct dictionary
    dictionary = dict()
    for i, w in enumerate(freq):
        dictionary[w] = i
    return dictionary
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    num_msg = len(messages)
    vocab_size = len(word_dictionary)
    freq_array = np.zeros([num_msg, vocab_size]).astype(int)
    for i, msg in enumerate(messages):
        sp = get_words(msg)
        for word in sp:
            if word in word_dictionary:
                j = word_dictionary[word]
                freq_array[i, j] += 1
    return freq_array
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    NOTE: the trained model is a dictionary with fields vocab @ (2, V), class @ (2,)
    vocab[i, k] gives P(x[j]=k | y=i), i in {0, 1}
    class[i] gives P(y=i), i in {0, 1}
    """
    # *** START CODE HERE ***
    N, V = matrix.shape
    assert N == len(labels)
    model = dict()
    phi_vocab = np.zeros([2, V])
    phi_class = np.zeros([2])
    # phi(k|y)
    for y in (0, 1):
        X = matrix[labels == y]
        total = np.sum(X, axis=0)
        denominator = np.sum(total) + V
        assert len(total) == V
        total += 1  # Laplace Smoothing
        assert np.sum(total) == denominator  # two methods should give the same denominator
        prob = total / denominator
        assert np.abs(np.sum(prob) - 1.0) < 1e-10, f"received: {np.sum(prob)}"
        phi_vocab[y, :] = prob
    assert np.all(
        (np.sum(phi_vocab, axis=1) - np.ones([2]) <= 1e-10)
        )
    # phi(y)
    phi_class[0] = np.mean(labels == 0)
    phi_class[1] = np.mean(labels == 1)
    assert np.sum(phi_class) == 1
    # Construct model
    model["vocab"] = phi_vocab
    model["class"] = phi_class
    return model
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    N, V = matrix.shape
    phi_vocab, phi_class = model["vocab"], model["class"]
    assert phi_vocab.shape[1] == V
    probs = np.zeros([N, 2])
    for i in range(N):
        word = matrix[i, :]  # The word representation.
        for y in (0, 1):
            probs[i, y] = np.log(phi_class[y]) + np.dot(
                np.log(phi_vocab)[y, :],
                matrix[i, :]
            )
    pred = (probs[:, 1] > probs[:, 0]).astype(int)
    return pred
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi_vocab, phi_class = model["vocab"], model["class"]
    influence = list()
    for word, index in dictionary.items():
        score = np.log(phi_vocab[1, index] / phi_vocab[0, index])
        influence.append((word, score))
    influence.sort(key=lambda x: -x[1])
    top = [word for word, _ in influence[:5]]
    assert len(top) == 5
    return top
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    results = list()
    for r in radius_to_consider:
        model = svm.svm_train(train_matrix, train_labels, r)
        pred = svm.svm_predict(model, val_matrix, r)
        accuracy = np.mean(pred == val_labels)
        print(f"Radius: {r}, Accuracy: {accuracy}")
        record = {"r": r, "acc": accuracy}
        results.append(record)
    results.sort(key=lambda x: -x["acc"])
    return results[0]["r"]
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
