import os
import re
import codecs


def zero_digits(s):
    """
    Replace all digits in a string with zeros.
    """
    return re.sub("\d", "0", s)


def load_sentences(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, "r", "utf8"):
        line = zero_digits(line.rstrip())
        if not line:
            if len(sentence) > 0:
                if "DOCSTART" not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if "DOCSTART" not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items) if v[1] > 2}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def word_mapping(sentences):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico["<UNK>"] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print(
        "Found %i unique words (%i in total)" % (len(dico), sum(len(x) for x in words))
    )
    return dico, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, mode=None, word_to_id=None, tag_to_id=None):
    """
    Prepare the dataset. Return 'data', which is a list of lists of dictionaries containing:
        - words (strings)
        - word indexes
        - tag indexes
    """
    assert mode == "train" or (mode == "test" and word_to_id and tag_to_id)

    if mode == "train":
        word_dic, word_to_id, id_to_word = word_mapping(sentences)
        tag_dic, tag_to_id, id_to_tag = tag_mapping(sentences)

    def f(x):
        return x

    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else "<UNK>"] for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append(
            {
                "str_words": str_words,
                "words": words,
                "tags": tags,
            }
        )

    if mode == "train":
        return data, {
            "word_to_id": word_to_id,
            "id_to_word": id_to_word,
            "tag_to_id": tag_to_id,
            "id_to_tag": id_to_tag,
        }
    else:
        return data


# MEMM Model
import os
import re
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import collections
import codecs


class MEMM(object):
    """
    MEMM Model
    """

    def __init__(
        self,
        dic,
        decode_type,
        regex_features,
        rear_context_window=4,
        forward_context_window=4,
        num_prior_tags=6,
    ):
        """
        Initialize the model.
        """

        self.num_words = len(dic["word_to_id"])
        self.num_tags = len(dic["tag_to_id"])

        self.rear_context_window = rear_context_window
        self.forward_context_window = forward_context_window
        self.num_prior_tags = num_prior_tags

        self.decode_type = decode_type
        tag_labels = list(dic["id_to_tag"].keys()) + [
            -1
        ]  # add [-1] for out of bounds positions
        self.tag_encoder = OneHotEncoder(handle_unknown="error", sparse_output=False).fit(
            np.array(tag_labels).reshape(-1, 1)
        )
        self.model = linear_model.LogisticRegression(
            penalty="l2", C=0.5, verbose=True, solver="lbfgs", max_iter=500, n_jobs=-1
        )  # MaxEnt model
        self.regular_expressions = collections.OrderedDict(regex_features.items())
        self.dic = dic
        self.word_features_cache = dict()
        self.tag_features_cache = dict()
        return

    def check_regex(self, pattern, word_str):
        return re.search(pattern, word_str) is not None

    def extract_feature(self, words, tags, i):
        """
        TODO: Complete the extract_feature to include tag features from prior words
        Extract word and tag features to predict i'th tag

        Args:
            words: dict, consisting of {k: k'th word in sentence}
            tags: dict, consisting of {k: k'th tag in sentence}
            i: the index of the tag we want to predict with this feature
        """

        # Get features for each word in the context window
        window = [
            words.get(i_x, -1)
            for i_x in range(
                i - self.rear_context_window, i + self.forward_context_window + 1
            )
        ]  # -1 for words out of bounds
        if (
            tuple(window) in self.word_features_cache
        ):  # caching speeds up feature extraction a bit
            word_features = self.word_features_cache[tuple(window)]
        else:
            word_strs = list(
                map(lambda word_id: self.dic["id_to_word"].get(word_id, ""), window)
            )
            word_features = list()
            for word in word_strs:
                for pattern in self.regular_expressions.values():
                    word_features.append(self.check_regex(pattern, word))
        self.word_features_cache[tuple(window)] = word_features

        prior_tags = list()
        # TODO: Set prior_tags to the list of tag ids for the last (self.num_prior_tags) tags
        # (6 points)
        # START HERE
        prior_tags = [tags.get(i_x, -1) for i_x in range(i - self.num_prior_tags, i)]
        # END

        if tuple(prior_tags) in self.tag_features_cache:
            tag_features = self.tag_features_cache[tuple(prior_tags)]
        else:
            tag_features = list()
            # TODO: Add one-hot encoding features to tag_features
            # (6 points)
            # START HERE
            for tag_id in prior_tags:
                one_hot = self.tag_encoder.transform([[tag_id]])
                tag_features.extend(one_hot.flatten())
            # END

        feature = np.append(word_features, tag_features)
        self.tag_features_cache[tuple(prior_tags)] = tag_features
        return feature.reshape(1, -1)

    def get_features_labels(self, sentence):
        """
        Returns the features and labels for each tag in a sentence.
        """
        words = dict(enumerate(sentence["words"]))
        tags = dict(enumerate(sentence["tags"]))
        features = list()
        labels = list()
        for i in range(0, len(tags)):
            feature = self.extract_feature(words, tags, i).flatten()
            label = tags[i]
            features.append(feature)
            labels.append(label)
        return features, labels

    def train(self, corpus):
        """
        Train an MEMM model using MLE estimates.

        Args:
            corpus is a list of dictionaries of the form:
            {'str_words': str_words,   ### List of string words
            'words': words,            ### List of word IDs
            'tags': tags}              ### List of tag IDs
            All three lists above have length equal to the sentence length for each instance.
        """

        X = list()
        y = list()
        print("Extracting features...")
        for sentence in tqdm(corpus):
            features, labels = self.get_features_labels(sentence)
            X.extend(features)
            y.extend(labels)
        print("Training MaxEnt model. This usually finishes within 1-3 minutes.")
        self.model.fit(X, y)

        return

    def greedy_decode(self, sentence):
        """
        Decode a single sentence in Greedy fashion
        Return a list of tags.
        """
        words = dict(enumerate(sentence))
        y_tags = dict()  # stores past tags
        for i in range(0, len(sentence)):
            feature = self.extract_feature(words, y_tags, i)
            y_hat = np.argmax(self.model.predict_proba(feature)).item()
            y_tags[i] = y_hat

        tags = [y_tags[i] for i in range(len(sentence))]
        assert len(tags) == len(sentence)
        return tags

    def tag(self, sentence):
        """
        Tag a sentence using a trained MEMM.
        """
        return self.greedy_decode(sentence)


# Training and Testing Model

import os
import time
import codecs
import json
from tqdm import tqdm
import numpy as np
import collections
from sklearn.metrics import f1_score, confusion_matrix


def tag_corpus(model, test_corpus, output_file, dic):
    if output_file:
        f_output = codecs.open(output_file, "w", "utf-8")
    start = time.time()

    num_correct = 0.0
    num_total = 0.0
    y_pred = []
    y_actual = []
    print("Tagging...")
    for i, sentence in enumerate(tqdm(test_corpus)):
        tags = model.tag(sentence["words"])
        str_tags = [dic["id_to_tag"][t] for t in tags]
        y_pred.extend(tags)
        y_actual.extend(sentence["tags"])

        # Check accuracy.
        num_correct += np.sum(np.array(tags) == np.array(sentence["tags"]))
        num_total += len([w for w in sentence["words"]])

        if output_file:
            f_output.write(
                "%s\n"
                % " ".join(
                    "%s%s%s" % (w, "__", y)
                    for w, y in zip(sentence["str_words"], str_tags)
                )
            )

    print(
        "---- %i lines tagged in %.4fs ----" % (len(test_corpus), time.time() - start)
    )
    if output_file:
        f_output.close()

    print("Overall accuracy: %s\n" % (num_correct / num_total))
    return y_pred, y_actual


def compute_score(y_pred, y_actual):
    A = confusion_matrix(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred, average=None)
    print("Confusion Matrix:\n", A)
    print("F-1 scores: ", f1)


def runMEMM(
    train_corpus,
    test_corpus,
    dic,
    decode_type,
    regex_features,
    rear_context_window,
    forward_context_window,
    num_prior_tags,
    output_file,
):
    # build and train the model
    model = MEMM(
        dic,
        decode_type,
        regex_features=regex_features,
        rear_context_window=rear_context_window,
        forward_context_window=forward_context_window,
        num_prior_tags=num_prior_tags,
    )
    model.train(train_corpus)

    print("Train results:")
    pred, real = tag_corpus(model, train_corpus, output_file, dic)

    print("Tags: ", dic["id_to_tag"])
    A = compute_score(pred, real)

    # test on validation
    print("\n-----------\nTest results:")
    pred, real = tag_corpus(model, test_corpus, output_file, dic)

    print("Tags: ", dic["id_to_tag"])
    A = compute_score(pred, real)


# Download the dataset
import requests

urls = [
    "https://princeton-nlp.github.io/cos484/assignments/a2/eng.train",
    "https://princeton-nlp.github.io/cos484/assignments/a2/eng.val",
]

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

for url in urls:
    response = requests.get(url)
    filename = url.split("/")[-1]
    file_path = os.path.join(current_dir, filename)
    with open(file_path, "wb") as f:
        f.write(response.content)

train_file = os.path.join(current_dir, "eng.train")
test_file = os.path.join(current_dir, "eng.val")

# Load the training data
train_sentences = load_sentences(train_file)
train_corpus, dic = prepare_dataset(
    train_sentences, mode="train", word_to_id=None, tag_to_id=None
)

# Load the testing data
test_sentences = load_sentences(test_file)
test_corpus = prepare_dataset(
    test_sentences,
    mode="test",
    word_to_id=dic["word_to_id"],
    tag_to_id=dic["tag_to_id"],
)

# Get frequencies for common words and their respective tag.
tag_word_examples = {key: collections.Counter() for key in dic["tag_to_id"].keys()}
for sentence in train_corpus:
    for word, tag in zip(sentence["str_words"], sentence["tags"]):
        tag_word_examples[dic["id_to_tag"][tag]][word] += 1

# Show the 20 most common words in each tag class.
for tag, examples in tag_word_examples.items():
    print(f"{tag}: {list(examples.most_common())[:20]}")

regex_features = {
    # 'feature name': r'pattern',  # intuition for these features
    "ENDS_IN_AN": r"(an|AN)$",  # words like Samoan, INDIAN, etc.
    "LEN_TWO_AND_CAPS": r"^[A-Z]{2}$",  # Locations like CA, UK, etc.
    "IS_PERIOD": r"^\.$",  # . is common and usually O
}

runMEMM(
    train_corpus=train_corpus,
    test_corpus=test_corpus,
    dic=dic,
    decode_type="greedy",
    regex_features=regex_features,
    rear_context_window=4,
    forward_context_window=4,
    num_prior_tags=5,
    output_file=None,
)

regex_features_2 = {
    # 'feature name': r'pattern',  # intuition for these features
    "ENDS_IN_AN": r"(an|AN)$",  # words like Samoan, INDIAN, etc.
    "LEN_TWO_AND_CAPS": r"^[A-Z]{2}$",  # Locations like CA, UK, etc.
    "IS_PERIOD": r"^\.$",  # . is common and usually O
    # TODO: Add at least 3 new regular expression features to improve model performance
    # (8 points)
    # START HERE
    "STARTS_WITH_CAPITAL": r"^[A-Z][a-z]+$",  # Proper nouns like names, places
    "CONTAINS_DIGITS": r"\d+",  # Words containing numbers, often part of organizations or misc
    "ALL_CAPS": r"^[A-Z]+$",  # All uppercase words, often organizations or locations
    "CONTAINS_HYPHEN": r"-",  # Hyphenated words, often compound names or organizations
    "ENDS_WITH_ING": r"ing$",  # Words ending with -ing, often not named entities
    "CONTAINS_APOSTROPHE": r"'",  # Words with apostrophes, often possessives
    # END
}

runMEMM(
    train_corpus=train_corpus,
    test_corpus=test_corpus,
    dic=dic,
    decode_type="greedy",
    regex_features=regex_features_2,
    rear_context_window=4,
    forward_context_window=4,
    num_prior_tags=5,
    output_file=None,
)
