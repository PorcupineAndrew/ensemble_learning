import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models import Word2Vec

class TextProcessing:
    """
    Processing util for working with text data.
    """
    def __init__(self, texts, labels, EMBEDDING_DIM=300, MAX_SEQUENCE_LENGTH=100, MAX_NB_WORDS=20000, VALIDATION_SPLIT=0.0):
        """
        Instantiates the class.

        Parameters
        ----------
        texts: A list (numpy array) containing texts of the docs to be classified.
        labels: A list containing labels for the docs to be classified.
        EMBEDDING_DIM: Dimension of the word embedding. Default is 300.
        MAX_SEQUENCE_LENGTH: Maximum length of the document. Default is 100.
        MAX_NB_WORDS: Maximum number of words-tokens to be considered. Default is 20000.
        VALIDATION_SPLIT: The fraction of the data to be used as validation.
        """
        self.texts = texts
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.labels_index = dict((x,i) for i,x in enumerate(np.sort(np.unique(labels))))
        self.labels = [self.labels_index[x] for x in labels]
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        self.tokenizer_string = None


    def preprocess(self, tokenizer_string=None):
        """
        Preprocess the textual data.

        Returns
        -------
        x_train: The processed-sequenced training data.
        y_train: Processed training labels
        x_val: The processed-sequenced validation data
        y_val: processed validation labels
        word_index: A dictionary containing the word-tokens and their indices for the sequencing.
        """
        if tokenizer_string is None:
            tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
            tokenizer.fit_on_texts(self.texts)
            self.tokenizer_string = tokenizer.to_json()
        else:
            self.tokenizer_string = tokenizer_string
            from keras.preprocessing.text import tokenizer_from_json
            tokenizer = tokenizer_from_json(tokenizer_string)
        sequences = tokenizer.texts_to_sequences(self.texts)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)
        labels = to_categorical(np.asarray(self.labels))

        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        # split the data into a training set and a validation set
        if (self.VALIDATION_SPLIT):
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)

            data = data[indices]
            labels = labels[indices]
            num_validation_samples = int(self.VALIDATION_SPLIT * data.shape[0])

            x_train = data[:-num_validation_samples]
            y_train = labels[:-num_validation_samples]
            x_val = data[-num_validation_samples:]
            y_val = labels[-num_validation_samples:]
        else:
            x_train = data
            y_train = labels
            x_val = None
            y_val = None

        return x_train, y_train, x_val, y_val, word_index


    def build_embedding_index_from_word2vec(self, fname):
        """
        Build an index of the word embeddings using google word2vec.

        Parameters
        ----------
        fname: Path to the file containing word2vecs model.
        vocab: A dict containing words with indices.

        Returns
        -------
        A dict containing words in the vocab and their word vecs.
        """
        print('Indexing word vectors.')

        w2v_model = Word2Vec.load(str(fname))
        return w2v_model
