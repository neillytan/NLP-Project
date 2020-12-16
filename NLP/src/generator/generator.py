import mxnet as mx
import nltk
from mxnet import gluon

from NLP.src.decoder import decoder
from NLP.src.pre_processing import pre_process
from NLP.src.training import model, training, utils


class Generator:
    """
    Using user provided input corpus, train an RNN and generates fake texts
    """

    def __init__(self, input_path, epoch=30, rnn_type='gru', num_embed=5,
                 num_hidden=5, num_layers=2, dropout=0, lr=1, batch_size=32,
                 use_pretrained_embedding=False, freeze_embedding=False, tie_weights=False,
                 context=mx.cpu()):
        """
        param input_path: path of input corpus
        param epoch: number of epochs for model training
        param rnn_type: type of RNN used, 'gru', 'lstm', 'rnn_relu' or 'rnn_tanh'
        param num_embed: embedding dimension
        param num_hidden: RNN hidden layer dimension
        param num_layers: number of RNN layers
        param dropout: between 0 and 1, dropout probability
        param lr: learning rate of sgd (current version defaults to sgd trainer)
        param batch_size: batch size
        param use_pretrained_embedding: use pretrained GloVe embeddings to initialize
            the input embedding layer
        param freeze_embedding: freeze training of embedding layers (input and output).
            Only do this when use_pretrained_embedding=True and tie_weights=True!
        param tie_weights: tie weights on input and output embedding layers
        param context: cpu (default) or gpu
        """
        self.context = context
        seq, self.word_idx, self.idx_word = pre_process.pre_process(input_path)
        self.seq = mx.nd.array(seq, ctx=self.context)
        self.vocab_size = len(self.word_idx)

        train_data = utils.batchify(self.seq, batch_size).as_in_context(self.context)
        if use_pretrained_embedding:
            # we use 25 dimensional pretrained embedding, so force this to 25
            num_embed = 25

        self.model_ = model.RNNModel(mode=rnn_type, vocab_size=self.vocab_size,
                                     num_embed=num_embed, num_hidden=num_hidden,
                                     num_layers=num_layers, dropout=dropout,
                                     tie_weights=tie_weights)
        self.model_.collect_params().initialize(mx.init.Xavier(), ctx=self.context)

        if use_pretrained_embedding:
            embedding_weights = utils.get_pretrained_weights(self.idx_word)
            self.model_.encoder.weight.set_data(embedding_weights)

        trainer = gluon.Trainer(self.model_.collect_params(), 'sgd',
                                {'learning_rate': lr, 'momentum': 0, 'wd': 0})
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        training.train(train_data=train_data, model=self.model_, trainer=trainer,
                       loss=loss, epochs=epoch, batch_size=batch_size,
                       freeze_embedding=freeze_embedding,
                       context=self.context)

    def decode(self, input_seq, decoder_mode='greedy', output_length=1,
               get_next_probability=False, sample_count=1, get_next_probability_count=20):
        """
        Decode the learned RNN and generate fake text
        param input_seq: string that the fake text is supposed to begin with
        param decoder_mode: decoder mode,
            'greedy': samples next word using the multinomial distribution predicted by the RNN,
                not a true argmax 'greedy' because it behaves very poorly.
            'sample': samples the whole output sequence for sample_count number of times and pick
                the one with highest propensity
        param output_length: length of the output fake sentence
        param get_next_probability: output the top get_next_probability_count probable word following
            the input_seq, as predicted by the RNN
        param sample_count: only work when decoder_mode='sample', number of samples generated
        param get_next_probability_count: only work when get_next_probability=True, number of top
            probable words to be outputted
        return: generated fake text followed by the input string
        """
        decoder_ = decoder.Decoder(self.model_, self.context)
        input_list = self.tokenize(input_seq)
        seq_output, ppensity = decoder_.decode(input_list, output_length, decoder_mode, sample_count,
                                               get_next_probability, self.word_idx,
                                               self.idx_word, get_next_probability_count=get_next_probability_count)
        return ' '.join([self.idx_word[idx] for idx in input_list + seq_output])

    def tokenize(self, input_seq):
        """
        Tokenize the input sequence and map it into a integer sequence
        param input_seq: string of input words, must be words from the input corpus
        return: list of integers corresponding to the input_seq
        """
        tokens = nltk.word_tokenize(input_seq)
        tokens = [w.lower() for w in tokens if w.isalpha()]
        try:
            return [self.word_idx[i] for i in tokens]
        except:
            raise ValueError('Provided word not in vocabulary')
