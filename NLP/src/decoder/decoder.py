import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt


class Decoder:
    """
    Decode the given RNN model and generate fake text
    """
    def __init__(self, model, context=mx.cpu()):
        """
        param model: RNN model for decoding
        param context: only work when decoder_mode='sample', number of samples generated
        """
        self.model = model
        self.context = context

    def decode(self, input_seq, output_length=1, mode='greedy', sample_count=1,
               get_next_probability=False, word_idx=None, idx_word=None, get_next_probability_count=20):
        """
        Decode the RNN model
        param input_seq: list of integers that the RNN decoding is supposed to begin with
        param output_length: length of output list
        param mode: decoder mode,
            'greedy': samples next word using the multinomial distribution predicted by the RNN,
                not a true argmax 'greedy' because it behaves very poorly.
            'sample': samples the whole output sequence for sample_count number of times and pick
                the one with highest propensity
        param sample_count: only work when decoder_mode='sample', number of samples generated
        param get_next_probability: output the top get_next_probability_count probable word following
            the input_seq, as predicted by the RNN
        param word_idx: word -> integer mapping for the RNN, only needed when get_next_probability=Ture
        param idx_word: integer -> word mapping for the RNN, only needed when get_next_probability=Ture
        param get_next_probability_count: only work when get_next_probability=True, number of top
            probable words to be outputted
        return: generated fake integer list followed by the input list
        """
        hidden = self.model.begin_state(func=mx.nd.zeros, batch_size=1,
                                        ctx=self.context)
        for i in range(len(input_seq)):
            seed = mx.nd.array(input_seq[i], ctx=self.context).reshape((1, 1))
            _, hidden = self.model(seed, hidden)

        if get_next_probability:
            if not word_idx or not idx_word:
                raise ValueError('word index and/or index word dictionary not passed')
            self.get_next_probability(input_seq, seed, hidden, word_idx, idx_word, get_next_probability_count)
        if mode == 'greedy':
            return self.greedy_decoder(seed, hidden, output_length)
        if mode == 'sample':
            return self.sample_decoder(seed, hidden, output_length, sample_count)
        if mode == 'beam_search':
            return self.beam_search_decoder(seed, hidden, output_length)

    def greedy_decoder(self, seed, hidden, output_length):
        """
        greedy decoder (sampled version), samples k next elements using the
        RNN generated multinomial distribution given the initial input and
        hidden state
        param seed: initial input
        param hidden: input hidden state of the RNN
        param output_length: length of output
        return: tuple of (RNN generated integer list, propensity of this list)
        """
        output_seq = []
        ppensity = 0
        for i in range(output_length):
            output, hidden = self.model(seed, hidden)
            seed = np.random.choice(len(output[0]), 1,
                                    p=mx.nd.softmax(output[0]).asnumpy())
            ppensity += output[0][seed].asscalar()
            seed = mx.nd.array([seed]).reshape((1, 1))
            output_seq.append(seed[0].asscalar())
        return output_seq, ppensity

    def sample_decoder(self, seed, hidden, output_length, sample_count=1):
        """
        Given the initial input and hidden state, sample sequence of given length
        in one go, multiple times using RNN predicted distribution, then pick the one with
        the highest propensity
        param seed: initial input
        param hidden: initial hidden state
        param output_length: length of output
        param sample_count: number of samples generated before picking the one with highest
            propensity
        return: (output sequence, list of propensity for all sampled sequences)
        """
        output_seq = []
        ppensity = np.zeros(sample_count)
        for k in range(sample_count):
            output_seq_current, ppensity[k] = self.greedy_decoder(seed,
                                                                  hidden, output_length)
            output_seq.append(output_seq_current)
        return output_seq[ppensity.argmax()], ppensity

    # def beam_search_decoder(self, seed, hidden, output_length):
        # TODO
        # return

    def get_next_probability(self, input_seq, seed, hidden, word_idx, idx_word, k):
        """
        Plot the probability of next word (top K probable) given the input sequence
        param input_seq: list of words that the RNN decoding is supposed to begin with.
            Note: this parameter is here solely for printing on the plot. We need to know
            the initial input and hidden state corresponding to this input sequence and
            they are generated above in the decode method.
        param seed: initial input
        param hidden: initial hidden state
        param word_idx: word -> integer mapping for the RNN, only needed when get_next_probability=Ture
        param idx_word: integer -> word mapping for the RNN, only needed when get_next_probability=Ture
        param k: top k probable words only. Note: This is also capped by the size of vocabulary and
            will not exceed 30 to avoid overloaded plot.
        """
        input_word = [idx_word[i] for i in input_seq]
        input_word = ' '.join(input_word)
        output, hidden = self.model(seed, hidden)
        output = mx.nd.softmax(output[0]).asnumpy()
        k = min(len(word_idx), 30, k)
        max_k_index = np.argsort(output)[-k:]
        label = np.array([idx_word[i] for i in max_k_index])
        plt.plot(label, output[max_k_index])
        plt.xticks(rotation=45)
        plt.title('Probability for next word of "{}"'.format(input_word))
        plt.grid()
        return
