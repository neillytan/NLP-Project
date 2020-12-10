import mxnet as mx
import numpy as np

class Decoder():
    def __init__(self, model, mode='greedy', n=1, context=mx.cpu()):
        self.model = model
        self.mode = mode
        self.n = n
        self.context = context
    
    def decode(self, input_seq, output_length=1):
        hidden = self.model.begin_state(func=mx.nd.zeros, batch_size=1, 
            ctx=self.context)
        for i in range(len(input_seq)):
            seed = mx.nd.array(input_seq[i], ctx=self.context).reshape((1,1))
            _, hidden = self.model(seed, hidden)
            
        if self.mode == 'greedy':
            return self.greedy_decoder(seed, hidden, output_length)
        if self.mode == 'sample':
            return self.sample_decoder(seed, hidden, output_length)
        if self.mode == 'beam_search':
            return self.beam_search_decoder(seed, hidden, output_length)
        
    def greedy_decoder(self, seed, hidden, output_length):
        output_seq = []
        ppensity = 0
        for i in range(output_length):
            output, hidden = self.model(seed, hidden)
            seed = np.random.choice(len(output[0]), 1, 
                p=mx.nd.softmax(output[0]).asnumpy())
            ppensity += output[0][seed].asscalar()
            seed = mx.nd.array([seed]).reshape((1,1))
            #print(output, output.max())
            output_seq.append(seed[0].asscalar())
        return output_seq, ppensity

    def sample_decoder(self, seed, hidden, output_length):
        output_seq = []
        ppensity = np.zeros(self.n)
        for k in range(self.n):
            output_seq_current, ppensity[k] = self.greedy_decoder(seed, 
                hidden, output_length)
            output_seq.append(output_seq_current)
        return output_seq[ppensity.argmax()], ppensity
    
    def beam_search_decoder(self, seed, hidden, output_length):
        # TODO
        return