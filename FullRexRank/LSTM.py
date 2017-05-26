import tensorflow as tf
from textloader import TextLoader
from tqdm import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.rnn_cell import BasicLSTMCell as lstm
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.seq2seq import rnn_decoder, sequence_loss
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.rnn_cell import _linear
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops


class mygru( RNNCell ):

    def __init__(self, cells, state_is_tuple=True):
        self.size = cells
        self.activation = tanh

    @property
    def state_size(self):
        return self.size

    @property
    def output_size(self):
        return self.size

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):
            h = state
            concat = _linear([inputs, h], 2 * self.size, False)
            r,z = array_ops.split(1,2,concat)
            r = sigmoid(r)
            z = sigmoid(z)
        with vs.variable_scope('h_2'):
            h_2 = self.activation(_linear([inputs, r*h], self.size, False))
            new_h = z* h + (1-z) * h_2

        return new_h, new_h

class TextLSTM():
    '''
    Defines a class of LSTM objects that will build an LSTM model that, once trained, can generate samples fairly quickly.
    The downside of this approach for text generation, and especially wikipedia articles is that it doesn't have any 
    concept of content. It is mostly just piecing togethr sensical strings of charchters. We'll see if we can assist it.
    '''
    def __init__(self,data_name, sample_size=300,iters=100, learning_rate=.01):
        self.sample_size = sample_size
        self.data_name = data_name
        self.batch_size=100
        self.sequence_length = 10
        self.iters=iters
        self.learning_rate = learning_rate
        
        self.data_loader = TextLoader( ".", self.batch_size, self.sequence_length )
        
        self.vocab_size = self.data_loader.vocab_size  # dimension of one-hot encodings
        self.create_graph()
        print(self.sample(num=self.sample_size, prime=self.data_name))


    def create_graph(self): 
        s_batch_size = 1
        state_dim = 128
        num_layers = 2
        tf.reset_default_graph()
        self.in_ph = tf.placeholder( tf.int32, [ self.batch_size, self.sequence_length ], name='inputs' )
        self.targ_ph = tf.placeholder( tf.int32, [ self.batch_size, self.sequence_length ], name='targets' )
        in_onehot = tf.one_hot( self.in_ph, self.vocab_size, name="input_onehot" )
        
        inputs = tf.split( 1, self.sequence_length, in_onehot )
        inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
        targets = tf.split( 1, self.sequence_length, self.targ_ph )
        
        weights = tf.Variable(tf.random_normal([state_dim, self.vocab_size]), name='weights')
        bias = tf.Variable(tf.random_normal([self.vocab_size]), name='bias')
        cell1 = mygru(state_dim, state_is_tuple=False)
        cell2 = mygru(state_dim, state_is_tuple=False)
        
        Mcell = MultiRNNCell([cell1,cell2], state_is_tuple=True)
        self.initial_state = Mcell.zero_state(self.batch_size, tf.float32)
        
        outputs, final_state = rnn_decoder(inputs, self.initial_state, Mcell)
        
        logits = [tf.matmul(output, weights) + bias for output in outputs]
        
        one_weights = [1. for l in range(len(logits))]
        
        loss = sequence_loss(logits, targets, one_weights)
        
        optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        
        tf.get_variable_scope().reuse_variables()
        
        self.s_in_ph = tf.placeholder( tf.int32, [s_batch_size], name='s_inputs' )
        
        s_in_onehot = tf.one_hot( self.s_in_ph, self.vocab_size, name="s_input_onehot" )
        s_inputs = s_in_onehot
        
        self.s_initial_state = Mcell.zero_state(s_batch_size, tf.float32)
        
        s_outputs, self.s_final_state = rnn_decoder([s_inputs], self.s_initial_state, Mcell)
        
        s_logits = [tf.matmul(output, weights) + bias for output in s_outputs]
        self.s_probs = tf.nn.softmax(s_logits[0])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.sess = tf.Session(config=config)
        self.sess.run( tf.global_variables_initializer() )
        summary_writer = tf.train.SummaryWriter( "./tf_logs", graph=self.sess.graph )

        self.lts = []
        last=0
        
        for j in tqdm(range(self.iters)):
        
            state = self.sess.run( self.initial_state )
            self.data_loader.reset_batch_pointer()
        
            for i in range(self.data_loader.num_batches ):
                
                x,y = self.data_loader.next_batch()
        
                # we have to feed in the individual states of the MultiRNN cell
                feed = { self.in_ph: x, self.targ_ph: y }
                for k, s in enumerate( self.initial_state ):
                    feed[s] = state[k]
        
                ops = [optim,loss]
                ops.extend( list(final_state) )
        
                # retval will have at least 3 entries:
                # 0 is None (triggered by the optim op)
                # 1 is the loss
                # 2+ are the new final states of the MultiRNN cell
                retval = self.sess.run( ops, feed_dict=feed )
        
                lt = retval[1]
                state = retval[2:]
        
                if i%1000==0 and j%20==0:

                    #print("{} {}\t{}\t{}".format( j, i, lt,self.learning_rate ))
                    #print(self.sample(num=self.sample_size, prime=self.data_name))
                    self.lts.append( lt )
                    last +=1
                    #print(last)
                    if len(self.lts) > 5 and abs(self.lts[-2]-self.lts[-1]) < .01 and last>10 and self.learning_rate > .0009:
                        self.learning_rate *= .6
                        last = 0
        
        
        summary_writer.close()
        pass

    def sample(self, num, prime ):
        '''
        This method will allow you to sample from the LSTM.
        num - Length of string in chrs you wish returned
        prime - The intial state that you want for generation. For simplicity, I set it to the name of the article.
        '''
    
        # prime the pump 
    
        # generate an initial state. this will be a list of states, one for
        # each layer in the multicell.
        s_state = self.sess.run( self.s_initial_state )
    
        # for each character, feed it into the sampler graph and
        # update the state.
        for char in prime[:-1]:
            x = np.ravel( self.data_loader.vocab[char] ).astype('int32')
            feed = { self.s_in_ph:x }
            for i, s in enumerate( self.s_initial_state ):
                feed[s] = s_state[i]
            s_state = self.sess.run( self.s_final_state, feed_dict=feed )
    
        # now we have a primed state vector; we need to start sampling.
        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.ravel( self.data_loader.vocab[char] ).astype('int32')
    
            # plug the most recent character in...
            feed = { self.s_in_ph:x }
            for i, s in enumerate( self.s_initial_state ):
                feed[s] = s_state[i]
            ops = [self.s_probs]
            ops.extend( list(self.s_final_state) )
    
            retval = self.sess.run( ops, feed_dict=feed )
    
            self.s_probsv = retval[0]
            s_state = retval[1:]
    
            # ...and get a vector of probabilities out!
    
            # now sample (or pick the argmax)
            # sample = np.argmax( s_probsv[0] )
            sample = np.random.choice( self.vocab_size, p=self.s_probsv[0] )
    
            pred = self.data_loader.chars[sample]
            ret += pred
            char = pred
        self.plot()
        return ret

        
    def plot(self):
        plt.plot( self.lts )
        plt.savefig('{}.png'.format(self.data_name))
        plt.show()
