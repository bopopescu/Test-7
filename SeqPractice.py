import  numpy as np
import tensorflow as tf
#import  helpers


tf.reset_default_graph()
sess = tf.InteractiveSession()

print tf.COMPILER_VERSION

# for making sequence of same size
PAD =0
#End Of Sentence
EOS=1

vocab_size=10
input_embedding_size=20 #lenght of Characters


encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units*2

#placeholder
encoder_inputs = tf.placeholder(shape=(None,None), dtype=tf.int32,name ='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32,name ='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None,None), dtype=tf.int32,name ='decoder_targets')


embeddings = tf.Variable(tf.random_uniform([vocab_size,input_embedding_size],-1.0,1),dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings,encoder_inputs)

from tensorflow.python.ops.rnn_cell import LSTMCell,LSTMStateTuple

encoder_cell=LSTMCell(encoder_hidden_units)

((encoder_fw_outputs,
  encoder_bw_outputs),
  (encoder_fw_final_state,
   encoder_bw_final_state))= (
     tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                     cell_bw=encoder_cell,
                                     inputs=encoder_inputs_embedded,
                                     sequence_length=encoder_inputs_length,
                                     dtype=tf.float32, time_major=True))
print 'hello'


