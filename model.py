# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time
from itertools import product

# Model modules
from parameters import *
import stimulus
import AdamOpt

# Match GPU IDs to nvidia-smi command
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'




class Model:

    def __init__(self, input_data, target_data, mask):

        self.input_data     = tf.unstack(input_data, axis=0)
        self.target_data    = tf.unstack(target_data, axis=0)
        self.time_mask      = tf.unstack(mask, axis=0)

        self.initialize_variables_and_states()
        self.run_model()
        self.optimize()


    def initialize_variables_and_states(self):

        self.var_dict = {}
        var_prefixes = ['W_in', 'W_rnn', 'W_out', 'b_rnn', 'b_out']
        for scope in par['scopes']:
            self.scope_dict = {}
            with tf.variable_scope(scope):
                for name in var_prefixes:
                    self.scope_dict[name] = tf.get_variable(name, initializer=par['var_inits'][scope][name])
                    if scope in par['input_scopes'] and 'out' in name:
                        self.scope_dict[name+'_'] = tf.get_variable(name+'_', initializer=par['var_inits'][scope][name])
            self.var_dict[scope] = self.scope_dict


        self.state_dict = {}
        for inp in par['input_scopes']:
            self.scope_dict = {}
            for out in ['self'] + par['output_scopes']:
                self.scope_dict[out] = [par['hid_inits'][inp][out]]
            self.state_dict[inp] = self.scope_dict


    def recurrent_layer(self, var_scope, x, h):

        W_in  = self.var_dict[var_scope]['W_in']
        W_rnn = self.var_dict[var_scope]['W_rnn']
        b_rnn = self.var_dict[var_scope]['b_rnn']

        W_rnn_eff = par['EI_matrix'][var_scope] @ tf.nn.relu(W_rnn) if par['EI'] else W_rnn
        h_post = h

        h = tf.nn.relu((1-par['alpha_neuron'])*h \
         + par['alpha_neuron']*(x @ W_in + h_post @ W_rnn_eff + b_rnn) \
         + tf.random_normal(h.shape, 0., par['noise_rnn'], dtype=tf.float32))

        return h


    def run_model(self):

        for t in range(par['num_time_steps']):

            self.outputs_dict = {}

            for input_scope in par['input_scopes']:

                if input_scope == 'encoder':
                    x = self.input_data[t]
                elif input_scope == 'generator':
                    x = tf.random_normal(shape=[par['batch_size'],par['n_generator']])

                W_mu  = self.var_dict[input_scope]['W_out']
                b_mu  = self.var_dict[input_scope]['b_out']
                W_si  = self.var_dict[input_scope]['W_out_']
                b_si  = self.var_dict[input_scope]['b_out_']

                h = self.state_dict[input_scope]['self'][-1]
                h = self.recurrent_layer(input_scope, x, h)

                mu = h @ W_mu + b_mu
                si = h @ W_si + b_si

                latent = mu + tf.exp(0.5*si)*tf.random_normal(si.shape)

                self.state_dict[input_scope]['self'].append(h)
                self.outputs_dict[input_scope+'_mu']  = mu
                self.outputs_dict[input_scope+'_si'] = si
                self.outputs_dict[input_scope+'_lat'] = latent

                for output_scope in par['output_scopes']:

                    x = latent

                    W_out = self.var_dict[output_scope]['W_out']
                    b_out = self.var_dict[output_scope]['b_out']

                    h = self.state_dict[input_scope][output_scope][-1]
                    h = self.recurrent_layer(output_scope, x, h)

                    out = tf.nn.relu(h @ W_out + b_out)

                    self.state_dict[input_scope][output_scope].append(h)
                    self.outputs_dict[input_scope+'_to_'+output_scope] = out


    def optimize(self):
        print('Maximize')


        quit()





def main(save_fn=None, gpu_id=None):
    """ Run supervised learning training """

    # Isolate requested GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Reset Tensorflow graph before running anything
    tf.reset_default_graph()

    # Define all placeholders
    x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_input']], 'stim')
    y = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_output']], 'out')
    m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')

    # Set up stimulus
    stim = stimulus.MultiStimulus()

    # Start Tensorflow session
    with tf.Session() as sess:

        # Select CPU or GPU
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y, m)

        # Initialize variables and start the timer
        sess.run(tf.global_variables_initializer())
        t_start = time.time()

        for i in range(par['num_train_batches']):

            # Generate a batch of stimulus data for training
            name, stim_in, y_hat, mk, _ = stim.generate_trial(0)

            # Put together the feed dictionary
            feed_dict = {x:stim_in, y:y_hat, m:mk}

            # Run the model
            _, task_loss, output = sess.run([model.train_op, model.task_loss, model.output], feed_dict=feed_dict)
            print(i, '|', task_loss)


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main(gpu_id=sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        quit('\nQuit via KeyboardInterrupt.')
