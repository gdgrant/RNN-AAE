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
                    if not ('W_rnn' in name and scope not in par['rnn_scopes']):
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

        if var_scope in par['rnn_scopes']:

            W_in  = self.var_dict[var_scope]['W_in']
            W_rnn = self.var_dict[var_scope]['W_rnn']
            b_rnn = self.var_dict[var_scope]['b_rnn']

            W_rnn_eff = par['EI_matrix'][var_scope] @ tf.nn.relu(W_rnn) if par['EI'] else W_rnn
            h_post = h

            h = tf.nn.relu((1-par['alpha_neuron'])*h \
             + par['alpha_neuron']*(x @ W_in + h_post @ W_rnn_eff + b_rnn) \
             + tf.random_normal(h.shape, 0., par['noise_rnn'], dtype=tf.float32))

            return h

        else:

            W_in  = self.var_dict[var_scope]['W_in']
            b_rnn = self.var_dict[var_scope]['b_rnn']

            h = tf.nn.relu(x @ W_in + b_rnn + tf.random_normal(h.shape, 0., par['noise_rnn'], dtype=tf.float32))

            return h


    def run_model(self):

        self.outputs_dict = {}
        for input_scope in par['input_scopes']:
            self.outputs_dict[input_scope+'_mu'] = []
            self.outputs_dict[input_scope+'_si'] = []
            self.outputs_dict[input_scope+'_lat'] = []
            for output_scope in par['output_scopes']:
                self.outputs_dict[input_scope+'_to_'+output_scope] = []

        for t in range(par['num_time_steps']):

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
                self.outputs_dict[input_scope+'_mu'].append(mu)
                self.outputs_dict[input_scope+'_si'].append(si)
                self.outputs_dict[input_scope+'_lat'].append(latent)

                for output_scope in par['output_scopes']:

                    x = latent

                    W_out = self.var_dict[output_scope]['W_out']
                    b_out = self.var_dict[output_scope]['b_out']

                    h = self.state_dict[input_scope][output_scope][-1]
                    h = self.recurrent_layer(output_scope, x, h)

                    out = tf.nn.relu(h @ W_out + b_out)

                    self.state_dict[input_scope][output_scope].append(h)
                    self.outputs_dict[input_scope+'_to_'+output_scope].append(out)


    def optimize(self):

        opt = AdamOpt.AdamOpt(tf.trainable_variables(), par['learning_rate'])
        eps = 1e-7

        # Putting together variable groups
        encoder  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        decoder  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
        VAE_vars = encoder + decoder

        generator     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        discriminator = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        GAN_vars      = generator + discriminator

        task_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='solution')

        # Task loss and training
        task_loss_list = [mask*tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=target+eps) \
            for out, target, mask in zip(self.outputs_dict['encoder_to_solution'], self.target_data, self.time_mask)]
        self.task_loss = tf.reduce_mean(tf.stack(task_loss_list))

        y_prob = [tf.nn.softmax(out) for out in self.outputs_dict['generator_to_solution']]
        self.entropy_loss = tf.reduce_mean(tf.stack([-m*tf.reduce_mean(-p_i * tf.log(p_i)) for p_i, m in zip(y_prob, self.time_mask)]))

        y_prob = [tf.nn.softmax(out) for out in self.outputs_dict['encoder_to_solution']]
        self.entropy_loss_enc = tf.reduce_mean(tf.stack([-m*tf.reduce_mean(-p_i * tf.log(p_i)) for p_i, m in zip(y_prob, self.time_mask)]))

        self.train_task         = opt.compute_gradients(self.task_loss, var_list=task_vars)
        self.train_task_entropy = opt.compute_gradients(self.entropy_loss, var_list=task_vars)


        # Autoencoder loss and training
        recon_loss_list = [tf.square(out-target) for out, target in \
            zip(self.outputs_dict['encoder_to_decoder'], self.input_data)]
        self.recon_loss = tf.reduce_mean(tf.stack(recon_loss_list))

        si = self.outputs_dict['encoder_si']
        mu = self.outputs_dict['encoder_mu']
        latent_loss_list = [par['act_latent_cost'] * -0.5 * tf.reduce_sum(1+si_t-tf.square(mu_t)-tf.exp(si_t), axis=-1) \
            for mu_t, si_t in zip(mu, si)]
        self.act_latent_loss = tf.reduce_mean(tf.stack(latent_loss_list))

        self.train_VAE = opt.compute_gradients(self.recon_loss + self.act_latent_loss, var_list=VAE_vars)


        # Discriminator loss and training
        self.discr_gen_loss = tf.reduce_mean(tf.square(tf.stack(self.outputs_dict['generator_to_discriminator'], axis=0) - par['discriminator_gen_target']))
        self.discr_act_loss = tf.reduce_mean(tf.square(tf.stack(self.outputs_dict['encoder_to_discriminator'], axis=0) - par['discriminator_act_target']))

        self.gener_gen_loss = tf.reduce_mean(tf.square(tf.stack(self.outputs_dict['generator_to_discriminator'], axis=0) - par['discriminator_act_target']))
        self.gener_act_loss = tf.reduce_mean(tf.square(tf.stack(self.outputs_dict['encoder_to_discriminator'], axis=0) - par['discriminator_gen_target']))

        si = self.outputs_dict['generator_si']
        mu = self.outputs_dict['generator_mu']
        latent_loss_list = [par['gen_latent_cost'] * -0.5 * tf.reduce_sum(1+si_t-tf.square(mu_t)-tf.exp(si_t), axis=-1) \
            for mu_t, si_t in zip(mu, si)]
        self.gen_latent_loss = tf.reduce_mean(tf.stack(latent_loss_list))

        self.generator_loss = self.gener_gen_loss + self.gener_act_loss + self.gen_latent_loss
        self.discriminator_loss = self.discr_gen_loss + self.discr_act_loss

        self.train_generator     = opt.compute_gradients(self.generator_loss, var_list=generator)
        self.train_discriminator = opt.compute_gradients(self.discriminator_loss, var_list=discriminator)


        self.reset_adam_op = opt.reset_params()


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
    m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], 1], 'mask')

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

        print('\nStarting training.\n')

        # Training autoencoder
        for i in range(par['num_autoencoder_batches']):

            # Generate a batch of stimulus data for training
            # and put together the model's feed dictionary
            name, stim_in, y_hat, mk, _ = stim.generate_trial(0)
            feed_dict = {x:stim_in, y:y_hat, m:mk[...,np.newaxis]}

            # Run the model
            _, recon_loss, latent_loss = sess.run([model.train_VAE, model.recon_loss, model.act_latent_loss], feed_dict=feed_dict)

            if i%200 == 0:
                print('{:4} | Recon: {:5.3f} | Lat: {:5.3f}'.format(i, recon_loss, latent_loss))


        sess.run(model.reset_adam_op)


        # Training generative adversarial network
        for i in range(par['num_GAN_batches']):

            for j in range(3):
                if j == 0:
                    trainer = model.train_generator
                    curr = 'G'
                else:
                    trainer = model.train_discriminator
                    curr = 'D'

                # Generate a batch of stimulus data for training
                # and put together the model's feed dictionary
                name, stim_in, y_hat, mk, _ = stim.generate_trial(0)
                feed_dict = {x:stim_in, y:y_hat, m:mk[...,np.newaxis]}

                # Run the model
                _, gen_loss, discr_loss, gen_latent = sess.run([trainer, model.generator_loss, \
                    model.discriminator_loss, model.gen_latent_loss], feed_dict=feed_dict)

                if i%200 == 0 and j in [0,2]:
                    print('{:4} | {} | Gen: {:6.3f} | Discr: {:6.3f} | Lat: {:5.3f}'.format(i, curr, gen_loss, discr_loss, gen_latent))




if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main(gpu_id=sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        quit('\nQuit via KeyboardInterrupt.')
