### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
from itertools import product, chain

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par
par = {
    # Setup parameters
    'save_dir'                  : './savedir/',
    'include_rule_signal'       : False,
    'training_method'           : 'supervised',

    # Network configuration
    'exc_inh_prop'              : 0.8,

    # Network shape
    'num_motion_tuned'          : 64,
    'num_fix_tuned'             : 4,
    'num_rule_tuned'            : 0,

    'n_latent'                  : 12,
    'n_discriminator'           : 2,
    'n_generator'               : 100,

    'input_scopes'              : ['encoder', 'generator'],
    'output_scopes'             : ['decoder', 'discriminator', 'solution'],
    'rnn_scopes'                : ['generator', 'solution'],
    'solution_hidden'           : 256,
    'discriminator_hidden'      : 128,
    'encoder_hidden'            : 200,
    'generator_hidden'          : 200,
    'decoder_hidden'            : 200,

    # Timings and rates
    'dt'                        : 100,
    'learning_rate'             : 1e-3,
    'membrane_time_constant'    : 100,
    'connection_prob'           : 1.0,

    # Variance values
    'clip_max_grad_val'         : 1.0,
    'rnn_noise_sd'              : 0.05,
    'stim_noise_sd'             : 0.0,
    'stim_noise_mean'           : 0.0,

    # Task specs
    'task'                      : 'multistim',
    'multistim_trial_length'    : 2000,
    'mask_duration'             : 0,
    'dead_time'                 : 200,

    # Tuning function data
    'num_motion_dirs'           : 8,
    'tuning_height'             : 4.0,

    # Cost values
    'spike_cost'                : 1e-7,
    'act_latent_cost'           : 2e-4,
    'gen_latent_cost'           : 2e-4,

    # Training specs
    'batch_size'                : 512,
    'num_autoencoder_batches'   : 2001,
    'num_GAN_batches'           : 8001,
    'num_train_batches'         : 8001,
    'num_entropy_batches'       : 2001,
    'num_final_test_batches'    : 10,
}


############################
### Dependent parameters ###
############################


def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for (key, val) in updates.items():
        par[key] = val
        print('Updating : ', key, ' -> ', val)
    update_dependencies()


def update_dependencies():

    par['n_input']  = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
    par['n_output'] = par['num_motion_dirs'] + 1

    par['dt_sec']           = par['dt']/1000
    par['num_time_steps']   = par['multistim_trial_length']//par['dt']
    par['alpha_neuron']     = np.float32(par['dt']/par['membrane_time_constant'])

    par['noise_rnn'] = np.float32(np.sqrt(2*par['alpha_neuron'])*par['rnn_noise_sd'])
    par['noise_stim'] = np.float32(np.sqrt(2/par['alpha_neuron'])*par['stim_noise_sd'])

    if par['exc_inh_prop'] == 1.:
        par['EI'] = False
    else:
        par['EI'] = True

    par['scopes'] = par['input_scopes'] + par['output_scopes']

    par['var_inits'] = {}
    par['rnn_masks'] = {}
    par['EI_matrix']  = {}
    for scope in par['scopes']:
        scope_vars = {}

        if scope == 'encoder':
            input_size  = par['n_input']
            hidden_size = par['encoder_hidden']
            output_size = par['n_latent']
        elif scope == 'decoder':
            input_size  = par['n_latent']
            hidden_size = par['decoder_hidden']
            output_size = par['n_input']
        elif scope == 'generator':
            input_size  = par['n_generator']
            hidden_size = par['generator_hidden']
            output_size = par['n_latent']
        elif scope == 'discriminator':
            input_size  = par['n_latent']
            hidden_size = par['discriminator_hidden']
            output_size = par['n_discriminator']
        elif scope == 'solution':
            input_size  = par['n_latent']
            hidden_size = par['solution_hidden']
            output_size = par['n_output']

        scope_vars['W_in']  = initialize_weight([input_size, hidden_size])
        scope_vars['b_rnn'] = initialize_weight([1, hidden_size])
        scope_vars['W_out'] = initialize_weight([hidden_size, output_size])
        scope_vars['b_out'] = initialize_weight([1, output_size])
        if scope in par['input_scopes']:
            scope_vars['W_out_'] = initialize_weight([hidden_size, output_size])
            scope_vars['b_out_'] = initialize_weight([1, output_size])


        rnn_mask = np.ones([hidden_size, hidden_size], dtype=np.float32)
        rnn_mask -= np.eye(hidden_size) if par['EI'] else 0.
        scope_vars['W_rnn'] = rnn_mask*initialize_weight([hidden_size, hidden_size])

        EI_list = np.ones(hidden_size)
        if par['EI']:
            EI_list[int(np.round(hidden_size*par['exc_inh_prop'])):] = -1.
        EI_matrix = np.float32(np.diag(EI_list))

        par['var_inits'][scope] = scope_vars
        par['rnn_masks'][scope] = rnn_mask
        par['EI_matrix'][scope] = EI_matrix


    par['hid_inits'] = {}
    for input_scope in par['input_scopes']:
        scope_dict = {}
        for output_scope in ['self'] + par['output_scopes']:

            if input_scope == 'encoder' and output_scope == 'self':
                hidden_size = par['encoder_hidden']
            elif input_scope == 'generator' and output_scope == 'self':
                hidden_size = par['generator_hidden']
            elif output_scope == 'decoder':
                hidden_size = par['decoder_hidden']
            elif output_scope == 'discriminator':
                hidden_size = par['discriminator_hidden']
            elif output_scope == 'solution':
                hidden_size = par['solution_hidden']
            scope_dict[output_scope] = 0.1*np.ones((par['batch_size'], hidden_size), dtype=np.float32)
        par['hid_inits'][input_scope] = scope_dict


    par['discriminator_gen_target'] = np.zeros([par['num_time_steps'],par['batch_size'], par['n_discriminator']])
    par['discriminator_gen_target'][:,:,:par['n_discriminator']//2] = 1

    par['discriminator_act_target'] = np.zeros([par['num_time_steps'],par['batch_size'], par['n_discriminator']])
    par['discriminator_act_target'][:,:,par['n_discriminator']//2:] = 1


def initialize_weight(dims):
    w = 0.05*np.random.gamma(shape=0.25, scale=1.0, size=dims)
    w *= (np.random.rand(*dims) < par['connection_prob'])
    return np.float32(w)


update_dependencies()
print("--> Parameters successfully loaded.\n")
