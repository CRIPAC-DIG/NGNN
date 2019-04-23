import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import layers


######################model##########################
def weights(name, hidden_size, i):
    image_stdv = np.sqrt(1. / (2048))
    hidden_stdv = np.sqrt(1. / (hidden_size))
    if name == 'in_image':
        w = tf.get_variable(name='w/in_image_'+ str(i),
                            shape=[2048, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))
        #w = tf.get_variable(name='gnn/w/in_image_', shape=[2048, hidden_size], initializer=tf.random_normal_initializer)
    if name == 'out_image':
        w = tf.get_variable(name='w/out_image_' + str(i),
                            shape=[hidden_size, 2048],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))
        #w = tf.get_variable(name='w/out_image_', shape=[hidden_size, 2048], initializer=tf.random_normal_initializer)
    if name == 'hidden_state_out':
        w = tf.get_variable(name='w/hidden_state_out' + str(i),
                            shape=[hidden_size, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))
        #w = tf.get_variable(name='w/hidden_state_out_' + str(i), shape=[hidden_size, hidden_size], initializer=tf.random_normal_initializer)
    if name == 'hidden_state_in':
        #w = tf.get_variable(name='w/hidden_state_in_', shape=[hidden_size, hidden_size], initializer=tf.random_normal_initializer)
        w = tf.get_variable(name='w/hidden_state_in_' + str(i),
                            shape=[hidden_size, hidden_size],
                            initializer=tf.random_normal_initializer(stddev=hidden_stdv))

    return w


def biases(name, hidden_size, i):
    image_stdv = np.sqrt(1. / (2048))
    hidden_stdv = np.sqrt(1. / (hidden_size))
    if name == 'hidden_state_out':
        b = tf.get_variable(name='b/hidden_state_out' + str(i), shape=[hidden_size],
                        initializer=tf.random_normal_initializer(stddev=hidden_stdv))
        # b = tf.get_variable(name='b/hidden_state_out', shape=[hidden_size],
        #                 initializer=tf.random_normal_initializer)
    if name == 'hidden_state_in':
        b = tf.get_variable(name='b/hidden_state_in' + str(i), shape=[hidden_size],
                        initializer=tf.random_normal_initializer(stddev=hidden_stdv))
        # b = tf.get_variable(name='b/hidden_state_in', shape=[hidden_size],
        #                 initializer=tf.random_normal_initializer)
    if name == 'out_image':
        # b = tf.get_variable(name='b/out_image_', shape=[2048],
        #                     initializer=tf.random_normal_initializer)
        b = tf.get_variable(name='b/out_image_' + str(i), shape=[2048],
                            initializer=tf.random_normal_initializer(stddev=image_stdv))

    return b


def message_pass(x, hidden_size, batch_size, num_category, graph):


    w_hidden_state = weights('hidden_state_out', hidden_size, 0)
    #b_hidden_state = biases('hidden_state_out', hidden_size, 0)
    x_all = tf.reshape(tf.matmul(
        tf.reshape(x[:,0,:], [batch_size, hidden_size]),
        w_hidden_state),
                       [batch_size, hidden_size])
    for i in range(1, num_category):
        w_hidden_state = weights('hidden_state_out', hidden_size, i)
        #b_hidden_state = biases('hidden_state_out', hidden_size, i)
        x_all_ = tf.reshape(tf.matmul(
            tf.reshape(x[:, i, :], [batch_size, hidden_size]),
            w_hidden_state),
                           [batch_size, hidden_size])
        x_all = tf.concat([x_all, x_all_], 1)
    x_all = tf.reshape(x_all, [batch_size, num_category, hidden_size])
    x_all = tf.transpose(x_all, (0, 2, 1))  # [batch_size, hidden_size, num_category]

    x_ = x_all[0]
    graph_ = graph[0]
    x = tf.matmul(x_, graph_)
    for i in range(1, batch_size):
        x_ = x_all[i]
        graph_ = graph[i]
        x_ = tf.matmul(x_, graph_)
        x = tf.concat([x, x_], 0)
    x = tf.reshape(x, [batch_size, hidden_size, num_category])
    x = tf.transpose(x, (0, 2, 1))

    x_ = tf.reshape(tf.matmul(x[:, 0, :], weights('hidden_state_in', hidden_size, 0)),
                    [batch_size, hidden_size])
    for j in range(1, num_category):
        _x = tf.reshape(tf.matmul(x[:, j, :], weights('hidden_state_in', hidden_size, j)),
                        [batch_size, hidden_size])
        x_ = tf.concat([x_, _x], 1)
    x = tf.reshape(x_, [batch_size, num_category, hidden_size])

    return x


#def GNN(image, batch_size, hidden_size, keep_prob, n_steps, mask_x, num_category, graph):
def GNN(image, batch_size, hidden_size, n_steps, num_category, graph):

    gru_cell = GRUCell(hidden_size)
    w_in_image = weights('in_image', hidden_size, 0)
    h0 = tf.reshape(tf.matmul(image[:,0,:], w_in_image), [batch_size, hidden_size])  #initialize h0 [batchsize, hidden_state]
    for i in range(1, num_category):
        w_in_image = weights('in_image', hidden_size, i)
        h0 = tf.concat([h0, tf.reshape(
                tf.matmul(image[:,i,:], w_in_image), [batch_size, hidden_size])
                          ], 1)
    h0 = tf.reshape(h0, [batch_size, num_category, hidden_size])  # h0: [batchsize, num_category, hidden_state]
    # print (h0)
    h0 = tf.nn.tanh(h0)
    state = h0
    sum_graph = tf.reduce_sum(graph, reduction_indices=1)
    enable_node = tf.cast(tf.cast(sum_graph, dtype=bool), dtype=tf.float32)

    with tf.variable_scope("gnn"):
        for step in range(n_steps):
            if step > 0: tf.get_variable_scope().reuse_variables()
            #state = state * mask_x
            x = message_pass(state, hidden_size, batch_size, num_category, graph)
            # x = tf.reshape(x, [batch_size*num_category, hidden_size])
            # state = tf.reshape(state, [batch_size*num_category, hidden_size])
            (x_new, state_new) = gru_cell(x[0], state[0])
            state_new = tf.transpose(state_new, (1,0))
            state_new = tf.multiply(state_new, enable_node[0])
            state_new = tf.transpose(state_new, (1,0))
            for i in range(1, batch_size):
                (x_, state_) = gru_cell(x[i], state[i])  ##input of GRUCell must be 2 rank, not 3 rank
                state_ = tf.transpose(state_, (1,0))
                state_ = tf.multiply(state_, enable_node[i])
                state_ = tf.transpose(state_, (1,0))
                state_new = tf.concat([state_new, state_], 0)
            #x = tf.reshape(x, [batch_size, num_category, hidden_size])
            state = tf.reshape(state_new, [batch_size, num_category, hidden_size])  ##restore: 2 rank to 3 rank
            #state = state * mask_x
            #state = tf.nn.dropout(state, keep_prob)

    # w_out_image = weights('out_image', hidden_size, 0)
    # b_out_image = biases('out_image', hidden_size, 0)
    # output = tf.reshape(tf.matmul(state[:, 0, :], w_out_image) + b_out_image, [batch_size, 2048]) #initialize output : [batchsize, 2048]
    # for i in range(1, num_category):
    #     w_out_image = weights('out_image', hidden_size, i)
    #     b_out_image = biases('out_image', hidden_size, i)
    #     output = tf.concat([output, tf.reshape(
    #         tf.matmul(state[:, i, :], w_out_image) + b_out_image,
    #                        [batch_size, 2048])], 1)
    # output = tf.reshape(output, [batch_size, num_category, 2048])
    # output = tf.nn.tanh(output)

    return state, h0

