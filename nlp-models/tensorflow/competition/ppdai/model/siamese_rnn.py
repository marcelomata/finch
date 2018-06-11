from configs import args

import tensorflow as tf
import numpy as np
import time
import pprint


def load_embedding():
    t0 = time.time()
    embedding = np.load('../data/processed_files/embedding.npy')
    print("Load word_embed: %.2fs"%(time.time()-t0))
    return embedding


def cell_fn(sz):
    cell = tf.nn.rnn_cell.GRUCell(sz,
                                  kernel_initializer=tf.orthogonal_initializer())
    return cell


def mask_fn(x):
    return tf.sign(tf.reduce_sum(x, -1))


def rnn(x, cell_fw, cell_bw):
    seq_len = tf.count_nonzero(tf.reduce_sum(x, -1), 1)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                      cell_bw,
                                                      x,
                                                      seq_len,
                                                      dtype=tf.float32)
    outputs = tf.concat(outputs, -1)
    states = tf.concat(states, -1)
    return outputs, states


def embed(x, embedding, is_training):
    x = tf.nn.embedding_lookup(embedding, x)
    x = tf.layers.dropout(x, 0.1, is_training)
    return x


def clip_grads(loss):
    params = tf.trainable_variables()
    grads = tf.gradients(loss, params)
    clipped_grads, _ = tf.clip_by_global_norm(grads, args.clip_norm)
    return zip(clipped_grads, params)


def additive_attn(query, values, v, w, masks):
    query = tf.expand_dims(query, 1)
    align = v * tf.tanh(query + w(values))
    align = tf.reduce_sum(align, [2])

    paddings = tf.fill(tf.shape(align), float('-inf'))
    align = tf.where(tf.equal(masks, 0), paddings, align)

    align = tf.nn.softmax(align)
    align = tf.expand_dims(align, -1)
    val = tf.squeeze(tf.matmul(values, align, transpose_a=True), -1)
    return val


def mul_attn(query, values, w):
    query = tf.expand_dims(query, -1)
    align = tf.matmul(w(values), query)
    align = tf.squeeze(align, -1)
    align = tf.nn.softmax(align)
    align = tf.expand_dims(align, -1)
    val = tf.squeeze(tf.matmul(values, align, transpose_a=True), -1)
    return val


def forward(features, mode):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    x1, x2 = features['input1'], features['input2']
    batch_sz = tf.shape(x1)[0]
    x = tf.concat([x1, x2], axis=0)
    
    embedding = tf.convert_to_tensor(load_embedding())
    x = embed(x, embedding, is_training)
    mask = mask_fn(x)

    cell_fw, cell_bw = cell_fn(args.hidden_units//2), cell_fn(args.hidden_units//2)
    o, s = rnn(x, cell_fw, cell_bw)

    o1, o2 = tf.split(o, 2)
    s1, s2 = tf.split(s, 2)
    mask1, mask2 = tf.split(mask, 2)
    
    v = tf.get_variable('attn_v', [args.hidden_units])
    w = tf.layers.Dense(args.hidden_units)
    attn1 = additive_attn(s1, o2, v, w, mask2)
    attn2 = additive_attn(s2, o1, v, w, mask1)

    x = tf.concat([(s1 - s2),
                   (s1 * s2),
                   (attn1 - attn2),
                   (attn1 * attn2)], -1)
    x = tf.layers.dropout(x, 0.2, training=is_training)
    x = tf.layers.dense(x, args.hidden_units, tf.nn.leaky_relu)
    x = tf.squeeze(tf.layers.dense(x, 1), -1)
    
    return x


def model_fn(features, labels, mode):
    logits = forward(features, mode)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=tf.sigmoid(logits))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info('\n'+pprint.pformat(tf.trainable_variables()))
        global_step = tf.train.get_global_step()

        LR = {'start': 1e-3, 'end': 5e-4, 'steps': 50000}
        
        lr_op = tf.train.exponential_decay(
            LR['start'], global_step, LR['steps'], LR['end']/LR['start'])
        
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.to_float(labels)))

        train_op = tf.train.AdamOptimizer(lr_op).apply_gradients(
            clip_grads(loss_op), global_step=global_step)

        lth = tf.train.LoggingTensorHook({'lr': lr_op}, every_n_iter=100)
        
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, train_op=train_op, training_hooks=[lth])
