import tensorflow as tf


def gumbel_softmax_sample(logits, temperature):
    dist = tf.contrib.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
    return dist.sample()
