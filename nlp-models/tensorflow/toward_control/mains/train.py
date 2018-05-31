import tensorflow as tf
import pprint
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

from configs import args
from model import WakeSleepController
from data.imdb import WakeSleepDataLoader
from vocab.imdb import IMDBVocab
from trainers import WakeSleepTrainer
from log import create_logging


def main():
    create_logging()
    sess = tf.Session()
    vocab = IMDBVocab()
    dl = WakeSleepDataLoader(sess, vocab)

    model = WakeSleepController(dl, vocab)
    tf.logging.info('\n'+pprint.pformat(tf.trainable_variables()))
    trainer = WakeSleepTrainer(sess, model, dl, vocab)
    model.load(sess, args.vae_ckpt_dir)
    trainer.train()


if __name__ == '__main__':
    main()