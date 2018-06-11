from sklearn.metrics import log_loss

import tensorflow as tf
import numpy as np
import pandas as pd
import pprint

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

from log import create_logging
from configs import args
from data import DataLoader, PreProcessor
from model.siamese_rnn import model_fn


def get_val_labels():
    ori_train_csv = pd.read_csv('../data/original_files/train.csv')
    thres = int(len(ori_train_csv) * 0.9)
    val_csv = ori_train_csv[thres:]
    return val_csv['label'].values


def main():
    create_logging()

    dl = DataLoader()

    estimator = tf.estimator.Estimator(model_fn)
    
    y_true = get_val_labels()
    for _ in range(args.n_epochs):
        estimator.train(lambda: dl.train_input_fn())
        y_pred = list(estimator.predict(lambda: dl.val_input_fn()))
        tf.logging.info('\nVal Log Loss: %.3f\n' % log_loss(y_true, y_pred, eps=1e-15))
    submit_arr = np.asarray(list(estimator.predict(lambda: dl.predict_input_fn())))
    print(submit_arr.shape)
    
    submit = pd.DataFrame()
    submit['y_pre'] = submit_arr
    submit.to_csv('./submit_siamese_rnn.csv',index=False)


if __name__ == '__main__':
    main()