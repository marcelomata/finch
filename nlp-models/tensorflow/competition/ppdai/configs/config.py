import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--w_max_len', type=int, default=20)

parser.add_argument('--n_epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--hidden_units', type=int, default=160)
parser.add_argument('--clip_norm', type=int, default=5.0)


args = parser.parse_args()
