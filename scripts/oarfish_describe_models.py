import sys
import argparse

from oarfish.models import *
from oarfish.train import print_checkpoint_analysis


DEFAULT_BINARY = get_default_binary_model()
DEFAULT_MULTI = get_default_multi_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='print training and validation details about oarfish  models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--binary-model', type=str, default=DEFAULT_BINARY,
                        help='binary model to use for prediction')
    parser.add_argument('--multi-model', type=str, default=DEFAULT_MULTI,
                        help='multi-class model to use for prediction')
    args = parser.parse_args()
   
    for mdl in (args.binary_model, args.multi_model): 
        print_checkpoint_analysis(mdl)

