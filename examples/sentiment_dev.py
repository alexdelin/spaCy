#!/usr/bin/env python
"""
This is the main endpoint that is used to train and evaluate
Sentiment analyzers created with spaCy
"""
import argparse
import ConfigParser
import logging
import os
import sys
import time

from deep_learning_keras import evaluate_test_content, create_new_training, continue_training

# Script version. It's recommended to increment this with every change, to make
# debugging easier.
VERSION = '0.9.0'


# Set up logging.
log = logging.getLogger('{0}[{1}]'.format(os.path.basename(sys.argv[0]),
                                          os.getpid()))


def run():
    """Main entry point run by __main__ below. No need to change this usually.
    """
    args = parse_args()
    setup_logging(args)
    config = get_config(args)

    log.info('Starting process (version %s).', VERSION)
    log.debug('Arguments: %r', args)

    # run the application
    try:
        main(args, config)
    except Exception:
        log.exception('Processing error')


def main(args, config):
    """
    The main method. Any exceptions are caught outside of this method and will
    be handled.
    """
    if args.evaluate_flag:

        model_dir = config.get('model', 'model_dir')
        dev_dir = config.get('model', 'dev_dir')
        max_length = config.getint('model', 'max_length')

        log.warn('Evaluating Test Data in folder {}'.format(dev_dir))
        evaluate_test_content(model_dir, dev_dir, max_length)

    elif args.train_flag:

        if args.continue_flag:
            model_dir = config.get('model', 'model_dir')
            dev_dir = config.get('model', 'dev_dir')
            train_dir = config.get('model', 'train_dir')
            max_length = config.getint('model', 'max_length')
            nr_hidden = config.getint('model', 'nr_hidden')
            dropout = config.getfloat('model', 'dropout')
            learn_rate = config.getfloat('model', 'learn_rate')
            nb_epoch = config.getint('model', 'epochs')
            batch_size = config.getint('model', 'batch_size')
            nr_examples = config.getint('model', 'nr_examples')

            log.warn('Continuing Training of an existing model')
            continue_training(model_dir=model_dir, dev_dir=dev_dir, train_dir=train_dir,
                nr_hidden=nr_hidden, max_length=max_length,
                dropout=dropout, learn_rate=learn_rate,
                nb_epoch=nb_epoch, batch_size=batch_size, nr_examples=nr_examples)

        else:
            model_dir = config.get('model', 'model_dir')
            dev_dir = config.get('model', 'dev_dir')
            train_dir = config.get('model', 'train_dir')
            max_length = config.getint('model', 'max_length')
            nr_hidden = config.getint('model', 'nr_hidden')
            dropout = config.getfloat('model', 'dropout')
            learn_rate = config.getfloat('model', 'learn_rate')
            nb_epoch = config.getint('model', 'epochs')
            batch_size = config.getint('model', 'batch_size')
            nr_examples = config.getint('model', 'nr_examples')

            log.warn('Training a new model')
            create_new_training(model_dir=model_dir, dev_dir=dev_dir, train_dir=train_dir,
                nr_hidden=nr_hidden, max_length=max_length,
                dropout=dropout, learn_rate=learn_rate,
                nb_epoch=nb_epoch, batch_size=batch_size, nr_examples=nr_examples)

    else:
        raise ValueError('Please Select Either Train or Evaluate')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version=VERSION)
    parser.add_argument('--verbose', '-v', action='count',
                        help='Show additional information.')
    parser.add_argument('--log-file', dest='log_file',
                        help='Log file on disk.')
    parser.add_argument('--config-file', dest='config_file',
                        help='Configuration file to read settings from.')
    parser.add_argument('--evaluate', dest='evaluate_flag', action="store_true",
                        help='Only Evaluate the test content and exit')
    parser.add_argument('--train', dest='train_flag', action="store_true",
                        help='Train The model using the data in the train_dir folder')
    parser.add_argument('--continue', dest='continue_flag', action="store_true",
                        help='Continue Training an existing model')

    return parser.parse_args()


def setup_logging(args):
    """Set up logging based on the command line options.
    """
    # Set up logging
    fmt = '%(asctime)s %(name)s %(levelname)-8s %(message)s'
    if args.verbose == 1:
        level = logging.INFO
        logging.getLogger(
            'requests.packages.urllib3.connectionpool').setLevel(logging.WARN)
    elif args.verbose >= 2:
        level = logging.DEBUG
    else:
        # default value
        level = logging.WARN
        logging.getLogger(
            'requests.packages.urllib3.connectionpool').setLevel(logging.WARN)

    # configure the logging system
    if args.log_file:
        out_dir = os.path.dirname(args.log_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logging.basicConfig(
            filename=args.log_file, filemode='a', level=level, format=fmt)
    else:
        logging.basicConfig(level=level, format=fmt)

    # Log time in UTC
    logging.Formatter.converter = time.gmtime


def get_config(args):
    """Parse the config file and return a ConfigParser object.

    Always reads the `main.ini` file in the current directory (`main` is
    replaced by the current basename of the script).
    """
    cfg = ConfigParser.SafeConfigParser()

    root, _ = os.path.splitext(__file__)
    files = [root + '.ini']
    if args.config_file:
        files.append(args.config_file)

    log.debug('Reading config files: %r', files)
    cfg.read(files)
    return cfg


# This is run if this script is executed, rather than imported.
if __name__ == '__main__':
    run()
