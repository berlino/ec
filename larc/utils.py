import argparse

def commandLineArgs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--restrict-types",
                        dest="restrict_types",
                        default=False,
                        action="store_true")
    parser.add_argument("--use-cuda",
                        dest="use_cuda",
                        default=False,
                        action="store_true")
    parser.add_argument("--verbose",
                        dest="verbose",
                        default=False,
                        action="store_true")
    parser.add_argument("--rnn-decode",
                        dest="rnn_decode",
                        default=False,
                        action="store_true")
    parser.add_argument("--batch-size",
                        dest="batch_size",
                        default=32,
                        type=int)
    parser.add_argument("--lr",
                        dest="lr",
                        default=0.001,
                        type=float)
    parser.add_argument("--weight-decay",
                        dest="weight_decay",
                        default=0.0,
                        type=float)
    parser.add_argument("--beta",
                        dest="beta",
                        default=0.0,
                        type=float)
    parser.add_argument("--epochs-per-replay",
                        dest="epochs_per_replay",
                        default=0,
                        type=int)
    parser.add_argument("--beam-width",
                        dest="beam_width",
                        default=128,
                        type=int)
    parser.add_argument("--epsilon",
                        dest="epsilon",
                        default=0.3,
                        type=float)
    parser.add_argument("--num-cpus",
                        dest="num_cpus",
                        default=1,
                        type=int)
    parser.add_argument("--num-cycles",
                        dest="num_cycles",
                        default=1,
                        type=int)
    parser.add_argument("--max-p-per-task",
                        dest="max_p_per_task",
                        # something >> number of distinct programs per tasks
                        default=1000000,
                        type=int)
    parser.add_argument("--seed",
                        dest="seed",
                        default=0,
                        type=int)
    parser.add_argument("--jumpstart",
                        dest="jumpstart",
                        default=False,
                        help="Whether to jumpstart by training on set of ground truth programs first",
                        action="store_true")

    args = vars(parser.parse_args())
    return args
