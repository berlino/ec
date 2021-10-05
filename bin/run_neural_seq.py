try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from neural_seq.model import main
from neural_seq.utils import commandLineArgs
from neural_seq import decoderUtils
import os

if __name__ == '__main__':
    print("current PID:{}".format(os.getpid()))
    args = commandLineArgs()
    main(args)