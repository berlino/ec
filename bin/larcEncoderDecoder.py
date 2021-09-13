try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from larc.encoderDecoder import main
from larc import decoderUtils
import os

if __name__ == '__main__':
    print("current PID:{}".format(os.getpid()))
    main()

