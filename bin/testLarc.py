
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from larc.testDecoderUtils import main

if __name__ == '__main__':
    main()

