try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.arc.experiment_output import *
from dreamcoder.domains.arc.codex_synthesis import codex_synthesis_main

if __name__ == '__main__':

    codex_synthesis_main()
    # experiment_output_main("plot_marginals")

