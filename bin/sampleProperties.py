from dreamcoder.domains.list.runUtils import list_options
from dreamcoder.domains.list.sampleProperties import prop_sampling_options, main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    list_options(parser)
    prop_sampling_options(parser)

    args = vars(parser.parse_args())
    main(args)
