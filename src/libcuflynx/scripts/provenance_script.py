"""Write a calibration workflow run's provenance graph and diagram into its output directory."""

import argparse
import sys

from libcuflynx.reporting.provenance import write_provenance


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Write provenance_graph.json and provenance.svg, how every value of a '
                    'calibration workflow run was set, into its output directory. Does not '
                    'read user_inputs.yaml: everything comes from the output directory.')
    parser.add_argument('output_dir', help="the workflow run's output directory")
    args = parser.parse_args(argv)
    for path in write_provenance(args.output_dir):
        print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
