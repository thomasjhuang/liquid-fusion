import logging

from scripts.run_benchmark import parse_args, run_benchmark


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_benchmark(parse_args())
