"""Canonical hierarchical context pipeline. See --help; works from any cwd."""
import argparse
import json
from contextual.config import load_config, location


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON configuration, relative to the repository or absolute")
    parser.add_argument("--download", "--download-only", action="store_true", help="Fetch real parent-child conversation data")
    parser.add_argument("--prepare", action="store_true", help="Validate data and create leakage-safe splits")
    parser.add_argument("--train", action="store_true", help="Compute embeddings and train the hierarchical head")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate a saved checkpoint")
    parser.add_argument("--all", action="store_true", help="Download, prepare, train, and evaluate")
    parser.add_argument("--predict", help="Target text to score using the trained checkpoint")
    parser.add_argument("--parent", action="append", help="Ancestor text, repeat oldest to newest")
    parser.add_argument("--embeddings", action="store_true", help="Include message and conversation embeddings")
    args = parser.parse_args(argv)
    if not any((args.download, args.prepare, args.train, args.evaluate, args.all, args.predict is not None)):
        parser.print_help()
        return
    cfg = load_config(args.config)
    if args.download or args.all:
        from contextual.data import download
        print(download(cfg))
    if args.prepare or args.all:
        from contextual.data import prepare
        print(json.dumps(prepare(cfg), indent=2))
    if args.train or args.all:
        from contextual.training import train
        print(json.dumps(train(cfg), indent=2))
    elif args.evaluate:
        from contextual.training import evaluate
        print(json.dumps(evaluate(cfg), indent=2))
    if args.predict is not None:
        from contextual.inference import Predictor
        print(json.dumps(Predictor(location(cfg, "checkpoint")).predict(args.predict, args.parent,
                         args.embeddings), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
