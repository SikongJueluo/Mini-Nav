import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["train", "benchmark", "visualize"],
        help="Action to perform: train, benchmark, or visualize",
    )
    args = parser.parse_args()

    if args.action == "train":
        from compressors import FloatCompressor, train

        train(FloatCompressor(), 1, 32)
    elif args.action == "benchmark":
        from benchmarks import evaluate

        evaluate("Dinov2", "CIFAR-10", "Recall@10")
    else:  # visualize
        from visualizer import app

        app.run(debug=True)
