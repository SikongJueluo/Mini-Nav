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
        from compressors import train

        # 启动训练
        train(
            epoch_size=10, batch_size=64, lr=1e-4, checkpoint_path="hash_checkpoint.pt"
        )
    elif args.action == "benchmark":
        from benchmarks import evaluate

        evaluate("Dinov2", "CIFAR-10", "Recall@10")
    else:  # visualize
        from visualizer import app

        app.run(debug=True)
