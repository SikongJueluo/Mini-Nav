import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    args = parser.parse_args()

    if args.train:
        from compressors import FloatCompressor, train

        train(FloatCompressor(), 1, 32)
    else:
        from visualizer import app

        app.run(debug=True)
