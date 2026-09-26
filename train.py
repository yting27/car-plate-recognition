import argparse

from ultralytics import YOLO


def print_results(metrics):
    """Print per-class detection metrics in a readable table."""
    names = metrics.names  # {class_id: class_name}

    print("\n===== Validation Results =====")
    print(f"{'Class':<15}{'Images':<10}{'Instances':<12}{'P':<8}{'R':<8}{'mAP50':<8}{'mAP50-95':<10}")
    print("-" * 71)

    # Overall metrics
    print(
        f"{'all':<15}"
        f"{'':<10}"
        f"{'':<12}"
        f"{metrics.box.mp:<8.3f}"
        f"{metrics.box.mr:<8.3f}"
        f"{metrics.box.map50:<8.3f}"
        f"{metrics.box.map:<10.3f}"
    )

    # Per-class metrics
    for i, name in names.items():
        print(
            f"{name:<15}"
            f"{'':<10}"
            f"{'':<12}"
            f"{metrics.box.p[i]:<8.3f}" # Precision
            f"{metrics.box.r[i]:<8.3f}" # Recall
            f"{metrics.box.ap50[i]:<8.3f}"
            f"{metrics.box.ap[i]:<10.3f}"
        )

    print("\n===== Summary =====")
    for key, value in metrics.results_dict.items():
        print(f"{key:<25}{value:.4f}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a YOLO detection model.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Path to the pretrained model weights (default: yolo26n.pt). See https://docs.ultralytics.com/models",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./dataset/data.yaml",
        help="Path to the dataset data.yaml file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load a model
    model = YOLO(args.model)  # load a pretrained model

    # Train the model
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)

    # Print out the results
    print_results(results)


if __name__ == "__main__":
    main()
