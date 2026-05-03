import config.config as cfg
from experiments import run_experiment


def main():
    result = run_experiment(cfg, show_index=cfg.TEST_INDEX, compute_auc=True)
    print(f"dataset/category: {result['dataset']}/{result['category']}")
    print(f"device: {result['device']}")
    print(f"train/test size: {result['train_size']}/{result['test_size']}")
    print(f"test classes: {result['test_classes']}")
    print(f"image AUC: {result['image_auc']:.4f}")


if __name__ == "__main__":
    main()
