import argparse
import os
import sys

MOBILENETV3_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(MOBILENETV3_ROOT)
NAS_BENCH_201_ROOT = os.path.join(REPO_ROOT, "NAS_Bench_201")
for path in (REPO_ROOT, NAS_BENCH_201_ROOT, MOBILENETV3_ROOT):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from config.config import mobile_retrain_gpu


def _set_single_gpu(gpu):
    gpu = str(gpu).strip()
    if "," in gpu:
        raise ValueError(f"MobileNetV3 retraining expects one GPU id, got: {gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["DDE_NAG_GPU"] = gpu
    os.environ["EDNAG_GPU"] = gpu
    print(f"==> MobileNetV3 uses a single visible GPU: physical GPU {gpu}")


def main(dataset, gpu):
    _set_single_gpu(gpu)

    from experiments import exp_with_fixed_seed_in_meta_predictor

    dataset_name = {'cifar10': 'cifar10', 'cifar100': 'cifar100', 'aircraft': 'aircraft', 'pets': 'pets'}
    dataset = dataset.lower()
    assert dataset in dataset_name, f'ERROR: invalid dataset {dataset}'

    exp_with_fixed_seed_in_meta_predictor(dataset=dataset_name[dataset])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EvoDiff-NAS')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, aircraft, pets')
    parser.add_argument('--gpu', type=str, default=mobile_retrain_gpu, help='single physical GPU id, e.g. 0 or 1')
    args = parser.parse_args()
    main(args.dataset, args.gpu)
