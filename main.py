"""Download/partition CIFAR, then run asynchronous FLEX-FL training.

Run from a CUDA-enabled environment with the project dependencies installed.
The dataset preparation script uses Unix shell utilities (Linux/WSL).
"""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--data_root', type=Path, default=None)
    parser.add_argument('--work_dir', type=Path, default=Path('runs'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--local_iters', type=int, default=1)
    parser.add_argument('--total_iters', type=int, default=100)
    parser.add_argument('--wfo', type=float, default=0.4,
                        help='Adaptive coefficient interpolation toward 1 (default: 0.4)')
    args = parser.parse_args()

    if args.num_clients < 2:
        parser.error('--num_clients must be at least 2')
    for name in ('batch_size', 'local_iters', 'total_iters'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    if not 0 <= args.wfo <= 1:
        parser.error('--wfo must be between 0 and 1')

    project_dir = Path(__file__).resolve().parent
    # Resolve user paths before launching children from the project directory.
    data_root = (args.data_root or Path('data') / f'{args.dataset}_{args.num_clients}').resolve()
    work_dir = args.work_dir.resolve()

    prepare_command = [
        sys.executable, str(project_dir / 'create_data_splits.py'),
        '--dataset', args.dataset,
        '--root', str(data_root),
        '--num_parts', str(args.num_clients),
        '--batch_size', str(args.batch_size),
        '--overlap', '0',
        '--iid', 'True',
    ]
    train_command = [
        sys.executable, str(project_dir / 'run_p2p_async_fl.py'),
        '--dataset_name', args.dataset,
        '--num_classes', '10' if args.dataset == 'cifar10' else '100',
        '--arch_name', 'resnet18',
        '--data_root', str(data_root),
        '--work_dir', str(work_dir),
        '--num_clients', str(args.num_clients),
        '--gpu', str(args.gpu),
        '--batch_size', str(args.batch_size),
        '--local_iters', str(args.local_iters),
        '--total_iters', str(args.total_iters),
        '--train_val', 'Classic',
        '--fusion_protocol_name', 'peer_grad_ada_wfo',
        '--wfo', str(args.wfo),
        '--adm_prob', '0.4',
        '--n_peers', '1',
        '--latency_mu', '0',
        '--stde_mode', '0',
        # No --sync flag: run asynchronously.
    ]

    print('Step 1/2: Downloading and preparing IID dataset partitions...', flush=True)
    subprocess.run(prepare_command, cwd=project_dir, check=True)
    print('Step 2/2: Starting asynchronous training with peer_grad_ada_wfo...', flush=True)
    subprocess.run(train_command, cwd=project_dir, check=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
