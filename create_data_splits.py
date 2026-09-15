import os 

import argparse

def parse_bool(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Splitter")
    
    parser.add_argument('--num_parts', type=int, default=2, help='Number of parts to split the dataset into')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap ratio between splits')
    parser.add_argument('--iid', type=parse_bool, nargs='?', const=True, default=False, help='Whether the splits are IID (true/false)')
    parser.add_argument('--root', type=str, default='./data', help='Root directory to save the splits')
    parser.add_argument('--gpu', type=int, default=0, help='GPU  ID')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for local training')
    parser.add_argument('--dataset', choices=['cifar10','cifar100'], default='cifar10', help='Dataset to use')

     
    args = parser.parse_args()
    if args.num_parts <= 0 or args.batch_size <= 0:
        parser.error('--num_parts and --batch_size must be positive')
    if not 0 <= args.overlap < 1:
        parser.error('--overlap must be between 0 (inclusive) and 1 (exclusive)')
    return args

def main():
    from dataset import Dataset

    args = parse_args()


    dataset=args.dataset
    num_parts=args.num_parts
    overlap=args.overlap
    iid=args.iid
    print(iid)

    root=args.root
    gpu = args.gpu
    batch_size = args.batch_size

    dataset = Dataset(dataset,root).get_dataset()

    dataset.create_splits(num_parts,batch_size,overlap,iid)

    os.system(f'mkdir {root}/part_{num_parts+1}')
    os.system(f'cp -r {root}/central/train {root}/part_{num_parts+1}/')

    for i in range(num_parts+1):
        os.system(f'cp -r {root}/central/test {root}/part_{i+1}/')
        os.system(f'cp -r {root}/central/val {root}/part_{i+1}/')


if __name__ == '__main__':
    main()
