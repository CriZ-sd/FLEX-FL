# FLEX-FL

This repository contains the code for the following paper:

Sad, C., Masouros, D., Retsinas, G., Soudris, D., & Siozios, K. (2026). **FLEX: Flexible federated learning with asynchronous peer-to-peer communication and adaptive aggregation.** *IEEE Transactions on Parallel and Distributed Systems*.

FLEX-FL is a PyTorch research implementation for experimenting with peer-to-peer federated learning. Clients train on local dataset partitions and probabilistically combine their model weights with randomly selected peers. The code includes asynchronous execution, an optional iteration-based synchronization mechanism, several aggregation rules, and simulated client delays.

All clients run as Python threads in **one process on one selected CUDA device**. Peer communication uses direct access to client objects in shared memory.

## Repository layout

| File | Purpose |
| --- | --- |
| `run_p2p_async_fl.py` | Parses experiment arguments, generates per-client configurations, creates clients, and starts their threads. |
| `client.py` | Implements local training, evaluation, peer exchange, synchronization, and experiment recording. |
| `aggregation_algorithms.py` | Implements equal model averaging and peer-difference aggregation variants. |
| `ADM.py` | Makes probabilistic aggregation decisions and samples peers uniformly. |
| `dataset.py` | Creates CIFAR partitions and loads CIFAR or FEMNIST client datasets. |
| `netpool.py` | Defines neural networks and the model factory used by experiments. |
| `log.py` | Writes client logs, metrics, and model checkpoints. |
| `create_data_splits.py` | Command-line entry point for CIFAR partition generation. |
| `merge_testset_femnist.py` | Merges FEMNIST JSON files. |
| `explore_params.py` | Launches a Cartesian product of experiment settings as subprocesses. |

## Requirements and setup

The source imports these third-party packages:

- `torch`
- `torchvision`
- `numpy`
- `scipy`

Use a Python 3 environment compatible with your selected PyTorch and torchvision versions, and install a CUDA-enabled PyTorch build compatible with your GPU environment. The runner constructs a device string of the form `cuda:<gpu>`.

The examples use Bash syntax and assume a Linux environment, or WSL with working CUDA access. The dataset splitter uses Unix commands such as `cp` and `mkdir`.

From the repository directory, create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After installing an appropriate CUDA-enabled `torch`/`torchvision` pair in that environment, install the remaining dependencies and check GPU access:

```bash
python -m pip install numpy scipy
python -c "import torch, torchvision, numpy, scipy; print('CUDA available:', torch.cuda.is_available())"
```

CUDA must be available before starting a training experiment. All client models, optimizer state, and temporary aggregation models share the selected GPU, so memory requirements increase with the client count.

## Prepare datasets

### CIFAR-10 and CIFAR-100

The splitter downloads CIFAR through torchvision, reserves 20% of the original training set for validation, and writes serialized dataset objects to disk. Each client receives the same validation and test sets.

Create two IID CIFAR-10 partitions:

```bash
python create_data_splits.py \
  --dataset cifar10 \
  --root ./data/cifar10_2 \
  --num_parts 2 \
  --overlap 0 \
  --iid True
```

For CIFAR-100, use `--dataset cifar100` and a separate root directory.

Use `--iid True` to create IID partitions. The option accepts explicit Boolean values; `--iid` without a value also enables IID splitting.

The resulting layout is:

```text
data/cifar10_2/
├── central/
│   ├── train/data.pth
│   ├── val/data.pth
│   └── test/data.pth
├── part_1/
│   ├── train/data.pth
│   ├── val/data.pth
│   └── test/data.pth
├── part_2/
│   └── ...
└── part_3/                 # Extra partition containing all central training data
    └── ...
```

For `--num_parts N`, train the distributed experiment with `--num_clients N`. The extra `part_{N+1}` is a copy of the central training set, not another disjoint client partition. CIFAR download files also remain under the root.

For IID splitting, the implementation computes `overlap_train = int(overlap * training_size)` and uses sliding slices of length `part_size_train + overlap_train`, where `part_size_train = (training_size - overlap_train) // num_parts`. The splitter seeds Python, NumPy, and PyTorch with `42`. Set the training batch size when launching an experiment.

### FEMNIST

Prepare FEMNIST data as LEAF-style JSON. With the default dataset version `niid_1`, client IDs begin at `1` and map to filenames beginning at `0`:

```text
data/femnist/
├── train/
│   ├── all_data_0_niid_1_keep_100_train_8.json
│   ├── all_data_1_niid_1_keep_100_train_8.json
│   └── ...
└── test/
    ├── all_data_0_niid_1_keep_100_test_8.json
    ├── all_data_1_niid_1_keep_100_test_8.json
    ├── ...
    └── merged_test_set.json
```

Each file must contain `user_data`, mapping user identifiers to `x` image arrays and `y` labels. Images must reshape to `28 × 28`; use labels compatible with 62 output classes. The loader concatenates **all users in each file**, so a simulated client represents a file, not necessarily one writer. Images are converted to float tensors and normalized with mean and standard deviation `0.5`.

The merge helper combines `user_data` and reconstructs `users` and `num_samples`. It excludes the output file from the input scan. Prepare the merged test set with:

```bash
python merge_testset_femnist.py --folder_path ./data/femnist/test --output_path ./data/femnist/test/merged_test_set.json
```

## Run an experiment

### Small CIFAR-10 example

After preparing the two IID partitions above:

```bash
python run_p2p_async_fl.py \
  --dataset_name cifar10 \
  --arch_name resnet18 \
  --num_classes 10 \
  --data_root ./data/cifar10_2 \
  --work_dir ./runs \
  --num_clients 2 \
  --gpu 0 \
  --batch_size 32 \
  --local_iters 1 \
  --total_iters 100 \
  --train_val Classic \
  --fusion_protocol_name fedavg \
  --adm_prob 0.4 \
  --n_peers 1 \
  --latency_mu 0 \
  --stde_mode 0
```

`--stde_mode` is required. Specify `--train_val Classic` for standard local training or `--train_val FedProx` for training with a proximal penalty.

The model factory exposes these dataset/model combinations:

| Dataset | `--arch_name` | `--num_classes` | Input |
| --- | --- | --- | --- |
| `cifar10` | `resnet18` | `10` | RGB, 32 × 32 |
| `cifar100` | `resnet18` | `100` | RGB, 32 × 32 |
| `FEMNIST` | `FEMNIST_CNN` | `62` | Grayscale, 28 × 28 |

Set all three arguments explicitly when changing datasets.

### Training and communication behavior

1. Each client starts with its own independently initialized model and local data.
2. It performs `local_iters` minibatch optimizer steps, restarting iteration over its loader as needed.
3. With probability `adm_prob`, it selects `n_peers` other client IDs and requests their model states.
4. It aggregates its own state with the received states, loads the result, and evaluates it on the test loader.
5. It records metrics and repeats until its iteration counter reaches `total_iters`.

Iteration counts refer to **minibatch steps**. Each round creates a new training loader. The final local round is shortened when necessary to stop at `total_iters`.

Asynchronous execution is the default. `--sync` enables polling of client iteration counters before local training.

### Aggregation rules

Let `W0` be the local model state, `Wi` a selected peer state, and `pi = current_iter_i / total_iters_i` its progress. Operations apply to state-dictionary entries, including model buffers.

| `--fusion_protocol_name` | Current implementation |
| --- | --- |
| `fedavg` | Equal arithmetic mean of the local and selected peer states. |
| `peer_grad` | `W0 - wfo * sum(W0 - Wi)`. |
| `peer_grad_ada` | `W0 - sum((pi / (pi + p0)) * (W0 - Wi))`. |
| `peer_grad_ada_wfo` | Adds a per-peer coefficient to the progress-scaled difference. The coefficient uses a piecewise polynomial of the angle between flattened `W0` and `Wi - W0`; `wfo` controls its interpolation toward `1`. |
| `peer_grad_ada_rev` | Defines an alternative cosine-based coefficient using the direction `W0 - Wi`. |

### Main options

Run `python run_p2p_async_fl.py --help` in the configured environment for the full CLI.

| Option | Default | Meaning |
| --- | --- | --- |
| `--num_clients` | `10` | Number of client threads and required data partitions. |
| `--local_iters` / `--total_iters` | `1` / `100` | Steps per local round / target steps per client. Use positive integers. |
| `--gpu` | `0` | CUDA device index used by every client. |
| `--batch_size` / `--lr` | `32` / `0.01` | Local batch size and initial learning rate. |
| `--loss` / `--optimizer` | `crossentropy` / `sgd` | Cross-entropy loss and stochastic gradient descent. |
| `--momentume` / `--weight_decay` | `0.9` / `0.0005` | SGD momentum and weight decay; preserve the CLI spelling `momentume`. |
| `--scheduler` | `MultistepLR` | Multi-step learning-rate scheduler; use the exact value `MultistepLR`. |
| `--milestones` / `--gamma` | `50,75` / `0.1` | Milestones are percentages of `total_iters`; the scheduler advances every optimizer step. Choose percentages that correspond to integer step indices. |
| `--train_val` | `Classic` | Explicitly use `Classic` or `FedProx`. FedProx uses a fixed coefficient `mu = 1` and the model snapshot at the start of the local round. |
| `--latency_mu` | `0.1` | Multiplier for the mean sampled client delay. |
| `--stde_mode` | Required | Multiplier for delay standard deviation; supply one value. |
| `--adm_prob` | `0.4` | Probability of aggregation after a local round; use a value in `[0, 1]`. |
| `--n_peers` | `1` | Number of sampled client IDs; use `1` through `num_clients - 1`. |
| `--wfo` | `1` | Aggregation coefficient; interpretation depends on the rule above. |
| `--sync` | Off | Enables iteration-counter synchronization. |
| `--valid_interval` | `10` | Validation occurs at round ends whose iteration count is divisible by this positive interval. |
| `--save_update_interval` | `10` | Saves at round ends where `current_iter % interval == 0`, and at training completion. |
| `--log_update_interval` | `10` | Writes the full state dictionary at round ends where `current_iter % interval == 0`. |
| `--work_dir` | `./` | Parent directory for a newly generated timestamped run folder. |

Metrics are written after every round.

For CIFAR-100 the runner overrides milestones with `30,60,90` and gamma with `0.2`. For FEMNIST it overrides milestones with `20,40,60`, gamma with `0.1`, and weight decay with `0.0001`.

Client delays are sampled once per experiment from a normal distribution, then clipped at zero. Mean and standard deviation multipliers use dataset scales of `3.5` for CIFAR-10, `4.4` for CIFAR-100, and `4` for FEMNIST. Classic training sleeps once per minibatch; FedProx sleeps once per local round. These delays simulate additional client waiting time.

## Outputs

```text
runs/<timestamp>/
├── command.txt
├── client_1/
│   ├── g_configs.json
│   ├── train_configs.json
│   ├── aggregation_configs.json
│   ├── log_configs.json
│   ├── logfile
│   ├── experimental.json
│   └── checkpoints/
│       └── <timestamp>
└── client_2/
    └── ...
```

`experimental.json` contains validation and post-aggregation test loss/accuracy, aggregation decisions, selected peers, coefficients/angles, and training/aggregation/round/total timings. Accuracy is recorded as a fraction. Many entries use `[timestamp, current_iter, value]`; `wfo` and `theta` entries instead store the scalar or list returned by the aggregation rule. Test metrics are added after aggregation.

Checkpoints contain model weights in the format `{'state_dict': ...}`.

`command.txt` records the experiment configuration. The per-client JSON files store the settings used by each client.

## Parameter sweeps

`explore_params.py` builds combinations from list-valued arguments and launches all generated commands concurrently, assigning GPUs round-robin. It also rescales each requested `total_iters` by `5 / num_clients`, sets `log_freq` to `local_iters`, and sets checkpoint/log-update intervals to the rescaled total.

## License

Original FLEX-FL contributions are licensed under the [MIT License](LICENSE).

Copyright (c) 2026 C. Sad, D. Masouros, G. Retsinas, D. Soudris, and K. Siozios.

The adapted ResNet and VGG implementations retain their upstream Apache-2.0 and MIT terms, respectively. See [Third-party notices](THIRD_PARTY_NOTICES.md) and the full license texts in [licenses/](licenses/). External datasets and the associated paper are governed by their own terms.


If you use FLEX-FL in your research, please cite :) :

Sad, C., Masouros, D., Retsinas, G., Soudris, D., & Siozios, K. (2026).
FLEX: Flexible federated learning with asynchronous peer-to-peer
communication and adaptive aggregation.
IEEE Transactions on Parallel and Distributed Systems.
