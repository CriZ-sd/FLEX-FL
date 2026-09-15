import itertools
import subprocess
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple instances of a command with argument combinations.")
    parser.add_argument("--num_clients", type=int, nargs='+', required=True, help="Number of clients.")
    # parser.add_argument("--num_classes", type=int, nargs='+', required=True, help="Number of classes.")
    parser.add_argument("--dataset", type=str, nargs='+', required=True, help="Dataset name.")
    parser.add_argument("--work_dir", type=str, nargs='+', required=True, help="Working directory.")
    # parser.add_argument("--data_root", type=str, nargs='+', required=True, help="Data root directory.")
    parser.add_argument("--local_iters", type=int, nargs='+', required=True, help="Local iterations.")
    parser.add_argument("--total_iters", type=int, nargs='+', required=True, help="Total iterations.")
    parser.add_argument("--gpu", type=int, nargs='+', required=True, help="List of available GPUs to distribute workloads.")
    parser.add_argument("--batch_size", type=int, nargs='+', required=True, help="Batch size.")
    parser.add_argument("--lr", type=float, nargs='+', required=True, help="Learning rate.")
    parser.add_argument("--latency_mu", type=float, nargs='+', required=True, help="Latency mu.")
    parser.add_argument("--stde_mode", type=float, nargs='+', required=True, help="Latency stde.")
    parser.add_argument("--valid_interval", type=int, nargs='+', required=True, help="Validation interval.")
    parser.add_argument("--fusion_protocol_name", type=str, nargs='+', required=True, help="Fusion protocol name.")
    parser.add_argument("--train_val", type=str, nargs='+', required=True, help="Train use FedProx penalty(FedProx) or classic train (Classic)")
    parser.add_argument("--wfo", type=float, nargs='+', required=True, help="WFO value.")
    parser.add_argument("--adm_prob", type=float, nargs='+', required=True, help="Admission probability.")
    parser.add_argument("--n_peers", type=int, nargs='+', required=True, help="Number of peers for the topology.")
    parser.add_argument("--sync", action="store_true", help="Run in synchronous mode")

    return vars(parser.parse_args())

def generate_commands(args):
    data_folder = '/storage/data2/chsad'
    keys = [k for k in args.keys() if k not in ["gpu", "log_freq", "save_update_interval", "log_update_interval","sync"]]
    values = [args[k] for k in keys]
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    gpus = args["gpu"]
    num_gpus = len(gpus)
    # latency_rate = args["latency_rate"][0]
    commands = []
    # lat = {"FEMNIST":4, "cifar10":3.5, "cifar100":4.4}
    for i, combo in enumerate(all_combinations):
        # data_folder = '/home/ubuntu/RAFed/SimpleFED'
        if combo["dataset"] == 'cifar10':
            combo["data_root"] = f'{data_folder}/APEX-G/data/cifar10_{combo["num_clients"]}'
            combo["num_classes"] = 10
            combo["arch_name"] = 'resnet18'
        elif combo["dataset"] == 'cifar100':
            combo["data_root"] = f'{data_folder}/APEX-G/data/cifar100_{combo["num_clients"]}'
            combo["num_classes"] = 100
            combo["arch_name"] = 'resnet18'
        elif combo["dataset"] == 'FEMNIST':
            # combo["data_root"] = f'/home/ubuntu/leaf/data/femnist/data_10'
            combo["data_root"] = f'{data_folder}/leaf/data/femnist/data'
            combo["num_classes"] = 62
            combo["arch_name"] = 'FEMNIST_CNN'
        gpu_id = gpus[i % num_gpus]  # Distribute tasks across GPUs
        
        # if i%combo["num_clients"] >= latency_rate:
            # combo["latency"] = 0
        # if combo["latency"] > 0 :
        #     if i%combo["num_clients"] <= latency_rate/3:
        #         combo["latency"] = lat[combo["dataset"]]
        #     elif i%combo["num_clients"] <= 2*latency_rate/3:
        #         combo["latency"] = 2*lat[combo["dataset"]]
        #     elif i%combo["num_clients"] < latency_rate:
        #         combo["latency"] = 3*lat[combo["dataset"]]
        #     elif i%combo["num_clients"] >= latency_rate:
        #         combo["latency"] = 0

        combo["total_iters"] = int(combo["total_iters"] * 5/combo["num_clients"])
        combo["log_freq"] = combo["local_iters"]
        combo["save_update_interval"] = combo["total_iters"]
        combo["log_update_interval"] = combo["total_iters"]
        #combo["latency_rate"] = latency_rate
        
        cmd = ["python3", "run_p2p_async_fl.py"]
        for key, value in combo.items():
            cmd.append(f"--{key}")
            cmd.append(str(value))
        cmd.append("--gpu")
        cmd.append(str(gpu_id))
        if args["sync"]:
            cmd.append("--sync")
        commands.append((cmd, i))
    
    return commands

def run_commands(commands,work_dir):
    processes = []
    for cmd, idx in commands:
        log_file = f"{work_dir}/output_{idx}.txt"
        with open(log_file, "w") as out:
            print(f"Running: {' '.join(cmd)} (Logging to {log_file})")
            processes.append(subprocess.Popen(cmd, stdout=out, stderr=subprocess.PIPE))
    
    for p in processes:
        _, err = p.communicate()
        if err:
            print(f"Error encountered: {err.decode()}")

def main():
    args = parse_args()
    # print(args["work_dir"][0])
    commands = generate_commands(args)
    if not os.path.exists(args["work_dir"][0]):
        os.makedirs(args["work_dir"][0])
    run_commands(commands,args["work_dir"][0])

if __name__ == "__main__":
    main()
