from client import Client
import json
import os 
from netpool import ResNet18, ClientModel
from dataset import Dataset
import shlex
import math
import threading
import argparse
import time 
import os
import json
import numpy as np 

def config_files_gen(id, local_iters, total_iters, gpu, batch_size, lr, loss, 
                      optimizer, momentume, weight_decay, scheduler,milestones,gamma, latency, valid_interval, fusion_protocol_name, train_val,
                      wfo, adm_prob, n_peers, sync, wf_strategy, work_dir, 
                      log_freq, save_update_interval, log_update_interval, verbose):
    # Construct the client-specific directory
    client_dir = os.path.join(work_dir, f"client_{id}")
    
    # Ensure the directory exists
    os.makedirs(client_dir, exist_ok=True)

    # Global config dictionary (g_config.json)
    g_config_data = {
        'id': id,
        'local_iters': local_iters,
        'total_iters': total_iters,
        'gpu': f"cuda:{gpu}"
    }

    # Training config dictionary (train_configs.json)
    train_config_data = {
        'batch_size': batch_size,
        'lr': lr,
        'loss': loss,
        'optimizer': optimizer,
        'momentume':momentume,
        'weight_decay':weight_decay,
        'scheduler': scheduler,
        'milestones':milestones,
        'gamma':gamma,
        'latency': latency,
        'valid_interval': valid_interval,
        'train_val' : train_val
    }

    # Aggregation config dictionary (aggregation_configs.json)
    aggregation_config_data = {
        'fusion_protocol_name': fusion_protocol_name,
        'wfo': wfo,
        'wf_strategy': wf_strategy,
        'adm_prob': adm_prob,
        'sync': sync,
        'n_peers': n_peers
    }

    # Logging config dictionary (log_configs.json)
    log_config_data = {
        'work_dir': f'{work_dir}/client_{id}',
        'log_freq': log_freq,
        'save_update_interval': save_update_interval,
        'log_update_interval' : log_update_interval,
        'verbose': verbose
    }
    
    # Define paths for each configuration file
    g_config_file = os.path.join(client_dir, 'g_configs.json')
    train_config_file = os.path.join(client_dir, 'train_configs.json')
    aggregation_config_file = os.path.join(client_dir, 'aggregation_configs.json')
    log_config_file = os.path.join(client_dir, 'log_configs.json')

    # Write the global config data to g_config.json
    with open(g_config_file, 'w') as f:
        json.dump(g_config_data, f, indent=4)
    
    # Write the training config data to train_configs.json
    with open(train_config_file, 'w') as f:
        json.dump(train_config_data, f, indent=4)
    
    # Write the aggregation config data to aggregation_configs.json
    with open(aggregation_config_file, 'w') as f:
        json.dump(aggregation_config_data, f, indent=4)
    
    # Write the logging config data to log_configs.json
    with open(log_config_file, 'w') as f:
        json.dump(log_config_data, f, indent=4)

    print(f"Configuration files saved to {client_dir}")
    return g_config_data, train_config_data, aggregation_config_data, log_config_data

def gen_conf_file_for_all(num_clients,local_iters, total_iters, gpu, batch_size, lr, loss, 
                      optimizer, momentume, weight_decay, scheduler, milestones, gamma, c_latency, valid_interval, fusion_protocol_name, train_val, 
                      wfo, adm_prob,n_peers,sync, wf_strategy,work_dir, 
                      log_freq, save_update_interval, log_update_interval, verbose):
    for i in range(num_clients):
        config_files_gen(i+1,local_iters, total_iters, gpu, batch_size, lr, loss, 
                      optimizer, momentume, weight_decay, scheduler,milestones,gamma, c_latency[i], valid_interval,fusion_protocol_name, train_val, 
                      wfo,adm_prob, n_peers, sync, wf_strategy, work_dir, 
                      log_freq, save_update_interval,log_update_interval, verbose)

def read_client_json_configs(work_dir):
    """Reads JSON configuration files from the given directory."""
    gconfigs_f = f'{work_dir}/g_configs.json'
    train_configs_f = f'{work_dir}/train_configs.json'
    aggregation_configs_f = f'{work_dir}/aggregation_configs.json'
    log_configs_f = f'{work_dir}/log_configs.json'
    if not os.path.exists(gconfigs_f):
        raise FileNotFoundError(f'No file found at {gconfigs_f}')
    if not os.path.exists(train_configs_f):
        raise FileNotFoundError(f'No file found at {train_configs_f}')
    if not os.path.exists(aggregation_configs_f):
        raise FileNotFoundError(f'No file found at {aggregation_configs_f}')
    if not os.path.exists(log_configs_f):
        raise FileNotFoundError(f'No file found at {log_configs_f}')
    with open(gconfigs_f, 'r') as f:
        gconfigs = json.load(f)
    with open(train_configs_f, 'r') as f:
        train_configs = json.load(f)
    with open(aggregation_configs_f, 'r') as f:
        aggregation_configs = json.load(f)
    with open(log_configs_f, 'r') as f:
        log_configs = json.load(f)
    return gconfigs, train_configs, aggregation_configs, log_configs


def create_all_clients_add(num_clients,work_dir,num_classes,dataset_name,data_root,arch_name):
    """Creates all clients."""
    clients = []
    for i in range(num_clients):
        "Read client configs"
        # gconfigs, train_configs, aggregation_configs, log_configs = read_client_json_configs(f'{work_dir}/client_{i+1}')
        gconfigs, train_configs, aggregation_configs, log_configs = config_files_gen(i+1,local_iters, total_iters, gpu, batch_size, lr, loss, 
                      optimizer, momentume, weight_decay, scheduler,milestones,gamma, c_latency[i], valid_interval,fusion_protocol_name, train_val, 
                      wfo,adm_prob, n_peers, sync, wf_strategy, work_dir, 
                      log_freq, save_update_interval,log_update_interval, verbose)
        
        # nn_model = ResNet18(num_classes)
        nn_model = ClientModel(num_classes,arch_name).get_mymodel()
        dataset = Dataset(dataset_name,data_root).get_dataset()
        
        client = Client(gconfigs, train_configs, aggregation_configs, log_configs,nn_model,dataset,arch_name)
        clients.append(client)
    
    for client_ in clients:
        # rest_clients = clients[:client_.id-1]+clients[:client_.id:]
        # client_.add_other_clients(rest_clients)
        client_.add_other_clients(clients)
    
    return clients


def start_all_clients(num_clients,work_dir,num_classes,dataset_name,data_root,arch_name):
    try:
        """Starts all clients in parallel"""
        clients = create_all_clients_add(num_clients,work_dir,num_classes,dataset_name,data_root,arch_name)
        threads = []
        for client in clients:
            threads.append(threading.Thread(target=client.start_client))
            threads[-1].start()

        for t in threads:
            t.join()

    except KeyboardInterrupt:
        print("Main process received KeyboardInterrupt, terminating child processes...")
        for p in processes:
            p.terminate()  # Forcefully terminate each child process
            p.join()  # Ensure each process has finished    
    


if __name__ == '__main__':
    """add argument for the inputs"""
    parser = argparse.ArgumentParser(description='Peer to Peer Async FL')
    
    parser.add_argument('--num_clients', type=int, default=10, help='number of clients')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100','FEMNIST'], help='dataset')
    parser.add_argument('--arch_name', type=str, default='resnet18', choices=['resnet18', 'FEMNIST_CNN'],help='Neural network architecture name (resnet18 or FEMNIST)')
    parser.add_argument('--work_dir', type=str, default='./', help='working directory')
    parser.add_argument('--data_root', type=str, default='./data', help='data root')
    parser.add_argument('--local_iters', type=int, default=1, help='local iterations for the clients')
    parser.add_argument('--total_iters', type=int, default=100, help='total iterations')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--loss', type=str, default='crossentropy', choices=['crossentropy'], help='loss function')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd'], help='optimizer')
    parser.add_argument('--momentume', type=float, default=0.9, help='momentume')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--scheduler', type=str, default='MultistepLR', choices=['MultistepLR'], help='scheduler')
    parser.add_argument('--milestones', type=str, default='50,75', help='milestones')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--latency_mu', type=float, default=0.1, help='latency mu value')
    parser.add_argument("--stde_mode", type=float, nargs='+', required=True, help="stde for latency values (higher stde - higher hetero)")
    parser.add_argument('--valid_interval', type=int, default=10, help='validation interval')
    parser.add_argument('--fusion_protocol_name', type=str, default='fedavg', choices=['fedavg', 'peer_grad', 'peer_grad_ada', 'peer_grad_ada_wfo', 'peer_grad_ada_rev'], help='fusion protocol name')
    parser.add_argument('--train_val', type=str, default='Classic', choices=['Classic', 'FedProx'], help='Local training method')
    parser.add_argument('--wfo', type=float, default=1, help='wfo for aggregation')
    parser.add_argument('--adm_prob', type=float, default=0.4, help='adm prob for aggregation')
    parser.add_argument('--n_peers', type=int, default=1, help='number of peer for the topology')
    parser.add_argument("--sync", action="store_true", help="Run in synchronous mode")
    parser.add_argument('--wf_strategy', type=str, default='ada', help='wf strategy')
    parser.add_argument('--log_freq', type=int, default=10, help='log frequency')
    parser.add_argument('--save_update_interval', type=int, default=10, help='save model weights')
    parser.add_argument('--log_update_interval', type=int, default=10, help='log model weights')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')



    args = parser.parse_args()

    for name in ('num_clients', 'num_classes', 'local_iters', 'total_iters', 'batch_size',
                 'valid_interval', 'save_update_interval', 'log_update_interval', 'log_freq'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if not 1 <= args.n_peers < args.num_clients:
        parser.error('--n_peers must be between 1 and num_clients - 1')
    if not 0 <= args.adm_prob <= 1:
        parser.error('--adm_prob must be between 0 and 1')
    for name in ('latency_mu', 'weight_decay', 'momentume'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f'--{name} must be finite and nonnegative')
    if any(not math.isfinite(v) or v < 0 for v in args.stde_mode):
        parser.error('--stde_mode values must be finite and nonnegative')
    for name in ('lr', 'gamma'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'--{name} must be finite and positive')
    if not math.isfinite(args.wfo):
        parser.error('--wfo must be finite')
    try:
        milestone_values = [int(x) for x in args.milestones.split(',')]
    except ValueError:
        parser.error('--milestones must be comma-separated integer percentages')
    if any(not 0 <= value <= 100 for value in milestone_values):
        parser.error('--milestones must be percentages between 0 and 100')

    num_clients = args.num_clients
    work_dir_ = args.work_dir
    t = time.time()
    work_dir = f'{work_dir_}/{int(t)}_{int(100000*(t-int(t)))}'
    num_classes = args.num_classes
    dataset_name = args.dataset_name
    data_root = args.data_root
    arch_name = args.arch_name
    local_iters = args.local_iters
    total_iters = args.total_iters
    gpu = args.gpu
    batch_size = args.batch_size
    lr = args.lr
    loss = args.loss
    optimizer = args.optimizer
    momentume = args.momentume
    weight_decay = args.weight_decay
    scheduler = args.scheduler
    milestones = [int(x) for x in args.milestones.split(',')]   
    gamma = args.gamma
    stde_mode = args.stde_mode[0]
     # print(stde_mode)
    latency_mu = args.latency_mu
     # lat = {"FEMNIST":4, "cifar10":3.5, "cifar100":4.4}
     
    def generate_client_latencies(num_clients, dataset_name, mu_mode, stde_mode):
        # Latency means per dataset
        lat = {"FEMNIST": 4, "cifar10": 3.5, "cifar100": 4.4}

        if dataset_name not in lat:
            raise ValueError(f"Unknown dataset: {dataset_name}. Valid options: {list(lat.keys())}")

        mu = mu_mode * lat[dataset_name]
        stde = stde_mode * lat[dataset_name]
        # Generate latencies from normal distribution
        latencies = np.random.normal(loc=mu, scale=stde, size=num_clients)

        # Ensure all latencies are non-negative
        latencies = np.clip(latencies, a_min=0, a_max=None)

        return latencies
     
     
    c_latency = generate_client_latencies(num_clients,dataset_name,latency_mu,stde_mode)
    # c_latency= num_clients * [0]
    # if latency > 0:
    #     for i in range(num_clients):
    #         if i%num_clients <= latency_rate/3:
    #             c_latency[i] = lat[dataset_name]
    #         elif i%num_clients <= 2*latency_rate/3:
    #             c_latency[i] = 2*lat[dataset_name]
    #         elif i%num_clients < latency_rate:
    #             c_latency[i] = 3*lat[dataset_name]
    #         elif i%num_clients >= latency_rate:
    #             c_latency[i] = 0


    valid_interval = args.valid_interval
    fusion_protocol_name = args.fusion_protocol_name
    train_val = args.train_val
    wfo = args.wfo
    adm_prob = args.adm_prob
    n_peers = args.n_peers
    sync = args.sync
    wf_strategy = args.wf_strategy
    log_freq = args.log_freq
    save_update_interval = args.save_update_interval
    log_update_interval = args.log_update_interval
    verbose = args.verbose
    if args.dataset_name == 'cifar100':
        milestones = [30,60,90]
        gamma = 0.2
    if args.dataset_name == 'FEMNIST':
        milestones = [20,40,60]
        gamma = 0.1
        weight_decay = 1e-4
    command_args = vars(args).copy()
    command_args.update(milestones=','.join(map(str, milestones)), gamma=gamma,
                        weight_decay=weight_decay)
    command_parts = ['python3', 'run_p2p_async_fl.py']
    for key, value in command_args.items():
        if key == 'sync':
            if value:
                command_parts.append('--sync')
        else:
            command_parts.append(f'--{key}')
            command_parts.extend(map(str, value if isinstance(value, list) else [value]))
    command = shlex.join(command_parts)
    print(command)
    """ check and create folder and then save the command in a txt"""
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    with open(f'{work_dir}/command.txt', 'w') as f:
        f.write(command)
    
    gen_conf_file_for_all(num_clients,local_iters,total_iters,gpu,batch_size,lr,loss,optimizer,momentume,weight_decay,scheduler,milestones,gamma,c_latency,valid_interval,fusion_protocol_name,train_val,\
                        wfo,adm_prob,n_peers,sync,wf_strategy,work_dir,log_freq,save_update_interval,log_update_interval,verbose)
    
    start_all_clients(num_clients,work_dir,num_classes,dataset_name,data_root,arch_name)
