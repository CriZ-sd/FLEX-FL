import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import random_split
import random
import numpy as np
import os 
import json
from scipy.stats import dirichlet


class Cifar10 :
    def __init__(self,root,name) -> None:
        self.root=root
        self.name = name

    def create_splits_(self,num_parts, batch_size, overlap=0.0, iid=True):
        root = self.root
        # Ensure reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        # # Download CIFAR-10 dataset
        # transform = transforms.Compose([transforms.ToTensor()])
        # trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        # testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)

        # valid_loader = test_loader

        # Calculate overlap samples
        num_train = int(0.8*len(trainset)) #80% of train data
        num_test = len(testset)
        num_val = int(len(trainset) - num_train) #20% of train data

        trainset, validset = random_split(trainset, [num_train, num_val])

        overlap_train = int(overlap * num_train)
        # overlap_val = int(overlap * num_val)
        part_size_train = (num_train - overlap_train) // num_parts
        # part_size_val = (num_test - overlap_val) // num_parts

        # Create directories
        for i in range(num_parts):
            os.makedirs(os.path.join(root, f'part_{i+1}', 'train'), exist_ok=True)
            os.makedirs(os.path.join(root, f'part_{i+1}', 'test'), exist_ok=True)
            os.makedirs(os.path.join(root,f'part_{i+1}', 'val'), exist_ok=True)
        os.makedirs(os.path.join(root,f'central', 'train'), exist_ok=True)
        os.makedirs(os.path.join(root, f'central', 'test'), exist_ok=True)
        os.makedirs(os.path.join(root,f'central', 'val'), exist_ok=True)



        def save_subset(subset, path):
            torch.save(subset, path)
        for i in range(num_parts):
            save_subset(testset, os.path.join(root, f'part_{i+1}', 'test', 'data.pth'))
            save_subset(validset,os.path.join(root, f'part_{i+1}', 'val', 'data.pth'))

        save_subset(trainset, os.path.join(root, f'central', 'train', 'data.pth'))
        save_subset(testset, os.path.join(root, f'central', 'test', 'data.pth'))
        save_subset(validset, os.path.join(root, f'central', 'val', 'data.pth'))



        if iid:
            # Shuffle indices
            indices_train = np.arange(num_train)
            indices_val = np.arange(num_val)
            np.random.shuffle(indices_train)
            np.random.shuffle(indices_val)

            for i in range(num_parts):
                start_train = i * part_size_train
                end_train = start_train + part_size_train
                # start_val = i * part_size_val
                # end_val = start_val + part_size_val

                subset_train = Subset(trainset, indices_train[start_train:end_train + overlap_train])
                # subset_val = Subset(validset, indices_val[start_val:end_val + overlap_val])

                save_subset(subset_train, os.path.join(root, f'part_{i+1}', 'train', 'data.pth'))
                # save_subset(subset_val, os.path.join(root, f'part_{i+1}', 'val', 'data.pth'))
        else:
            class_indices_train = {i: np.where(np.array(trainset.targets) == i)[0] for i in range(10)}
            # class_indices_val = {i: np.where(np.array(validset.targets) == i)[0] for i in range(10)}

            for i in range(num_parts):
                part_indices_train = []
                part_indices_test = []

                for c in range(10):
                    np.random.shuffle(class_indices_train[c])
                    # np.random.shuffle(class_indices_val[c])

                    class_part_size_train = part_size_train // 10
                    # class_part_size_val = part_size_val // 10

                    start_train = i * class_part_size_train
                    end_train = start_train + class_part_size_train

                    # start_val = i * class_part_size_val
                    # end_val = start_val + class_part_size_val

                    part_indices_train.extend(class_indices_train[c][start_train:end_train + overlap_train // 10])
                    # part_indices_test.extend(class_indices_val[c][start_val:end_test + overlap_val // 10])

                subset_train = Subset(trainset, part_indices_train)
                # subset_val = Subset(testset, part_indices_test)

                save_subset(subset_train, os.path.join(root, f'part_{i+1}', 'train', 'data.pth'))
                # save_subset(subset_val, os.path.join(root, f'part_{i+1}', 'val', 'data.pth'))

        print(f"Dataset split into {num_parts} parts with overlap={overlap}, iid={iid}.")
    
    def create_splits(self, num_parts, batch_size, overlap=0.0, iid=True, alpha=0.01):
        root = self.root
        # Ensure reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

        # Split train and validation sets
        num_train = int(0.8 * len(trainset))  # 80% of train data
        num_test = len(testset)
        num_val = int(len(trainset) - num_train)  # 20% of train data

        trainset, validset = torch.utils.data.random_split(trainset, [num_train, num_val])

        overlap_train = int(overlap * num_train)
        part_size_train = (num_train - overlap_train) // num_parts

        # Create directories
        for i in range(num_parts):
            os.makedirs(os.path.join(root, f'part_{i+1}', 'train'), exist_ok=True)
            os.makedirs(os.path.join(root, f'part_{i+1}', 'test'), exist_ok=True)
            os.makedirs(os.path.join(root, f'part_{i+1}', 'val'), exist_ok=True)
        os.makedirs(os.path.join(root, f'central', 'train'), exist_ok=True)
        os.makedirs(os.path.join(root, f'central', 'test'), exist_ok=True)
        os.makedirs(os.path.join(root, f'central', 'val'), exist_ok=True)

        def save_subset(subset, path):
            torch.save(subset, path)

        # Save test and validation subsets
        for i in range(num_parts):
            save_subset(testset, os.path.join(root, f'part_{i+1}', 'test', 'data.pth'))
            save_subset(validset, os.path.join(root, f'part_{i+1}', 'val', 'data.pth'))

        save_subset(trainset, os.path.join(root, f'central', 'train', 'data.pth'))
        save_subset(testset, os.path.join(root, f'central', 'test', 'data.pth'))
        save_subset(validset, os.path.join(root, f'central', 'val', 'data.pth'))

        if iid:
            # Shuffle indices for IID splits
            indices_train = np.arange(num_train)
            indices_val = np.arange(num_val)
            np.random.shuffle(indices_train)
            np.random.shuffle(indices_val)

            for i in range(num_parts):
                start_train = i * part_size_train
                end_train = start_train + part_size_train

                subset_train = Subset(trainset, indices_train[start_train:end_train + overlap_train])
                save_subset(subset_train, os.path.join(root, f'part_{i+1}', 'train', 'data.pth'))
        else:
            # Non-IID using Dirichlet distribution
            # Access the targets via trainset.dataset.targets
            class_indices_train = {i: np.where(np.array(trainset.dataset.targets) == i)[0] for i in range(10)}

            # Dirichlet distribution parameters (controls the skew)
            dirichlet_params = np.full(10, alpha)  # Concentration parameter
            class_distribution_per_part = []

            # Simulate non-IID splits using Dirichlet distribution
            for i in range(num_parts):
                part_indices_train = []

                # Generate class proportions using Dirichlet distribution for each part
                class_proportions = dirichlet.rvs(dirichlet_params, size=1).flatten()

                # Normalize the proportions to sum up to the total number of samples for that part
                total_samples = len(trainset) // num_parts
                class_samples = (class_proportions * total_samples).astype(int)

                # Ensure each class has at least 1 sample in each part
                class_samples[class_samples == 0] = 1

                # Collect indices based on the proportions generated
                for c in range(10):
                    np.random.shuffle(class_indices_train[c])
                    part_indices_train.extend(class_indices_train[c][:class_samples[c]])

                # Save the subset for this part
                subset_train = Subset(trainset, part_indices_train)
                save_subset(subset_train, os.path.join(root, f'part_{i+1}', 'train', 'data.pth'))

            print(f"Dataset split into {num_parts} parts with overlap={overlap}, iid={iid}, alpha={alpha}.")    
    def load_client_sets(self,client_id, batch_size=64):
        root = self.root
        # Construct paths to the client's data
        train_path = os.path.join(root, f'part_{client_id}', 'train', 'data.pth')
        test_path = os.path.join(root, f'part_{client_id}', 'test', 'data.pth')
        val_path = os.path.join(root, f'part_{client_id}', 'val', 'data.pth')
        # Load the datasets
        train_dataset = torch.load(train_path,weights_only=False)
        test_dataset = torch.load(test_path,weights_only=False)
        valid_dataset = torch.load(val_path,weights_only=False)
        num_examples = {"trainset" : len(train_dataset), "testset" : len(test_dataset), 'validset':len(valid_dataset)}
        return train_dataset, test_dataset, valid_dataset, num_examples

    def load_client_loaders(self,train_dataset,test_dataset,valid_dataset,num_examples,gpu,batch_size=64):
        
        print(num_examples)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset,batch_size=batch_size, shuffle=False)
        num_examples = num_examples
        # train_loader, test_loader, valid_loader = train_loader.to(gpu), test_loader.to(gpu), valid_loader.to(gpu)

        return train_loader, test_loader, valid_loader, num_examples

class Cifar100 :
    def __init__(self,root,name) -> None:
        self.root=root
        self.name = name
    
    def create_splits(self,num_parts, batch_size, overlap=0.0, iid=True):
        root = self.root
        # Ensure reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        # Define the transform to apply to the images (e.g., normalization, resizing)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Normalize with the CIFAR-100 mean and std
        # ])

        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # Randomly crop with padding
        transforms.RandomHorizontalFlip(),         # Randomly flip horizontally
        transforms.ToTensor(),                     # Convert to tensor
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # Dataset-specific normalization
                         std=[0.2675, 0.2565, 0.2761]),  # Dataset-specific normalization
        ])
# 
        # Testing transforms for CIFAR-100
        test_transform = transforms.Compose([
        transforms.ToTensor(),                     # Convert to tensor
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # Dataset-specific normalization
                         std=[0.2675, 0.2565, 0.2761]),  # Dataset-specific normalization
        ])

        # Download the CIFAR-100 training dataset
        trainset = torchvision.datasets.CIFAR100(root=root, train=True,
                                                 download=True, transform=train_transform)

        # Download the CIFAR-100 test dataset
        testset = torchvision.datasets.CIFAR100(root=root, train=False,
                                                download=True, transform=test_transform)

        # Create data loaders to iterate through the dataset
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        valid_loader = test_loader

        # Calculate overlap samples
        num_train = int(0.8*len(trainset)) #80% of train data
        num_test = len(testset)
        num_val = int(len(trainset) - num_train) #20% of train data

        trainset, validset = random_split(trainset, [num_train, num_val])

        overlap_train = int(overlap * num_train)
        # overlap_val = int(overlap * num_val)
        part_size_train = (num_train - overlap_train) // num_parts
        # part_size_val = (num_test - overlap_val) // num_parts

        # Create directories
        for i in range(num_parts):
            os.makedirs(os.path.join(root, f'part_{i+1}', 'train'), exist_ok=True)
            os.makedirs(os.path.join(root, f'part_{i+1}', 'test'), exist_ok=True)
            os.makedirs(os.path.join(root,f'part_{i+1}', 'val'), exist_ok=True)
        os.makedirs(os.path.join(root,f'central', 'train'), exist_ok=True)
        os.makedirs(os.path.join(root, f'central', 'test'), exist_ok=True)
        os.makedirs(os.path.join(root,f'central', 'val'), exist_ok=True)



        def save_subset(subset, path):
            torch.save(subset, path)
        for i in range(num_parts):
            save_subset(testset, os.path.join(root, f'part_{i+1}', 'test', 'data.pth'))
            save_subset(validset,os.path.join(root, f'part_{i+1}', 'val', 'data.pth'))

        save_subset(trainset, os.path.join(root, f'central', 'train', 'data.pth'))
        save_subset(testset, os.path.join(root, f'central', 'test', 'data.pth'))
        save_subset(validset, os.path.join(root, f'central', 'val', 'data.pth'))



        if iid:
            # Shuffle indices
            indices_train = np.arange(num_train)
            indices_val = np.arange(num_val)
            np.random.shuffle(indices_train)
            np.random.shuffle(indices_val)

            for i in range(num_parts):
                start_train = i * part_size_train
                end_train = start_train + part_size_train
                # start_val = i * part_size_val
                # end_val = start_val + part_size_val

                subset_train = Subset(trainset, indices_train[start_train:end_train + overlap_train])
                # subset_val = Subset(validset, indices_val[start_val:end_val + overlap_val])

                save_subset(subset_train, os.path.join(root, f'part_{i+1}', 'train', 'data.pth'))
                # save_subset(subset_val, os.path.join(root, f'part_{i+1}', 'val', 'data.pth'))
        else:
            class_indices_train = {i: np.where(np.array(trainset.targets) == i)[0] for i in range(100)}
            # class_indices_val = {i: np.where(np.array(validset.targets) == i)[0] for i in range(10)}

            for i in range(num_parts):
                part_indices_train = []
                part_indices_test = []

                for c in range(100):
                    np.random.shuffle(class_indices_train[c])
                    # np.random.shuffle(class_indices_val[c])

                    class_part_size_train = part_size_train // 100
                    # class_part_size_val = part_size_val // 10

                    start_train = i * class_part_size_train
                    end_train = start_train + class_part_size_train

                    # start_val = i * class_part_size_val
                    # end_val = start_val + class_part_size_val

                    part_indices_train.extend(class_indices_train[c][start_train:end_train + overlap_train // 100])
                    # part_indices_test.extend(class_indices_val[c][start_val:end_test + overlap_val // 10])

                subset_train = Subset(trainset, part_indices_train)
                # subset_val = Subset(testset, part_indices_test)

                save_subset(subset_train, os.path.join(root, f'part_{i+1}', 'train', 'data.pth'))
                # save_subset(subset_val, os.path.join(root, f'part_{i+1}', 'val', 'data.pth'))

        print(f"Dataset split into {num_parts} parts with overlap={overlap}, iid={iid}.")
    
    def load_client_sets(self,client_id, batch_size=64):
        root = self.root
        # Construct paths to the client's data
        train_path = os.path.join(root, f'part_{client_id}', 'train', 'data.pth')
        test_path = os.path.join(root, f'part_{client_id}', 'test', 'data.pth')
        val_path = os.path.join(root, f'part_{client_id}', 'val', 'data.pth')


        # Load the datasets
        train_dataset = torch.load(train_path,weights_only=False)
        test_dataset = torch.load(test_path,weights_only=False)
        valid_dataset = torch.load(val_path,weights_only=False)
        
        num_examples = {"trainset" : len(train_dataset), "testset" : len(test_dataset), 'validset':len(valid_dataset)}


        return train_dataset, test_dataset, valid_dataset, num_examples
    
    def load_client_loaders(self,train_dataset,test_dataset,valid_dataset,num_examples,gpu,batch_size=64):
        

        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset,batch_size=batch_size, shuffle=False)
        num_examples = num_examples


        return train_loader, test_loader, valid_loader, num_examples

# class FEMNIST:
#     def __init__(self, root,name):
#         """
#         Handles loading and processing of the FEMNIST dataset from LEAF.

#         Args:
#             root (str): Path to the FEMNIST data directory.
            
#         """
#         self.root = root
#         self.name = name

#     def load_client_sets(self, client_id,batch_size):
#         """
#         Loads training, validation, and test data for a specific client.

#         Args:
#             client_id (str): Client identifier (e.g., 'f0000_00').

#         Returns:
#             tuple: (train_dataset, test_dataset, valid_dataset, num_examples)
#         """
#         train_path = os.path.join(self.root, "train", f"all_data_{client_id}.json")
#         test_path = os.path.join(self.root, "test", f"all_data_{client_id}.json")
#         val_path = os.path.join(self.root, "val", f"all_data_{client_id}.json")

#         train_data = self._load_json(train_path, client_id)
#         test_data = self._load_json(test_path, client_id)
#         val_data = self._load_json(val_path, client_id)

#         # Convert to PyTorch Datasets
#         train_dataset = self._create_dataset(train_data["x"], train_data["y"])
#         test_dataset = self._create_dataset(test_data["x"], test_data["y"])
#         valid_dataset = self._create_dataset(val_data["x"], val_data["y"])

#         num_examples = {
#             "trainset": len(train_dataset),
#             "testset": len(test_dataset),
#             "validset": len(valid_dataset),
#         }

#         return train_dataset, test_dataset, valid_dataset, num_examples

#     def load_client_loaders(self, train_dataset,test_dataset,valid_dataset,num_examples,gpu,batch_size=64):
#         """
#         Creates DataLoaders for a given client's dataset.

#         Args:
#             train_dataset (Dataset): Training dataset.
#             test_dataset (Dataset): Test dataset.
#             valid_dataset (Dataset): Validation dataset.
#             num_examples (dict): Number of examples in each set.

#         Returns:
#             tuple: (train_loader, test_loader, valid_loader, num_examples)
#         """
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#         valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

#         return train_loader, test_loader, valid_loader, num_examples

#     def _load_json(self, file_path, client_id):
#         """
#         Loads client-specific data from a LEAF JSON file.

#         Args:
#             file_path (str): Path to the JSON file.
#             client_id (str): Client identifier.

#         Returns:
#             dict: Dictionary containing 'x' (images) and 'y' (labels).
#         """
#         with open(file_path, "r") as f:
#             data = json.load(f)

#         if client_id not in data["users"]:
#             raise ValueError(f"Client {client_id} not found in {file_path}")

#         client_data = data["user_data"][client_id]
#         images = np.array(client_data["x"]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
#         labels = np.array(client_data["y"])

#         return {"x": images, "y": labels}

# import os
# import json
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

# # Define the image size based on FEMNIST dataset
# IMAGE_SIZE = 28  # This might be 28x28 based on FEMNIST format

class FEMNISTDataset:
    """
    Custom Dataset for FEMNIST data.
    """
    def __init__(self, images, labels, transform=None):
        self.images = torch.tensor(images, dtype=torch.float32)  # Convert to float32
        self.labels = torch.tensor(labels, dtype=torch.long)  # Keep labels as long
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].numpy()  # Convert to NumPy (HWC format expected by transforms)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)  # Apply transform (expects PIL or NumPy)

        return image, label


class FEMNIST:
    def __init__(self, root, name, version="niid_1"):
        """
        Handles loading and processing of the FEMNIST dataset from LEAF.

        Args:
            root (str): Path to the FEMNIST data directory.
            name (str): Name of the dataset (for logging or reference).
            version (str): Version of the dataset (e.g., "default", "niid", etc.).
        """
        self.root = root
        self.name = name
        self.version = version  # Add version to handle different dataset types (default, niid, etc.)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5 (for grayscale)
        ])

    def load_client_sets(self, client_id, batch_size=64):
        """
        Loads training, validation, and test data for a specific client.

        Args:
            client_id (str): Client identifier (e.g., 'f0000_00').
            batch_size (int): Batch size for data loading.

        Returns:
            tuple: (train_dataset, test_dataset, valid_dataset, num_examples)
        """
        # Specify paths based on version
        file_suffix = f"_{self.version}" if self.version != "default" else ""
        
        # Construct file paths for train, test, and validation data
        train_path = os.path.join(self.root, "train", f"all_data_{client_id-1}{file_suffix}_keep_100_train_8.json")
        # test_path = os.path.join(self.root, "test", f"all_data_{client_id-1}{file_suffix}_keep_10_test_9.json")
        val_path = os.path.join(self.root, "test", f"all_data_{client_id-1}{file_suffix}_keep_100_test_8.json")
        test_path = os.path.join(self.root, "test", f"merged_test_set.json")
        # val_path = os.path.join(self.root, "test", f"merged_test_set.json")

        # Load the data from JSON files
        train_data = self._load_json(train_path, client_id)
        test_data = self._load_json(test_path, client_id)
        val_data = self._load_json(val_path, client_id)

        # Convert to PyTorch Datasets
        train_dataset = self._create_dataset(train_data["x"], train_data["y"])
        test_dataset = self._create_dataset(test_data["x"], test_data["y"])
        valid_dataset = self._create_dataset(val_data["x"], val_data["y"])

        num_examples = {
            "trainset": len(train_dataset),
            "testset": len(test_dataset),
            "validset": len(valid_dataset),
        }

        return train_dataset, test_dataset, valid_dataset, num_examples

    def load_client_loaders(self, train_dataset, test_dataset, valid_dataset, num_examples, gpu, batch_size=64):
        """
        Creates DataLoaders for a given client's dataset.

        Args:
            train_dataset (Dataset): Training dataset.
            test_dataset (Dataset): Test dataset.
            valid_dataset (Dataset): Validation dataset.
            num_examples (dict): Number of examples in each set.

        Returns:
            tuple: (train_loader, test_loader, valid_loader, num_examples)
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, valid_loader, num_examples

    def _load_json(self, file_path, client_id):
        IMAGE_SIZE = 28
        """
        Loads client-specific data from a LEAF JSON file.

        Args:
            file_path (str): Path to the JSON file.
            client_id (str): Client identifier.

        Returns:
            dict: Dictionary containing 'x' (images) and 'y' (labels).
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # if client_id not in data["users"]:
        #     raise ValueError(f"Client {client_id} not found in {file_path}")
        # Iterate over all user IDs and extend the merged lists
        client_data = {'x':[],'y':[]}
        for user_id, values in data['user_data'].items():
            client_data['x'].extend(values['x'])
            client_data['y'].extend(values['y'])
        # client_data = data["user_data"]
        images = np.array(client_data["x"]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
        labels = np.array(client_data["y"])

        return {"x": images, "y": labels}

    def _create_dataset(self, images, labels):
        """
        Converts images and labels into a PyTorch Dataset.

        Args:
            images (numpy.ndarray): Images (shape: N x IMAGE_SIZE x IMAGE_SIZE).
            labels (numpy.ndarray): Labels (shape: N).

        Returns:
            FEMNISTDataset: A PyTorch Dataset instance.
        """
        return FEMNISTDataset(images, labels, transform=self.transform)



class Dataset :
    def __init__(self, dataset_name, root):
        self.dataset_name = dataset_name
        self.root = root
        # torch.serialization.add_safe_globals([Subset])

        
    def get_dataset(self):
        if self.dataset_name == 'cifar10':
            return Cifar10(self.root,self.dataset_name)
        elif self.dataset_name == 'cifar100':
            return Cifar100(self.root,self.dataset_name)
        elif self.dataset_name == 'FEMNIST':
            return FEMNIST(self.root,self.dataset_name)
        else:
            raise ValueError("Invalid dataset name")
