import torch
import netpool
import torch.nn.functional as F
import numpy as np
from math import pi

class Aggregation_algorithm:
    def __init__(self, fusion_protocol_name,dataset,gpu,arch_name):
        self.fusion_protocol_name = fusion_protocol_name
        self.dataset = dataset
        if self.dataset == 'cifar10':
            self.num_classes = 10
        elif self.dataset == 'cifar100':
            self.num_classes = 100
        elif self.dataset == 'FEMNIST':
            self.num_classes = 62
        else:
            print(f'Dataset {self.dataset} not implemented in aggregation algorithm')
        self.gpu = gpu
        self.arch_name = arch_name
    def fedavg(self,update_list):
        update_list = [u['weights'] for u in update_list]
        
        # global_model_state_dict = netpool.ResNet18(num_classes = self.num_classes).to(self.gpu).state_dict()
        global_model_state_dict = netpool.ClientModel(num_classes = self.num_classes,arch_name=self.arch_name).get_mymodel().to(self.gpu).state_dict()
        
        weight_sum_dict = {key: torch.zeros_like(value) for key, value in global_model_state_dict.items()}
        
        for i,local_dict in enumerate(update_list):
           #  print(local_dict['state_dict'])
           #  local_dict=local_dict.to(DEVICE)
            for key in global_model_state_dict.keys():
                if key in local_dict:
                    weight_sum_dict[key]= weight_sum_dict[key].float()
                    weight_sum_dict[key] +=  local_dict[key].float() # added .float()
                else:
                    raise KeyError(f"Key '{key}' not found in local model state_dict.")
        num_local_models = len(update_list)
        for key in global_model_state_dict.keys():
            weight_sum_dict[key] = weight_sum_dict[key] / num_local_models
        
        return weight_sum_dict
    
    def peer_grad(self,update_list,wf=0.5):
        "input update list each update : {state_dict,curent_iter,total_iter}"
        "compute the difference between update list [0] and [1] gradient-like"
        "return as state dict the upadte[0] updted with the computed gradient"
        wf = wf
        # updated_state_dict = netpool.ResNet18(self.num_classes).state_dict()
        updated_state_dict = netpool.ClientModel(num_classes = self.num_classes,arch_name=self.arch_name).get_mymodel().state_dict()

        model_update_list = [u['weights'] for u in update_list]
        global_model_state_dict = model_update_list[0]
        weight_diff_dict = {key: torch.zeros_like(value) for key, value in model_update_list[0].items()}
        for i,local_dict in enumerate(model_update_list[1:]):
            for key in global_model_state_dict.keys():
                if key in local_dict:
                    weight_diff_dict[key] = weight_diff_dict[key].float() +  model_update_list[0][key].float() - local_dict[key].float() 
                else:
                    raise KeyError(f"Key '{key}' not found in local model state_dict.")
            for key in global_model_state_dict.keys():
                if key in local_dict:
                    updated_state_dict[key] = model_update_list[0][key] - wf * weight_diff_dict[key]
                else:
                    raise KeyError(f"Key '{key}' not found in local model state_dict.")       
        
        return updated_state_dict

    # def peer_grad_ada(self,update_list,wfo=0.5):
    #     "input update list each update : {state_dict,curent_iter,total_iter}"
    #     "compute the difference between update list [0] and [1] gradient-like"
    #     "return as state dict the upadte[0] updted with the computed gradient"
    #     wfo = wfo
    #     # updated_state_dict = netpool.ResNet18(self.num_classes).state_dict()
    #     updated_state_dict = netpool.ClientModel(num_classes = self.num_classes,arch_name=self.arch_name).get_mymodel().state_dict()

    #     model_update_list = [u['weights'] for u in update_list]
    #     progress_list = [u['current_iter']/u['total_iters'] for u in update_list]
        
    #     global_model_state_dict = model_update_list[0]
    #     for key in model_update_list[0].keys():
    #         if key in updated_state_dict:
    #             updated_state_dict[key] = model_update_list[0][key] 
    #         else:
    #             raise KeyError(f"Key '{key}' not found in local model state_dict.")   
    #     weight_diff_dict = {key: torch.zeros_like(value) for key, value in model_update_list[0].items()}
    #     for ((i,local_dict),p) in zip(enumerate(model_update_list[1:]),progress_list[1:]):
    #         for key in global_model_state_dict.keys():
    #             if key in local_dict:
    #                 # weight_diff_dict[key] +=  (p/p+progress_list[0])*(model_update_list[0][key] - local_dict[key])
    #                 # print(f"model_update_list[0][key]: {model_update_list[0][key].dtype}, local_dict[key]: {local_dict[key].dtype}")

    #                 # weight_diff_dict[key] =  weight_diff_dict[key].float() +   (p / (p + progress_list[0])) * (model_update_list[0][key].float() - local_dict[key].float())
    #                 updated_state_dict[key] = updated_state_dict[key].float() - wfo * (p / (p + progress_list[0])) * (updated_state_dict[key].float() - local_dict[key].float())
    #             else:
    #                 raise KeyError(f"Key '{key}' not found in local model state_dict.")
    #     # for key in global_model_state_dict.keys():
    #     #     if key in local_dict:
    #     #         updated_state_dict[key] = model_update_list[0][key] - wfo * weight_diff_dict[key]
    #     #     else:
    #     #         raise KeyError(f"Key '{key}' not found in local model state_dict.")       
        
    #     return updated_state_dict
    def peer_grad_ada(self, update_list,const):
        """
        Compute gradient-like update for W0 using dynamic wfo computed from cosine similarity
        between W0 and (W0 - Wi) for each peer Wi.
        """
        wfo=1
        model_update_list = [u['weights'] for u in update_list]
        progress_list = [u['current_iter'] / u['total_iters'] for u in update_list]

        global_model_state_dict = model_update_list[0]
        base_progress = progress_list[0]

        updated_state_dict = netpool.ClientModel(num_classes=self.num_classes, arch_name=self.arch_name).get_mymodel().state_dict()

        # Copy W0 to updated_state_dict
        for key in global_model_state_dict:
            if key not in updated_state_dict:
                raise KeyError(f"Key '{key}' not in model.")
            updated_state_dict[key] = global_model_state_dict[key].clone()
        
        
        for peer_dict, p in zip(model_update_list[1:], progress_list[1:]):
            for key in global_model_state_dict:
                if key not in peer_dict:
                    raise KeyError(f"Key '{key}' not in peer model.")
                diff = global_model_state_dict[key].float() - peer_dict[key].float()
                scaled_diff = (p / (p + base_progress) if p + base_progress > 0 else 0.0) * diff
                updated_state_dict[key] = updated_state_dict[key].float() -  wfo * scaled_diff

        return updated_state_dict, [], [] 
        
    def peer_grad_ada_wfo(self, update_list,const):
        """
        Compute gradient-like update for W0 using dynamic wfo computed from cosine similarity
        between W0 and (W0 - Wi) for each peer Wi.
        """
        model_update_list = [u['weights'] for u in update_list]
        progress_list = [u['current_iter'] / u['total_iters'] for u in update_list]

        global_model_state_dict = model_update_list[0]
        base_progress = progress_list[0]

        updated_state_dict = netpool.ClientModel(num_classes=self.num_classes, arch_name=self.arch_name).get_mymodel().state_dict()

        # Copy W0 to updated_state_dict
        for key in global_model_state_dict:
            if key not in updated_state_dict:
                raise KeyError(f"Key '{key}' not in model.")
            updated_state_dict[key] = global_model_state_dict[key].clone()

        # Flatten function
        def flatten_model(model_dict):
            return torch.cat([v.view(-1).float() for v in model_dict.values()])

        flat_W0 = flatten_model(global_model_state_dict)

        # Piecewise function for wfo(cosθ) (folder femnist_1)
        # def compute_wfo(cos_theta):
        #     if cos_theta >= 0.707:
        #         return -3.414 * cos_theta + 4.414
        #     elif cos_theta >= 0.0:
        #         return 1.414 * cos_theta + 1.0
        #     elif cos_theta >= -0.707:
        #         return 1.414 * cos_theta + 1.0
        #     else:
        #         return -3.414 * cos_theta - 2.414
        
        # folder femnist_2
        # def w_theta(theta):
        #     if 0 <= theta < pi/2:
        #         return 2 - (2/pi)*theta
        #     elif pi/2 <= theta < pi:
        #         return 1 - (2/pi)*(theta - pi/2)
        #     elif pi <= theta < 3*pi/2:
        #         return (2/pi)*(theta - pi)
        #     elif 3*pi/2 <= theta <= 2*pi:
        #         return 1 + (2/pi)*(theta - 3*pi/2)
        #     else:
        #         return 0
        
        # femnist 4
        # Define the piecewise cubic polynomial wfo(θ)
        
        # def w_theta(theta):
        #     theta = np.mod(theta, 2*np.pi)
        #     if 0 <= theta <= np.pi/4:
        #         return 1.62195 * theta**2 - 0.000637 * theta + 1.0
        #     elif np.pi/4 < theta <= np.pi/2:
        #         return -1.78e-15 * theta**3 + 1.62195 * theta**2 - 5.09487 * theta + 5.001
        #     elif np.pi/2 < theta <= 3*np.pi/4:
        #         return -1.62195 * theta**2 + 5.09614 * theta - 3.003
        #     elif 3*np.pi/4 < theta <= np.pi:
        #         return 1.78e-15 * theta**3 - 1.62195 * theta**2 + 10.19037 * theta - 15.006
        #     elif np.pi < theta <= 5*np.pi/4:
        #         return -1.78e-15 * theta**3 - 1.62195 * theta**2 + 10.19165 * theta - 15.01
        #     elif 5*np.pi/4 < theta <= 3*np.pi/2:
        #         return -4.44e-15 * theta**3 - 1.62195 * theta**2 + 15.28588 * theta - 35.015
        #     elif 3*np.pi/2 < theta <= 7*np.pi/4:
        #         return -4.44e-15 * theta**3 + 1.62195 * theta**2 - 15.28715 * theta + 37.021
        #     elif 7*np.pi/4 < theta <= 2*np.pi:
        #         return 5.33e-15 * theta**3 + 1.62195 * theta**2 - 20.38138 * theta + 65.028
        #     else:
        #         return 1  # fallback, shouldn't be used

        # # femnist 5 
        # def w_theta(x):
        #     p = np.pi
        #     if 0 <= x <= p/4:
        #         return (2/p)*x + 1
        #     elif p/4 < x <= p/2:
        #         return (-2/p)*x + 2
        #     elif p/2 < x <= 3*p/4:
        #         return (-2/p)*x + 2
        #     elif 3*p/4 < x <= p:
        #         return (2/p)*x - 1
        #     elif p < x <= 5*p/4:
        #         return (-2/p)*x + 3
        #     elif 5*p/4 < x <= 3*p/2:
        #         return (2/p)*x - 2
        #     elif 3*p/2 < x <= 7*p/4:
        #         return (2/p)*x - 2
        #     elif 7*p/4 < x <= 2*p:
        #         return (-2/p)*x + 5
        #     else:
        #         return None  # Outside the domain
        
        
        # # femnist 6
        # def w_theta(x):
        #     p = np.pi
        #     if 0 <= x <= p/4:
        #         y= (2/p)*x + 1
        #     elif p/4 < x <= p/2:
        #         y= (-2/p)*x + 2
        #     elif p/2 < x <= 3*p/4:
        #         y= (-2/p)*x + 2
        #     elif 3*p/4 < x <= p:
        #         y= (2/p)*x - 1
        #     elif p < x <= 5*p/4:
        #         y= (-2/p)*x + 3
        #     elif 5*p/4 < x <= 3*p/2:
        #         y= (2/p)*x - 2
        #     elif 3*p/2 < x <= 7*p/4:
        #         y= (2/p)*x - 2
        #     elif 7*p/4 < x <= 2*p:
        #         y= (-2/p)*x + 5
        #     else:
        #         return None  # Outside the domain
            
        #     return 1.6 * (y - 1.0) + 1.0
        
        
        # femnist 7
        # def w_theta(theta):
        #     y = None
        #     if 0 <= theta <= np.pi/4:
        #         y = 1.62195 * theta**2 - 0.000637 * theta + 1.0
        #     elif np.pi/4 < theta <= np.pi/2:
        #         y = -1.78e-15 * theta**3 + 1.62195 * theta**2 - 5.09487 * theta + 5.001
        #     elif np.pi/2 < theta <= 3*np.pi/4:
        #         y = -1.62195 * theta**2 + 5.09614 * theta - 3.003
        #     elif 3*np.pi/4 < theta <= np.pi:
        #         y = 1.78e-15 * theta**3 - 1.62195 * theta**2 + 10.19037 * theta - 15.006
        #     elif np.pi < theta <= 5*np.pi/4:
        #         y = -1.78e-15 * theta**3 - 1.62195 * theta**2 + 10.19165 * theta - 15.01
        #     elif 5*np.pi/4 < theta <= 3*np.pi/2:
        #         y = -4.44e-15 * theta**3 - 1.62195 * theta**2 + 15.28588 * theta - 35.015
        #     elif 3*np.pi/2 < theta <= 7*np.pi/4:
        #         y = -4.44e-15 * theta**3 + 1.62195 * theta**2 - 15.28715 * theta + 37.021
        #     elif 7*np.pi/4 < theta <= 2*np.pi:
        #         y = 5.33e-15 * theta**3 + 1.62195 * theta**2 - 20.38138 * theta + 65.028
        #     else:
        #         y = 1  # fallback
        
        #     return 0.8 * y + 0.2
        
        
        # femnist 8
        # def w_theta(theta):
        #     y = None
        #     if 0 <= theta <= np.pi/4:
        #         y = 1.62195 * theta**2 - 0.000637 * theta + 1.0
        #     elif np.pi/4 < theta <= np.pi/2:
        #         y = -1.78e-15 * theta**3 + 1.62195 * theta**2 - 5.09487 * theta + 5.001
        #     elif np.pi/2 < theta <= 3*np.pi/4:
        #         y = -1.62195 * theta**2 + 5.09614 * theta - 3.003
        #     elif 3*np.pi/4 < theta <= np.pi:
        #         y = 1.78e-15 * theta**3 - 1.62195 * theta**2 + 10.19037 * theta - 15.006
        #     elif np.pi < theta <= 5*np.pi/4:
        #         y = -1.78e-15 * theta**3 - 1.62195 * theta**2 + 10.19165 * theta - 15.01
        #     elif 5*np.pi/4 < theta <= 3*np.pi/2:
        #         y = -4.44e-15 * theta**3 - 1.62195 * theta**2 + 15.28588 * theta - 35.015
        #     elif 3*np.pi/2 < theta <= 7*np.pi/4:
        #         y = -4.44e-15 * theta**3 + 1.62195 * theta**2 - 15.28715 * theta + 37.021
        #     elif 7*np.pi/4 < theta <= 2*np.pi:
        #         y = 5.33e-15 * theta**3 + 1.62195 * theta**2 - 20.38138 * theta + 65.028
        #     else:
        #         y = 1  # fallback

        #     return 0.7 * y + 0.3  # new scaling to range [0.3, 1.7]
        
        # femnist 9
        def w_theta(theta,const):
            y = None
            if 0 <= theta <= np.pi/4:
                y = 1.62195 * theta**2 - 0.000637 * theta + 1.0
            elif np.pi/4 < theta <= np.pi/2:
                y = -1.78e-15 * theta**3 + 1.62195 * theta**2 - 5.09487 * theta + 5.001
            elif np.pi/2 < theta <= 3*np.pi/4:
                y = -1.62195 * theta**2 + 5.09614 * theta - 3.003
            elif 3*np.pi/4 < theta <= np.pi:
                y = 1.78e-15 * theta**3 - 1.62195 * theta**2 + 10.19037 * theta - 15.006
            elif np.pi < theta <= 5*np.pi/4:
                y = -1.78e-15 * theta**3 - 1.62195 * theta**2 + 10.19165 * theta - 15.01
            elif 5*np.pi/4 < theta <= 3*np.pi/2:
                y = -4.44e-15 * theta**3 - 1.62195 * theta**2 + 15.28588 * theta - 35.015
            elif 3*np.pi/2 < theta <= 7*np.pi/4:
                y = -4.44e-15 * theta**3 + 1.62195 * theta**2 - 15.28715 * theta + 37.021
            elif 7*np.pi/4 < theta <= 2*np.pi:
                y = 5.33e-15 * theta**3 + 1.62195 * theta**2 - 20.38138 * theta + 65.028
            else:
                y = 1  # fallback
        
            # return 0.6 * y + 0.4  # scale to range [0.4, 1.6]
            return (1-const) * y + const
        
        def compute_wfo(cos_theta,const):
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            y = w_theta(theta,const)
            print(cos_theta,theta,y)
            return y,theta
        
        
        wfos=[]
        thetas=[]
        # Main update
        for peer_dict, p in zip(model_update_list[1:], progress_list[1:]):
            flat_Wi = flatten_model(peer_dict)
            delta = flat_Wi - flat_W0

            # Cosine similarity between W0 and (W0 - Wi)
            cos_theta = F.cosine_similarity(flat_W0.unsqueeze(0), delta.unsqueeze(0)).item()
            wfo_i, theta = compute_wfo(cos_theta,const)
            # wfo_i = max(0.5,compute_wfo(cos_theta))
            wfos.append(wfo_i)
            thetas.append(theta)
            # Apply update per key
            for key in global_model_state_dict:
                if key not in peer_dict:
                    raise KeyError(f"Key '{key}' not in peer model.")
                diff = global_model_state_dict[key].float() - peer_dict[key].float()
                scaled_diff = (p / (p + base_progress) if p + base_progress > 0 else 0.0) * diff
                updated_state_dict[key] = updated_state_dict[key].float() -  wfo_i * scaled_diff

        return updated_state_dict, wfos, thetas 

    def peer_grad_ada_rev(self, update_list):
        """
        Compute gradient-like update for W0 using dynamic wfo computed from cosine similarity
        between W0 and (W0 - Wi) for each peer Wi.
        """
        model_update_list = [u['weights'] for u in update_list]
        progress_list = [u['current_iter'] / u['total_iters'] for u in update_list]

        global_model_state_dict = model_update_list[0]
        base_progress = progress_list[0]

        updated_state_dict = netpool.ClientModel(num_classes=self.num_classes, arch_name=self.arch_name).get_mymodel().state_dict()

        # Copy W0 to updated_state_dict
        for key in global_model_state_dict:
            if key not in updated_state_dict:
                raise KeyError(f"Key '{key}' not in model.")
            updated_state_dict[key] = global_model_state_dict[key].clone()

        # Flatten function
        def flatten_model(model_dict):
            return torch.cat([v.view(-1).float() for v in model_dict.values()])

        flat_W0 = flatten_model(global_model_state_dict)

        # Piecewise function for wfo(cosθ)
        def compute_wfo(cos_theta):
            if cos_theta >= 0.707:
                return -3.414 * cos_theta + 4.414
            elif cos_theta >= 0.0:
                return 1.414 * cos_theta + 1.0
            elif cos_theta >= -0.707:
                return 1.414 * cos_theta + 1.0
            else:
                return -3.414 * cos_theta - 2.414
        wfos=[]
        # Main update
        for peer_dict, p in zip(model_update_list[1:], progress_list[1:]):
            flat_Wi = flatten_model(peer_dict)
            delta = flat_W0 - flat_Wi

            # Cosine similarity between W0 and (W0 - Wi)
            cos_theta = F.cosine_similarity(flat_W0.unsqueeze(0), delta.unsqueeze(0)).item()
            wfo_i = compute_wfo(cos_theta)
            wfos.append(wfo_i)
            # Apply update per key
            for key in global_model_state_dict:
                if key not in peer_dict:
                    raise KeyError(f"Key '{key}' not in peer model.")
                diff = global_model_state_dict[key].float() - peer_dict[key].float()
                scaled_diff = (p / (p + base_progress) if p + base_progress > 0 else 0.0) * diff
                updated_state_dict[key] = updated_state_dict[key].float() -  wfo_i * scaled_diff

        return updated_state_dict, wfos

    def aggregate(self,update_list,wfo,theta):
        if self.fusion_protocol_name == 'fedavg':
            return self.fedavg(update_list), wfo, theta
        elif self.fusion_protocol_name == 'peer_grad':
            return self.peer_grad(update_list,wf=wfo), wfo,theta
        elif self.fusion_protocol_name == 'peer_grad_ada_wfo':
            return self.peer_grad_ada_wfo(update_list,const=wfo)
        elif self.fusion_protocol_name == 'peer_grad_ada':
            return self.peer_grad_ada(update_list,wfo)
        elif self.fusion_protocol_name == 'peer_grad_ada_rev':
            weights, wfos = self.peer_grad_ada_rev(update_list)
            return weights, wfos, theta
        else:
            raise ValueError('Invalid fusion protocol name')
