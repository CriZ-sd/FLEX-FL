import torch
import os 
import ADM
from aggregation_algorithms import Aggregation_algorithm
import threading 
import time
import random
from log import Logger
import copy


class Client :
    # g_configs : {id, local_iters, total_iters, gpu}
    # train_configs : {batch_size, lr, loss, optimizer, scheduler,latency}
    # aggregation_configs : {fusion_protocol_name, wfo, wf_strategy, pending_interval}
    # log_configs : {work_dir, log_freq, save_update_freq, verbose}
    # g_buffer : queue includes pending client ids
    # peer_buffer_list : list queues (one for each client), includes the peers id when a new request arrives and client is pending 
    # model
    # dataset
    def __init__(self,g_configs,train_configs,aggregation_configs,log_configs,model,dataset,arch_name):
        self.id = g_configs['id']
        self.local_iters = g_configs['local_iters']
        self.total_iters = g_configs['total_iters']
        self.gpu = g_configs['gpu']
        self.batch_size = train_configs['batch_size']
        self.lr = train_configs['lr']
        self.valid_interval = train_configs['valid_interval']
        self.train_val = train_configs['train_val']
        self.theta = 0 
    
        #self.optimizer = train_configs['optimizer']
        #self.scheduler = train_configs['scheduler']
        self.latency = train_configs['latency']
        self.fusion_protocol_name = aggregation_configs['fusion_protocol_name']
        self.wfo = aggregation_configs['wfo']
        self.wf_strategy = aggregation_configs['wf_strategy']
        self.adm_prob = aggregation_configs['adm_prob']
        self.n_peers = aggregation_configs['n_peers']
        if aggregation_configs['sync']:
            self.sync = True
        else:
            self.sync = False

        self.work_dir = log_configs['work_dir']
        self.log_freq = log_configs['log_freq']
        self.save_update_interval = log_configs['save_update_interval']
        self.log_update_interval = log_configs['log_update_interval']

        self.verbose = log_configs['verbose']
        # self.client_list = clients_list
        self.model = model.to(self.gpu)
        self.prev_model = copy.deepcopy(self.model) 
        if train_configs['loss'] == 'crossentropy':
            self.loss = torch.nn.CrossEntropyLoss().to(self.gpu)
        else:
            raise(ValueError(f'This loss function not supported'))
        if train_configs['optimizer'] == 'sgd':
            self.momentume = train_configs['momentume']
            self.weight_decay = train_configs['weight_decay']
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,momentum=self.momentume, weight_decay=self.weight_decay)
        else:
            raise(ValueError('This optimizer is not supported'))
        if train_configs['scheduler'] == 'MultistepLR':
            milestones = train_configs['milestones']
            self.gamma = train_configs['gamma']
            self.milestones = [m/100*self.total_iters for m in milestones ]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=self.gamma)
        else:
            raise(ValueError('This scheduler is not supported'))

        self.dataset = dataset
        # t_1 = time.time()
        self.train_set, self.valid_set, self.test_set, self.num_examples = self.dataset.load_client_sets(self.id, batch_size=self.batch_size)
        # t_2 = time.time()
        # print(f'{self.id} : {t_2-t_1}')
        self.updating = False
        self.update_iter_sync = False
        self.ADM = ADM.ADM(prob=self.adm_prob,type_='uniform')
        self.current_iter = 0
        self.current_iter_sync = 0
        self.arch_name = arch_name
        self.aggregator = Aggregation_algorithm(fusion_protocol_name=self.fusion_protocol_name,dataset=self.dataset.name,gpu=self.gpu,arch_name=self.arch_name)
        self.client_ids=[]
        
        self.update_lock = threading.Condition()
        self.sync_lock = threading.Condition()
        self.clientLogger = Logger(self.work_dir,f'client_{self.id}',{'id':self.id})
        self.experimental = {'valid_loss':[],'valid_accuracy':[],'agg_test_loss':[],'agg_test_accuracy':[],\
                             'adm':[],'peer':[], 'wfo':[], 'theta':[],\
                            'train_time':[],'aggregation_time':[],'round_time':[], 'total_time':[]}
        

    def update_experimental(self,valid_loss=None,valid_accuracy=None,agg_test_loss=None,agg_test_accuracy=None,\
                            adm=None,peer=None,wfo=None,theta=None,train_time=None,aggregation_time=None,round_time=None,total_time=None):
        if valid_loss is not None:
            self.experimental['valid_loss'].append(valid_loss)
        
        if valid_accuracy is not None:
            self.experimental['valid_accuracy'].append(valid_accuracy)
        
        if agg_test_loss is not None:
            self.experimental['agg_test_loss'].append(agg_test_loss)
        
        if agg_test_accuracy is not None:
            self.experimental['agg_test_accuracy'].append(agg_test_accuracy)
        
        if adm is not None:
            self.experimental['adm'].append(adm)
        
        if peer is not None:
            self.experimental['peer'].append(peer)
        
        if wfo is not None:
            self.experimental['wfo'].append(wfo)
        
        if theta is not None:
            self.experimental['theta'].append(theta)
            
        if train_time is not None:
            self.experimental['train_time'].append(train_time)
        
        if aggregation_time is not None : 
            self.experimental['aggregation_time'].append(aggregation_time)
        
        if round_time is not None:
            self.experimental['round_time'].append(round_time)
        
        if total_time is not None:
            self.experimental['total_time'].append(total_time)

    def create_work_dir(self):
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
            self.clientLogger.write_message(f'Created dir {self.work_dir}')

    def add_other_clients(self,client_list):
        self.client_list = client_list
        self.client_ids = [client.id for client in client_list if client.id != self.id]
        if not 1 <= self.n_peers <= len(self.client_ids):
            raise ValueError("n_peers must be between 1 and the number of other clients")

    def local_train_valid_classic(self,train_loader,valid_loader):
        self.clientLogger.write_message(f'Starting training for {self.local_iters} iters')
        # local training
        start_train_time = time.time()
        self.model.train()
        round_iter_counter = 0
        
        while round_iter_counter < self.local_iters and self.current_iter < self.total_iters:
            for images, labels in train_loader:
                
                if round_iter_counter >= self.local_iters or self.current_iter >= self.total_iters:  # Stop exactly at local_iters
                    break
                    
                if self.latency > 0:
                    # print(self.latency)
                    time.sleep(self.latency)
                self.current_iter +=1 
                with self.sync_lock:
                    self.update_iter_sync = True
                self.clientLogger.write_message(f'updating iter mode on',printme=False)
                self.current_iter_sync +=1
                with self.sync_lock:
                    self.update_iter_sync = False
                    self.sync_lock.notify_all()
                self.clientLogger.write_message(f'updating iter mode off',printme=False)

                
                images = images.to(self.gpu)
                labels = labels.to(self.gpu)
                
                # self.clientLogger.write_message(f'id : {self.id} | round iters : {round_iter_counter}')

                # print(f'>>> id : {self.id} | round iters : {round_iter_counter}')
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                round_iter_counter += 1

        self.clientLogger.write_message(f'Finished training for {self.local_iters} iters')
        
        self.clientLogger.write_message(f'Starting local evaluation')
        

        end_train_time = time.time()
        self.update_experimental(train_time=(time.time(),self.current_iter,end_train_time-start_train_time))
        # print(self.current_iter,self.current_iter % self.valid_interval, (self.current_iter) % self.valid_interval )
        # if (self.current_iter % self.valid_interval == 0) or ((self.current_iter) % self.valid_interval == 0):
        if (self.current_iter) % self.valid_interval == 0:
            self.model.eval()
            # local validation
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images = images.to(self.gpu)
                    labels = labels.to(self.gpu)
                    outputs = self.model(images)
                    loss = self.loss(outputs, labels)

                    total_loss += loss.item() * images.size(0)  # Accumulate total loss
                    total_correct += (outputs.argmax(dim=1) == labels).sum().item()  # Count correct predictions
                    total_samples += images.size(0)  # Keep track of total samples

            # Compute average loss and accuracy
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples  
            self.clientLogger.write_message(f'Finished local evaluation loss : {avg_loss} | acc : {accuracy}')
            timestamp = time.time()
            self.update_experimental(valid_loss=(timestamp,self.current_iter,avg_loss),valid_accuracy=(timestamp,self.current_iter,accuracy))
    
    def local_train_valid_fedprox(self,train_loader,valid_loader):
        mu = 1 # ref sto papaer FedProx
        self.clientLogger.write_message(f'Starting training for {self.local_iters} iters')
        # local training
        start_train_time = time.time()
        prev_model_weights = {k: v.clone().detach().to(self.gpu) for k, v in self.prev_model.state_dict().items()}
        self.model.train()
        round_iter_counter = 0
        
        while round_iter_counter < self.local_iters and self.current_iter < self.total_iters:
            for images, labels in train_loader:
                if round_iter_counter >= self.local_iters or self.current_iter >= self.total_iters:  # Stop exactly at local_iters
                    break
                self.current_iter +=1 
                with self.sync_lock:
                    self.update_iter_sync = True
                self.clientLogger.write_message(f'updating iter mode on',printme=False)
                self.current_iter_sync +=1
                with self.sync_lock:
                    self.update_iter_sync = False
                    self.sync_lock.notify_all()
                self.clientLogger.write_message(f'updating iter mode off',printme=False)
                
                images = images.to(self.gpu)
                labels = labels.to(self.gpu)

                # print(f'>>> id : {self.id} | round iters : {round_iter_counter}')
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss(outputs, labels)

                # Add FedProx proximal term: ||w - prev_w||^2
                prox_loss = 0.0
                for name, param in self.model.named_parameters():
                    prox_loss += ((param - prev_model_weights[name]) ** 2).sum()
            
                loss += (mu / 2) * prox_loss  # Add FedProx penalty


                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                round_iter_counter += 1

        self.clientLogger.write_message(f'Finished training for {self.local_iters} iters')
        
        self.clientLogger.write_message(f'Starting local evaluation')
        if self.latency > 0:
            time.sleep(float(self.latency))

        end_train_time = time.time()
        self.update_experimental(train_time=(time.time(),self.current_iter,end_train_time-start_train_time))
        # if (self.current_iter % self.valid_interval == 0) or ((self.current_iter) % self.valid_interval == 0):
        if (self.current_iter) % self.valid_interval == 0:
            self.model.eval()
            # local validation
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images = images.to(self.gpu)
                    labels = labels.to(self.gpu)
                    outputs = self.model(images)
                    loss = self.loss(outputs, labels)

                    total_loss += loss.item() * images.size(0)  # Accumulate total loss
                    total_correct += (outputs.argmax(dim=1) == labels).sum().item()  # Count correct predictions
                    total_samples += images.size(0)  # Keep track of total samples

            # Compute average loss and accuracy
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples  
            self.clientLogger.write_message(f'Finished local evaluation loss : {avg_loss} | acc : {accuracy}')
            timestamp = time.time()
            self.update_experimental(valid_loss=(timestamp,self.current_iter,avg_loss),valid_accuracy=(timestamp,self.current_iter,accuracy))
    
    def local_train_valid(self,train_loader,valid_loader):
        if self.train_val == 'FedProx':
            self.clientLogger.write_message(f'FedProx')
            return self.local_train_valid_fedprox(train_loader,valid_loader)
        elif self.train_val == 'Classic':
            self.clientLogger.write_message(f'Classic')
            return self.local_train_valid_classic(train_loader,valid_loader)
        raise ValueError("train_val must be Classic or FedProx")

    def local_test(self,test_loader):
        self.clientLogger.write_message(f'Starting  testing')

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.gpu),labels.to(self.gpu)
                outputs = self.model(images)
                loss = self.loss(outputs, labels)

                total_loss += loss.item() * images.size(0)  # Accumulate loss
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()  # Count correct predictions
                total_samples += images.size(0)  # Track total samples

        # Compute average test loss and accuracy
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        self.clientLogger.write_message(f'Finished  testing loss : {avg_loss} | accuracy : {accuracy}')
        timestamp = time.time()
        self.update_experimental(agg_test_loss=(timestamp,self.current_iter,avg_loss),agg_test_accuracy=(timestamp,self.current_iter,accuracy))

    
    def update_transmitter(self):
        with self.update_lock:
            t_time = time.time()
            while self.updating:
                if ((t_time - time.time())%5 == 0):
                    self.clientLogger.write_message(f'Client {self.id} updating mode on waiting to finish')
                self.update_lock.wait()  # Wait for notification

        update_to_send = {'weights':self.model.state_dict(), 'current_iter': self.current_iter, \
                          'local_iters': self.local_iters, 'total_iters': self.total_iters}
        self.clientLogger.write_message('Update succesfully sent')
        return update_to_send


    def update_reciever(self,peer_id):
        self.clientLogger.write_message(f'Waiting to recieve update from client {peer_id}')
        update_to_receive = self.client_list[peer_id-1].update_transmitter()
        self.clientLogger.write_message(f'Succsesfully recieved update from client {peer_id}')
        
        return update_to_receive

    def check_my_iter(self):
        with self.sync_lock:
            t_time = time.time()
            while self.update_iter_sync:
                if ((time.time()-t_time)%1 == 0):
                    self.clientLogger.write_message(f'Client {self.id} updating iter mode on waiting to finish')
                self.sync_lock.wait()   
        my_iter = self.current_iter_sync
        return my_iter

    def sync_clients(self):
        synced  = False
        current_iters = []
        for client in self.client_list:
            self.clientLogger.write_message(f'Waiting to recieve current iter from client {client.id}',printme=False)
            current_iters.append(client.check_my_iter())
            self.clientLogger.write_message(f'Succsesfully recieved current iter from client {client.id}',printme=False)
        #current_iters.append(self.current_iter)
        synced = all(x == current_iters[0] for x in current_iters) if current_iters else True
        if synced == True:
            self.clientLogger.write_message(f'All clients synced with current iteration at : {current_iters[0]}')
            current_iters = []
        else:
            self.clientLogger.write_message(f'Client iters : {current_iters}')
            if self.current_iter < max(current_iters):
                synced = True
            current_iters = []
        return synced



    def local_round_async(self):
        train_loader, valid_loader,test_loader,_ = self.dataset.load_client_loaders(self.train_set,self.test_set,self.valid_set,self.num_examples,self.gpu, batch_size=self.batch_size)
        self.clientLogger.write_message(f'Data loaders ready',printme=False)
        with self.update_lock:
            self.updating = True
            self.clientLogger.write_message(f'updating mode on')
        self.prev_model = copy.deepcopy(self.model)
        self.local_train_valid(train_loader,valid_loader)
        with self.update_lock:
            self.updating = False
            self.update_lock.notify_all()
            self.clientLogger.write_message(f'updating mode off')


        "adm decision"
        adm_decision = self.ADM.adm_decision()
        self.update_experimental(adm=(time.time(),self.current_iter,adm_decision))
        self.clientLogger.write_message(f'ADM decision : {adm_decision}')
        if adm_decision == True:
            peer_id = self.ADM.chose_peer(self.client_ids)
            self.update_experimental(peer=(time.time(),self.current_iter,peer_id))
            self.clientLogger.write_message(f'peer id : {peer_id}')
            peer_update = self.update_reciever(peer_id)
            local_update = {'weights':self.model.state_dict(), 'current_iter': self.current_iter, 'local_iters': self.local_iters, 'total_iters': self.total_iters}
            "aggregate"
            agg_start = time.time()
            self.model.train()
            self.clientLogger.write_message(f'Aggregating .. ')
            new_local_update, wfos, theta = self.aggregator.aggregate([local_update,peer_update],self.wfo,self.theta)
            agg_end = time.time()
            self.clientLogger.write_message(f'Aggregation done')
            self.update_experimental(aggregation_time=(time.time(),self.current_iter,agg_end-agg_start),wfo = wfos, theta = theta)

            "load new aggregated model"

            with self.update_lock:
                self.updating = True
                self.clientLogger.write_message(f'Updating mode on')
                self.model.load_state_dict(new_local_update)
                self.clientLogger.write_message(f'Updated model loaded')
                self.updating = False
                self.update_lock.notify_all()
                self.clientLogger.write_message(f'Updating mode off')
                self.local_test(test_loader)

    def local_round_sync(self):
        train_loader, valid_loader,test_loader,_ = self.dataset.load_client_loaders(self.train_set,self.test_set,self.valid_set,self.num_examples,self.gpu, batch_size=self.batch_size)
        self.clientLogger.write_message(f'Data loaders ready',printme=False)
        
        while not self.sync_clients():
            self.clientLogger.write_message(f'Waiting for clients to be ready')

            time.sleep(5)
        
        with self.update_lock:
            self.updating = True
            self.clientLogger.write_message(f'updating mode on')
        self.prev_model = copy.deepcopy(self.model)
        self.local_train_valid(train_loader,valid_loader)
        with self.update_lock:
            self.updating = False
            self.update_lock.notify_all()
            self.clientLogger.write_message(f'updating mode off')


        "adm decision"
        adm_decision = self.ADM.adm_decision()
        self.update_experimental(adm=(time.time(),self.current_iter,adm_decision))
        self.clientLogger.write_message(f'ADM decision : {adm_decision}')
        if adm_decision == True:
            peer_id = self.ADM.chose_peer(self.client_ids)
            self.update_experimental(peer=(time.time(),self.current_iter,peer_id))
            self.clientLogger.write_message(f'peer id : {peer_id}')
            peer_update = self.update_reciever(peer_id)
            local_update = {'weights':self.model.state_dict(), 'current_iter': self.current_iter, 'local_iters': self.local_iters, 'total_iters': self.total_iters}
            "aggregate"
            agg_start = time.time()
            self.model.train()
            self.clientLogger.write_message(f'Aggregating .. ')
            new_local_update, wfos, theta = self.aggregator.aggregate([local_update,peer_update],self.wfo,self.theta)
            agg_end = time.time()
            self.clientLogger.write_message(f'Aggregation done')
            self.update_experimental(aggregation_time=(time.time(),self.current_iter,agg_end-agg_start),wfo=wfos, theta=theta)

            "load new aggregated model"

            with self.update_lock:
                self.updating = True
                self.clientLogger.write_message(f'Updating mode on')
                self.model.load_state_dict(new_local_update)
                self.clientLogger.write_message(f'Updated model loaded')
                self.updating = False
                self.update_lock.notify_all()
                self.clientLogger.write_message(f'Updating mode off')
                self.local_test(test_loader)

    def local_round_multi_topology(self):
        train_loader, valid_loader,test_loader,_ = self.dataset.load_client_loaders(self.train_set,self.test_set,self.valid_set,self.num_examples,self.gpu, batch_size=self.batch_size)
        self.clientLogger.write_message(f'Data loaders ready',printme=False)
        
        "synchronize"
        if self.sync == True:
            while not self.sync_clients():
                self.clientLogger.write_message(f'Waiting for clients to be ready')

                time.sleep(5)
        
        
        with self.update_lock:
            self.updating = True
            self.clientLogger.write_message(f'updating mode on')
        self.prev_model = copy.deepcopy(self.model)
        self.local_train_valid(train_loader,valid_loader)
        with self.update_lock:
            self.updating = False
            self.update_lock.notify_all()
            self.clientLogger.write_message(f'updating mode off')


        "adm decision"
        adm_decision = self.ADM.adm_decision()
        self.update_experimental(adm=(time.time(),self.current_iter,adm_decision))
        self.clientLogger.write_message(f'ADM decision : {adm_decision}')
        if adm_decision == True:
            peer_ids = self.ADM.chose_multiple_peers(self.client_ids,self.n_peers)
            self.update_experimental(peer=(time.time(),self.current_iter,peer_ids))
            self.clientLogger.write_message(f'peer id : {peer_ids}')
            peer_updates = [] 
            for pid in peer_ids:
                peer_updates.append(self.update_reciever(pid))
            local_update = {'weights':self.model.state_dict(), 'current_iter': self.current_iter, 'local_iters': self.local_iters, 'total_iters': self.total_iters}
            "aggregate"
            agg_start = time.time()
            self.model.train()
            self.clientLogger.write_message(f'Aggregating .. ')
            new_local_update, wfos, theta = self.aggregator.aggregate([local_update]+peer_updates,self.wfo,self.theta)
            agg_end = time.time()
            self.clientLogger.write_message(f'Aggregation done')
            self.update_experimental(aggregation_time=(time.time(),self.current_iter,agg_end-agg_start),wfo=wfos, theta = theta)

            "load new aggregated model"

            with self.update_lock:
                self.updating = True
                self.clientLogger.write_message(f'Updating mode on')
                self.model.load_state_dict(new_local_update)
                self.clientLogger.write_message(f'Updated model loaded')
                self.updating = False
                self.update_lock.notify_all()
                self.clientLogger.write_message(f'Updating mode off')
                self.local_test(test_loader)



    def local_round(self):
        if self.n_peers == 1:
            if self.sync == True:
                self.clientLogger.write_message(f'Sync')
                return self.local_round_sync()
            elif self.sync == False:
                self.clientLogger.write_message(f'Async')
                return self.local_round_async()
            else:
                raise(ValueError(f'Value {self.sync} not supported for sync variable'))
        elif self.n_peers > 1:
            self.clientLogger.write_message(f'Topology')
            return self.local_round_multi_topology()


            
    def start_client(self):
        self.clientLogger.write_experimental(self.experimental)
        
        self.clientLogger.write_message(f'Client started')
        
        
        total_start = time.time()
        self.create_work_dir()
        while self.current_iter < self.total_iters:
            self.clientLogger.write_message(f'STARTING Local Round {int(self.current_iter/self.local_iters)}')
            round_start = time.time()
            self.local_round()
            round_end = time.time()
            self.update_experimental(round_time=(time.time(),self.current_iter,round_end-round_start))
            
            self.clientLogger.write_message(f'ROUND DONE | current iter : {self.current_iter}')

            # if (self.current_iter % self.log_freq == 0):
            self.clientLogger.write_experimental(self.experimental)
            if (self.current_iter % self.save_update_interval == 0):
                self.clientLogger.save_update_log(self.model)
            if (self.current_iter % self.log_update_interval == 0):
                self.clientLogger.write_message(self.model.state_dict(),printme=False)
                        
        if self.current_iter % self.save_update_interval != 0:
            self.clientLogger.save_update_log(self.model)
        self.update_experimental(total_time=(time.time(),self.current_iter,time.time()-total_start))
        self.clientLogger.write_experimental(self.experimental)


        

        
    
    



