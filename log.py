import logging 
import time
import torch 
import os
import json

class Logger:
    # logfile : path/file to save the logs, name: a unique name for each different logger(you can use the filename)
    def __init__(self,client_dir, name,client_info):
        logfile = f'{client_dir}/logfile'
        chkpoint_path = f'{client_dir}/checkpoints'
        experimental_path = f'{client_dir}/experimental.json'
        FORMAT = '%(asctime)s - client : %(id) - 15s %(message)s'
        if not os.path.exists(chkpoint_path):
            os.makedirs(chkpoint_path)
            print(f"Folder '{chkpoint_path}' created.\n")
            # self.Logger.write_message(f'Folder {folder_path} created.\n')
            
        else:
            print(f" Folder '{chkpoint_path}' already exists.\n")

        self.handler = logging.FileHandler(logfile)
        self.logger = logging.getLogger(name)
        self.formatter = logging.Formatter(FORMAT)
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.client_info = client_info
        self.chkpoint_path = chkpoint_path
        self.experimental_path = experimental_path
    # txt: the text that sould be logged, client_info : {clientid:'..'}
    def write_message(self,txt,printme=True):
        self.logger.warning(txt,extra=self.client_info)
        if printme:
            print(f'{time.time()} - client : {self.client_info["id"]} | {txt}')

    def save_update_log(self,model):
        # save_update(model,f'{self.chkpoint_path}/{time.time()}')
        try:
            update = {'state_dict':model.state_dict()}
        except :
            update = {'state_dict': model}
        torch.save(update,f'{self.chkpoint_path}/{time.time()}')
    "overwrite the old"
    def write_experimental(self,jsondata):
        with open(self.experimental_path,'w') as f:
            json.dump(jsondata,f)


