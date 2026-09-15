import random
class ADM:
    def __init__(self,prob,type_='uniform') -> None:
        self.prob = prob
        self.type = type_

    def adm_decision(self):
        "return True with probability prob else return False"
        if self.type == 'uniform':
            return random.random() < self.prob
        else:
            raise ValueError("Invalid type")
    
    def chose_peer(self,available_ids):
        "return a random peer from available_ids"
        return random.choice(available_ids)
    
    def chose_multiple_peers(self,available_ids,n_peers):
        if n_peers > len(available_ids):
            raise ValueError("Wrong topology, too many peers")
        return random.sample(available_ids,n_peers)
