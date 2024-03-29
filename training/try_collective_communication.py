import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

class Network(nn.Module):
    def __init__(self, input_size=4, output_size=1):
        super(Network, self).__init__()
        #self.fc1 = nn.Linear(input_size, input_size+1)
        #self.fc2 = nn.Linear(input_size+1, output_size)
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        #x = self.fc1(x)
        #out =  self.fc2(x)
        out = self.fc(x)
        return out

def train(rank, world_size):
    # Initialize the process group for DDP
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:12345', rank=rank, world_size=world_size)
    
    input_data = torch.rand((4,2)).to(rank)
    target = torch.rand(input_data.shape[0]).unsqueeze(1).to(rank)
    input_data = input_data[rank]
    target = target[rank]
    
    print(f"Before gather, Data= {input_data}")
    
    # Code for gathering input_data from all processes to process 0
    group = dist.new_group([i for i in range(world_size)])
    # dist.gather(input_data, gather_list=None, dst=0, group=group) not supported by process group nccl https://github.com/pytorch/pytorch/issues/55893#issuecomment-1022727055
    tensor_list = [torch.empty_like(input_data) for _ in range(world_size)]
    dist.all_gather(tensor_list, input_data, group)
    input_data = torch.cat(tensor_list, dim=0).reshape(world_size, -1)
    
    print(f"After gather, data= \n{input_data}")
    input_data = torch.mean(input_data, dim=0)
    
    # Create local model
    local_model = Network(input_size=2)
    local_model = local_model.to(rank)
    local_model = DDP(local_model, device_ids=[rank])
    
    prediction = local_model(input_data)
    
    loss_f = nn.MSELoss()
    loss = loss_f(prediction, target)
    loss = loss * rank
    loss.backward()
    
    print("Data= ", input_data)
    print("Target= ", target)
    print("Loss= ", loss)
    #print(f"rank: {rank}","net = ", local_model)
    for name, param in local_model.named_parameters():
        #if param.requires_grad and param.grad is not None:
            print(f"Rank: {rank}, Parameter: {name}, Value: {param.data}, Gradient: {param.grad}")
            
    dist.destroy_process_group()
    
if __name__ == '__main__':
    # Spawn multiple processes for DDP training
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size)
    
    

        
"""class Loss_Func(nn.Module):
    def __init__(self):
        super(Loss_Func, self).__init__()
    
    def forward(self, predictions, targets):
        loss = torch.mean(targets - predictions)
        # find rank of current device 
        # weight the loss with the rank (to make gradient = rank)
        return loss"""
        