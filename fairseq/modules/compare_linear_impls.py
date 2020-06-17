import torch

from jain_modules import Linear_meProp, LinearCRS, LinearShawn

k = 256
batch=64
input_dim = 512
output_dim = 1024

n_steps = 10000

for layer in (
    Linear_meProp(input_dim, output_dim, k=k, unified=True),
    LinearCRS(input_dim, output_dim, k=k, strategy='det_top_k'),
    LinearShawn(input_dim, output_dim, k=k),
    torch.nn.Linear(input_dim, output_dim),
    LinearCRS(input_dim, output_dim, k=k, strategy='first_k'),
):
    layer.cuda()
    # Generate a random matrix of size (batch, input_dim)
    random_input = torch.rand(batch, input_dim)
    random_input = random_input.cuda()
    # do forward with layer n_steps times
    # and time it

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(n_steps):
        _ = layer(random_input)
    end.record()
    end.synchronize()

    # time in ms -- https://pytorch.org/docs/master/cuda.html#torch.cuda.Event.elapsed_time
    total_forward_time = start.elapsed_time(end)

    print('layer_type {} total_forward_time(ms) {} mean_forward_time(ms) {}'.format(layer, total_forward_time, total_forward_time / n_steps))


'''
layer_type Linear_meProp (512 -> 1024 <- unifiedk=256) total_forward_time(ms) 1268.2677001953125 mean_forward_time(ms) 0.12682677001953124
layer_type LinearCRS (512 -> 1024 <- CRS strategy=det_top_k, k=256) total_forward_time(ms) 2584.95263671875 mean_forward_time(ms) 0.258495263671875
layer_type LinearShawn (512 -> 1024 <- shawnunified256) total_forward_time(ms) 907.5820922851562 mean_forward_time(ms) 0.09075820922851563
layer_type Linear(in_features=512, out_features=1024, bias=True) total_forward_time(ms) 425.53399658203125 mean_forward_time(ms) 0.04255339965820312
layer_type LinearCRS (512 -> 1024 <- CRS strategy=first_k, k=256) total_forward_time(ms) 2705.900634765625 mean_forward_time(ms) 0.2705900634765625
'''
