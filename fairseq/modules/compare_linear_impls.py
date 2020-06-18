# run as `python -O -W ignore compare_linear_impls.py`

import torch

from jain_modules import Linear_meProp, LinearCRS, LinearShawn

if 0: 
    k = 256
    batch = 2048
    input_dim = 512
    output_dim = 1024

    n_steps = 10000

    for layer in (
        torch.nn.Linear(input_dim, output_dim),
        Linear_meProp(input_dim, output_dim, k=k, unified=True),
        LinearShawn(input_dim, output_dim, k=k),
        LinearCRS(input_dim, output_dim, k=k, strategy='det_top_k'),
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

        torch.cuda.synchronize()
        start.record()
        for i in range(n_steps):
            _ = layer(random_input)
        end.record()
        # end.synchronize()
        torch.cuda.synchronize()

        # time in ms -- https://pytorch.org/docs/master/cuda.html#torch.cuda.Event.elapsed_time
        total_forward_time = start.elapsed_time(end)

        print('layer_type {} total_forward_time(ms) {} mean_forward_time(ms) {}'.format(layer, total_forward_time, total_forward_time / n_steps))


    '''
    root@15d26b441bfd:~/code/fairseq/modules#
    python -O -W ignore compare_linear_impls.py python -O -W ignore compare_linear_impls.p python -O -W ignore compare_linear_impls.py python -O -W ignore compare_linear_impls.py python -O -W ignore compare_linear_impls.py
    layer_type Linear(in_features=512, out_features=1024, bias=True) total_forward_time(ms) 4171.64892578125 mean_forward_time(ms) 0.417164892578125
    layer_type Linear_meProp (512 -> 1024 <- unifiedk=256) total_forward_time(ms) 3948.5595703125 mean_forward_time(ms) 0.39485595703125
    layer_type LinearShawn (512 -> 1024 <- shawnunified256) total_forward_time(ms) 3995.633056640625 mean_forward_time(ms) 0.3995633056640625
    layer_type LinearCRS (512 -> 1024 <- CRS strategy=det_top_k, k=256) total_forward_time(ms) 4531.474609375 mean_forward_time(ms) 0.4531474609375
    layer_type LinearCRS (512 -> 1024 <- CRS strategy=first_k, k=256) total_forward_time(ms) 3720.227783203125 mean_forward_time(ms) 0.3720227783203125

    layer_type Linear(in_features=512, out_features=1024, bias=True) total_forward_time(ms) 4168.1376953125 mean_forward_time(ms) 0.41681376953125
    layer_type Linear_meProp (512 -> 1024 <- unifiedk=256) total_forward_time(ms) 3948.574462890625 mean_forward_time(ms) 0.3948574462890625
    layer_type LinearShawn (512 -> 1024 <- shawnunified256) total_forward_time(ms) 3989.570068359375 mean_forward_time(ms) 0.3989570068359375
    layer_type LinearCRS (512 -> 1024 <- CRS strategy=det_top_k, k=256) total_forward_time(ms) 4509.60888671875 mean_forward_time(ms) 0.450960888671875
    layer_type LinearCRS (512 -> 1024 <- CRS strategy=first_k, k=256) total_forward_time(ms) 3750.550537109375 mean_forward_time(ms) 0.3750550537109375

    layer_type Linear(in_features=512, out_features=1024, bias=True) total_forward_time(ms) 4159.78173828125 mean_forward_time(ms) 0.415978173828125
    layer_type Linear_meProp (512 -> 1024 <- unifiedk=256) total_forward_time(ms) 3945.65478515625 mean_forward_time(ms) 0.394565478515625
    layer_type LinearShawn (512 -> 1024 <- shawnunified256) total_forward_time(ms) 3984.50146484375 mean_forward_time(ms) 0.398450146484375
    layer_type LinearCRS (512 -> 1024 <- CRS strategy=det_top_k, k=256) total_forward_time(ms) 4529.640625 mean_forward_time(ms) 0.4529640625
    layer_type LinearCRS (512 -> 1024 <- CRS strategy=first_k, k=256) total_forward_time(ms) 3693.35498046875 mean_forward_time(ms) 0.369335498046875

    layer_type Linear(in_features=512, out_features=1024, bias=True) total_forward_time(ms) 4161.73095703125 mean_forward_time(ms) 0.416173095703125
    layer_type Linear_meProp (512 -> 1024 <- unifiedk=256) total_forward_time(ms) 3953.92333984375 mean_forward_time(ms) 0.395392333984375
    layer_type LinearShawn (512 -> 1024 <- shawnunified256) total_forward_time(ms) 3977.74609375 mean_forward_time(ms) 0.397774609375
    layer_type LinearCRS (512 -> 1024 <- CRS strategy=det_top_k, k=256) total_forward_time(ms) 4519.92578125 mean_forward_time(ms) 0.451992578125
    layer_type LinearCRS (512 -> 1024 <- CRS strategy=first_k, k=256) total_forward_time(ms) 3732.649169921875 mean_forward_time(ms) 0.3732649169921875

    layer_type Linear(in_features=512, out_features=1024, bias=True) total_forward_time(ms) 4179.4619140625 mean_forward_time(ms) 0.41794619140625
    layer_type Linear_meProp (512 -> 1024 <- unifiedk=256) total_forward_time(ms) 3957.28076171875 mean_forward_time(ms) 0.395728076171875
    layer_type LinearShawn (512 -> 1024 <- shawnunified256) total_forward_time(ms) 3993.972412109375 mean_forward_time(ms) 0.3993972412109375
    layer_type LinearCRS (512 -> 1024 <- CRS strategy=det_top_k, k=256) total_forward_time(ms) 4570.45263671875 mean_forward_time(ms) 0.457045263671875
    layer_type LinearCRS (512 -> 1024 <- CRS strategy=first_k, k=256) total_forward_time(ms) 3737.5556640625 mean_forward_time(ms) 0.37375556640625
    '''

n_trials = 10000
batch_sizes = [2**i for i in range(12)]
dim_sizes = [2**i for i in range(12)]

print('batch, input_dim, output_dim, total_forward_time(ms), mean_forward_time(ms)')
for batch in batch_sizes:
    for input_dim in dim_sizes:
        for output_dim in dim_sizes:
            layer = torch.nn.Linear(input_dim, output_dim)
            layer.cuda()
            # Generate a random matrix of size (batch, input_dim)
            random_input = torch.rand(batch, input_dim)
            random_input = random_input.cuda()
            # do forward with layer n_trials times
            # and time it

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for i in range(n_trials):
                _ = layer(random_input)
            end.record()
            # end.synchronize()
            torch.cuda.synchronize()

            # time in ms -- https://pytorch.org/docs/master/cuda.html#torch.cuda.Event.elapsed_time
            total_forward_time = start.elapsed_time(end)

            print('{},{},{},{},{}'.format(batch, input_dim, output_dim, total_forward_time, total_forward_time / n_trials))
