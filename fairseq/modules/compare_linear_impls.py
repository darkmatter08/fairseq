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
        LinearCRS(input_dim, output_dim, k=k, strategy='single_norm'),
        LinearCRS(input_dim, output_dim, k=k, strategy='random'),
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
$ python -O -W ignore compare_linear_impls.py && python -O -W ignore compare_linear_impls.py && python -O -W ignore compare_linear_impls.py && python -O -W ignore compare_linear_impls.py && python -O -W ignore compare_linear_impls.py
FULL_DW_EXPERIMENT=False
layer_type Linear(in_features=512, out_features=1024, bias=True) total_forward_time(ms) 4017.324951171875 mean_forward_time(ms) 0.4017324951171875
layer_type Linear_meProp (512 -> 1024 <- unified k=256) total_forward_time(ms) 3959.61474609375 mean_forward_time(ms) 0.395961474609375
layer_type LinearShawn (512 -> 1024 <- shawnunified256) total_forward_time(ms) 3994.381103515625 mean_forward_time(ms) 0.3994381103515625
layer_type LinearCRS (512 -> 1024 <- CRS strategy=det_top_k, k=256) total_forward_time(ms) 4531.86962890625 mean_forward_time(ms) 0.453186962890625
layer_type LinearCRS (512 -> 1024 <- CRS strategy=first_k, k=256) total_forward_time(ms) 3666.534912109375 mean_forward_time(ms) 0.3666534912109375
layer_type LinearCRS (512 -> 1024 <- CRS strategy=single_norm, k=256) total_forward_time(ms) 4002.515869140625 mean_forward_time(ms) 0.4002515869140625
layer_type LinearCRS (512 -> 1024 <- CRS strategy=random, k=256) total_forward_time(ms) 3857.185302734375 mean_forward_time(ms) 0.3857185302734375
FULL_DW_EXPERIMENT=False
layer_type Linear(in_features=512, out_features=1024, bias=True) total_forward_time(ms) 3997.516357421875 mean_forward_time(ms) 0.3997516357421875
layer_type Linear_meProp (512 -> 1024 <- unified k=256) total_forward_time(ms) 3968.75048828125 mean_forward_time(ms) 0.396875048828125
layer_type LinearShawn (512 -> 1024 <- shawnunified256) total_forward_time(ms) 3988.140380859375 mean_forward_time(ms) 0.3988140380859375
layer_type LinearCRS (512 -> 1024 <- CRS strategy=det_top_k, k=256) total_forward_time(ms) 4527.595703125 mean_forward_time(ms) 0.4527595703125
layer_type LinearCRS (512 -> 1024 <- CRS strategy=first_k, k=256) total_forward_time(ms) 3591.301513671875 mean_forward_time(ms) 0.3591301513671875
layer_type LinearCRS (512 -> 1024 <- CRS strategy=single_norm, k=256) total_forward_time(ms) 4019.962646484375 mean_forward_time(ms) 0.4019962646484375
layer_type LinearCRS (512 -> 1024 <- CRS strategy=random, k=256) total_forward_time(ms) 3706.046630859375 mean_forward_time(ms) 0.3706046630859375
FULL_DW_EXPERIMENT=False
layer_type Linear(in_features=512, out_features=1024, bias=True) total_forward_time(ms) 4018.667724609375 mean_forward_time(ms) 0.4018667724609375
layer_type Linear_meProp (512 -> 1024 <- unified k=256) total_forward_time(ms) 3953.080810546875 mean_forward_time(ms) 0.3953080810546875
layer_type LinearShawn (512 -> 1024 <- shawnunified256) total_forward_time(ms) 3994.32861328125 mean_forward_time(ms) 0.399432861328125
layer_type LinearCRS (512 -> 1024 <- CRS strategy=det_top_k, k=256) total_forward_time(ms) 4537.98388671875 mean_forward_time(ms) 0.453798388671875
layer_type LinearCRS (512 -> 1024 <- CRS strategy=first_k, k=256) total_forward_time(ms) 3540.82568359375 mean_forward_time(ms) 0.354082568359375
layer_type LinearCRS (512 -> 1024 <- CRS strategy=single_norm, k=256) total_forward_time(ms) 3987.59228515625 mean_forward_time(ms) 0.398759228515625
layer_type LinearCRS (512 -> 1024 <- CRS strategy=random, k=256) total_forward_time(ms) 3648.0478515625 mean_forward_time(ms) 0.36480478515625
FULL_DW_EXPERIMENT=False
layer_type Linear(in_features=512, out_features=1024, bias=True) total_forward_time(ms) 3988.84619140625 mean_forward_time(ms) 0.398884619140625
layer_type Linear_meProp (512 -> 1024 <- unified k=256) total_forward_time(ms) 3945.65625 mean_forward_time(ms) 0.394565625
layer_type LinearShawn (512 -> 1024 <- shawnunified256) total_forward_time(ms) 3968.849853515625 mean_forward_time(ms) 0.3968849853515625
layer_type LinearCRS (512 -> 1024 <- CRS strategy=det_top_k, k=256) total_forward_time(ms) 4529.6025390625 mean_forward_time(ms) 0.45296025390625
layer_type LinearCRS (512 -> 1024 <- CRS strategy=first_k, k=256) total_forward_time(ms) 3648.54296875 mean_forward_time(ms) 0.364854296875
layer_type LinearCRS (512 -> 1024 <- CRS strategy=single_norm, k=256) total_forward_time(ms) 4009.1650390625 mean_forward_time(ms) 0.40091650390625
layer_type LinearCRS (512 -> 1024 <- CRS strategy=random, k=256) total_forward_time(ms) 3643.500732421875 mean_forward_time(ms) 0.3643500732421875
FULL_DW_EXPERIMENT=False
layer_type Linear(in_features=512, out_features=1024, bias=True) total_forward_time(ms) 4015.52001953125 mean_forward_time(ms) 0.401552001953125
layer_type Linear_meProp (512 -> 1024 <- unified k=256) total_forward_time(ms) 3954.179931640625 mean_forward_time(ms) 0.3954179931640625
layer_type LinearShawn (512 -> 1024 <- shawnunified256) total_forward_time(ms) 3992.530517578125 mean_forward_time(ms) 0.3992530517578125
layer_type LinearCRS (512 -> 1024 <- CRS strategy=det_top_k, k=256) total_forward_time(ms) 4524.02685546875 mean_forward_time(ms) 0.452402685546875
layer_type LinearCRS (512 -> 1024 <- CRS strategy=first_k, k=256) total_forward_time(ms) 3854.040283203125 mean_forward_time(ms) 0.3854040283203125
layer_type LinearCRS (512 -> 1024 <- CRS strategy=single_norm, k=256) total_forward_time(ms) 4010.0 mean_forward_time(ms) 0.401
layer_type LinearCRS (512 -> 1024 <- CRS strategy=random, k=256) total_forward_time(ms) 3957.444091796875 mean_forward_time(ms) 0.3957444091796875
    '''

if 1:
    n_trials = 10000
    batch_sizes = [2**i for i in range(12)]
    dim_sizes = [2**i for i in range(6, 12)]

    print('batch, input_dim, output_dim, total_forward_time(ms), mean_forward_time(ms)')
    for batch in batch_sizes:
        for input_dim in dim_sizes:
            for output_dim in dim_sizes:
                # layer = torch.nn.Linear(input_dim, output_dim)
                layer = LinearCRS(input_dim, output_dim, k=64, strategy='det_top_k')
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
