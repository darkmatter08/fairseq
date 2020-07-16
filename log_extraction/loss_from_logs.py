import numpy as np
import os
import sys

try:
    path = os.path.join(sys.argv[1], 'all_training_logs_in_one_file.csv')
    # print(path)
    result = np.genfromtxt(path, delimiter=',', names=True, dtype=None) 

    all_loss = result[result['metric'] == b'loss']['value']
    idx_min_loss = np.argmin(all_loss)
    min_loss = all_loss[idx_min_loss]

    def printarr(arr):
        print(','.join([str(i) for i in arr]))

    if 0:
        print('all_loss:::')
        print(len(all_loss))
        printarr(all_loss)

        print('min_loss, idx_of_min_loss:::')
        min_prints = [min_loss, idx_min_loss]
        printarr(min_prints)

    best_loss = np.minimum.accumulate(all_loss)
    # print('best_loss @ epoch 10,20,30,-1:::')
    best_losses = [best_loss[10], best_loss[20], best_loss[30], best_loss[-1]]
    printarr(best_losses)

except:
    print('FAILED:::', path)
