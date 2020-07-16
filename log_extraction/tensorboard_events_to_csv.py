#!/usr/bin/env python3
# copied from::: 
# https://gist.github.com/ptschandl/ef67bbaa93ec67aba2cab0a7af47700b

'''
This script exctracts training variables from all logs from 
tensorflow event files ("event*"), writes them to Pandas 
and finally stores in long-format to a CSV-file including
all (readable) runs of the logging directory.

The magic "5" infers there are only the following v.tags:
[lr, loss, acc, val_loss, val_acc]
'''

import tensorflow as tf
import glob
import os
import pandas as pd

DESIRED_FIELDS = ['loss', 'lr']
DESIRED_FIELDS = ['loss', 'lr', 'bleu']
# DESIRED_FIELDS = ['loss']
N_FIELDS = len(DESIRED_FIELDS)

# Get all event* runs from logging_dir subdirectories
logging_dir = './logs'
event_paths = glob.glob(os.path.join(logging_dir, "*","event*"))

all_tags_discovered = set()

# Extraction function
def sum_log(path):
    runlog = pd.DataFrame(columns=['metric', 'value'])
    # try:
    # all_tags = [e.summary.value.tag for e in tf.compat.v1.train.summary_iterator(path)]
    for e in tf.compat.v1.train.summary_iterator(path):
        for v in e.summary.value:
            if v.tag not in all_tags_discovered:
                all_tags_discovered.add(v.tag)
            if v.tag in DESIRED_FIELDS:
                r = {'metric': v.tag, 'value': v.simple_value}
                runlog = runlog.append(r, ignore_index=True)
    
    # # Dirty catch of DataLossError
    # except:
    #     print('Event file possibly corrupt: {}'.format(path))
    #     return None
    # something = [[i]*N_FIELDS for i in range(0, len(runlog)//N_FIELDS)]
    # runlog['epoch'] = [item for sublist in something for item in sublist]
    something = [[i]*N_FIELDS for i in range(0, len(runlog)//N_FIELDS + 1)]
    runlog['epoch'] = [item for sublist in something for item in sublist][:len(runlog)]  # hack around epochs issue...
    
    return runlog


# Call & append
all_log = pd.DataFrame()
for path in event_paths:
    if 'train_inner' in path:
        continue
    log = sum_log(path)
    if log is not None:
        if all_log.shape[0] == 0:
            all_log = log
        else:
            all_log = all_log.append(log)

print(all_tags_discovered)

# Inspect
print(all_log.shape)
print(all_log)
all_log.head()    
            
# Store
all_log.to_csv('all_training_logs_in_one_file.csv', index=None)



'''

import tensorflow as tf
import glob
import os
import pandas as pd


# Get all event* runs from logging_dir subdirectories
logging_dir = './logs'
event_paths = glob.glob(os.path.join(logging_dir, "*","event*"))


# Extraction function
def sum_log(path):
    runlog = pd.DataFrame(columns=['metric', 'value'])
    # try:
    for e in tf.compat.v1.train.summary_iterator(path):
        for v in e.summary.value:
            if v.tag in ['loss']:
                r = {'metric': v.tag, 'value': v.simple_value}
                runlog = runlog.append(r, ignore_index=True)
    
    # # Dirty catch of DataLossError
    # except:
    #     print('Event file possibly corrupt: {}'.format(path))
    #     return None
    something = [[i]*1 for i in range(0, len(runlog)//1)]
    runlog['epoch'] = [item for sublist in something for item in sublist]
    
    return runlog


# Call & append
all_log = pd.DataFrame()
for path in event_paths:
    log = sum_log(path)
    if log is not None:
        if all_log.shape[0] == 0:
            all_log = log
        else:
            all_log = all_log.append(log)


# Inspect
print(all_log.shape)
all_log.head()    
            
# Store
all_log.to_csv('all_training_logs_in_one_file.csv', index=None)
'''
