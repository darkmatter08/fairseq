LOG=$1
echo $LOG
grep "train_wall " /mnt/storagerr1/sys/jobs/application_$LOG/stdout/1/stdout.txt | grep "epoch 001" | awk '{printf $(NF-3)}'
echo -n ,
grep "train_wall " /mnt/storagerr1/sys/jobs/application_$LOG/stdout/1/stdout.txt | grep "epoch 002" | awk '{printf $(NF-3)}'
echo -n ,
grep "train_wall " /mnt/storagerr1/sys/jobs/application_$LOG/stdout/1/stdout.txt | grep "epoch 003" | awk '{printf $(NF-3)}'
echo -n ,
grep "train_wall " /mnt/storagerr1/sys/jobs/application_$LOG/stdout/1/stdout.txt | grep "epoch 004" | awk '{printf $(NF-3)}'
echo -n ,
grep "train_wall " /mnt/storagerr1/sys/jobs/application_$LOG/stdout/1/stdout.txt | grep "epoch 030" | awk '{printf $(NF-3)}'
echo 
