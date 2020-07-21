LOG=$1
# echo $LOG
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


# jains@MININT-LMU5H9C:~$ for log in 1583898264103_223202 1583898264103_223246 1583898264103_223212 1583898264103_223231 1583898264103_223224 1583898264103_223232 1583898264103_223233 1583898264103_223203 1583898264103_223206 1583898264103_223225 1583898264103_223210 1583898264103_223193 1583898264103_223248 1583898264103_223208 1583898264103_223217 1583898264103_223199; do ./train_wall.sh $log; done