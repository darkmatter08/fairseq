LOG=$1
echo $LOG
grep "valid | epoch 010 " /mnt/storagerr1/sys/jobs/application_$LOG/stdout/1/stdout.txt | awk '{printf $NF}'
echo -n ,
grep "valid | epoch 020 " /mnt/storagerr1/sys/jobs/application_$LOG/stdout/1/stdout.txt | awk '{printf $NF}'
echo -n ,
grep "valid | epoch 030 " /mnt/storagerr1/sys/jobs/application_$LOG/stdout/1/stdout.txt | awk '{printf $NF}'
echo -n ,
grep "valid | epoch 120 " /mnt/storagerr1/sys/jobs/application_$LOG/stdout/1/stdout.txt | awk '{printf $NF}'
echo 

# grep "valid | epoch 010 " application_$LOG/stdout/1/stdout.txt && grep "valid | epoch 020 " application_$LOG/stdout/1/stdout.txt  && grep "valid | epoch 030 " application_$LOG/stdout/1/stdout.txt && grep "valid | epoch 120 " application_$LOG/stdout/1/stdout.txt


# for log in 1583898264103_223336 1583898264103_223330 1583898264103_223334 1583898264103_223326 1583898264103_223323 1583898264103_223333 1583898264103_223331 1583898264103_223324 1583898264103_223325 1583898264103_223328 1583898264103_223335 1583898264103_223327;
# do
#     ./best_loss.sh $log;
# done