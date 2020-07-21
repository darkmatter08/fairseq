JOB=$1
EXPERIMENT=$2
python /home/jains/Documents/fairseq/log_extraction/tflogs2pandas.py ~/phillytools/projects/pt_transformers/$EXPERIMENT/pt-results/application_$JOB/logs/valid --write-csv -o ~/phillytools/projects/pt_transformers/$EXPERIMENT/pt-results/application_$JOB/logs/valid

python /home/jains/Documents/fairseq/log_extraction/tflogs2pandas.py ~/phillytools/projects/pt_transformers/$EXPERIMENT/pt-results/application_$JOB/logs/train --write-csv -o ~/phillytools/projects/pt_transformers/$EXPERIMENT/pt-results/application_$JOB/logs/train

# for log in 1583898264103_223198 1583898264103_223239 1583898264103_223245 1583898264103_223200 1583898264103_223209 1583898264103_223227 1583898264103_223190 1583898264103_223237; do ./tensorboard_to_csv.sh $log 0c080845; done
