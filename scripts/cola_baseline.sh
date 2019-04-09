# Baseline experiments on cola-analysis.
# This is part of the spring 19 ling-3340 seminar course.
# Before running the following code, be sure to set up environment,
# run download_glue_data to download cola,
# and set project directory to where you want to store the records & saved models

# random-elmo
python main.py --config_file "config/spring19_seminar/cola_elmo.conf" --overrides "exp_name = cola-elmo-baseline, run_name = random-elmo, allow_untrained_encoder_parameters = 1, allow_missing_task_map = 1, do_pretrain = 0" 

# elmo
python main.py --config_file "config/spring19_seminar/cola_elmo.conf" --overrides "exp_name = cola-elmo-baseline, run_name = elmo" 

# gpt
python main.py --config_file "config/spring19_seminar/cola_gpt.conf" --overrides "exp_name = cola-gpt-baseline, run_name = gpt" 

# bert
python main.py --config_file "config/spring19_seminar/cola_bert.conf" --overrides "exp_name = cola-bert-baseline, run_name = bert" 

# bert frozen minimal pairs
python main.py --config_file "config/spring19_seminar/colapair_bertF.conf" --override "exp_name = cola-pair, run_name = bert_frozen"

# bert tune minimal pairs
python main.py --config_file "config/spring19_seminar/colapair_bertT.conf" --override "exp_name = cola-pair, run_name = bert_tune"