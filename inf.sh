#!/bin/bash

#SBATCH --job-name=all_lang_cont
#SBATCH -A research
#SBATCH -p long
#SBATCH -c 10
#SBATCH -t 04-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output all_lang_cont
#SBATCH --ntasks 1

cd ~/tweet_inti


#inf1

python3 emoji_bert.py bert-base-multilingual-cased 18 0.00001 0.2 mbert_all
python3 emoji_bert.py cardiffnlp/twitter-xlm-roberta-base-sentiment 18 0.00001 0.0 xlmr_tweet_all
python3 emoji_bert.py Twitter/twhin-bert-base 18 0.00001 0.2 tweet_bert_all

#inf2
python3 emoji_bert.py distilbert-base-multilingual-cased 18 0.00001 0.2 dmbert_all



python3 emoji_vec.py cardiffnlp/twitter-xlm-roberta-base-sentiment 18 0.00001 0.0 xlmr_tweet_emvec
python3 no_emoji.py cardiffnlp/twitter-xlm-roberta-base-sentiment 18 0.00001 0.0 xlmr_tweet_no_em

#inf3
python3 no_trans.py cardiffnlp/twitter-xlm-roberta-base-sentiment 18 0.00001 0.0 xlmr_tweet_no_trans
python3 notrans_noemoji.py cardiffnlp/twitter-xlm-roberta-base-sentiment 18 0.00001 0.0 xlmr_tweet_notrans_noemo


python3 non_filter.py cardiffnlp/twitter-xlm-roberta-base-sentiment 18 0.00001 0.0 xlmr_tweet_nofilter


#inf2 
python3 emoji_bert.py cardiffnlp/twitter-xlm-roberta-base-sentiment 18 0.00001 0.0 xlmr_tweet_all
python3 emoji_bert.py cardiffnlp/twitter-xlm-roberta-base-sentiment 6 0.00001 0.0 xlmr_tweet_all


echo "COMPLIETE"
sleep 1d
