# SpiVG


## step
1. follow [] to get and pretreat datasets
2. train model with:
   + on SumMe:
     ```
     python train.py --cfg configs/video-summarization/SumMe/SPELL_default.yaml --split 4
     ```
   + on TVSum:
     ```
     python train.py --cfg configs/video-summarization/TVSum/SPELL_default.yaml --split 4
     ```
   + on VideoXum:
     ```
     python train_videoxum.py --cfg configs/video-summarization/VideoXum/SPELL_default.yaml --split 4
     ```
3. evaluate with:
   + on SumMe:
     ```
     python eval.py --exp_name SPELL_VS_SumMe_default --eval_type VS_max --split 4
     ```
   + on TVSum:
     ```
     python eval.py --exp_name SPELL_VS_TVSum_default --eval_type VS_avg --split 4
     ```
   + on VideoXum:
     ```
     python eval_videoxum.py --exp_name SPELL_VS_VideoXum_default --eval_type VS_avg --split 4
     ```
