# SpiVG: Spiking Variational Graph Representation Inference for Video Summarization

## Experment

### Initialize project files
+ download this project using git
  ```
  git clone https://github.com/AragornHorse/SpiVG.git
  ```
+ download datasets to ./data/
  + download TVsum and SumMe following the steps of
    ```
    https://github.com/IntelLabs/GraVi-T/blob/main/docs/GETTING_STARTED_VS.md
    ```
  + download VideoXum from
     ```
      https://huggingface.co/datasets/jylins/videoxum
     ```
+ If you want to download model parameters, put them in ./results/
+ the final struture should be:
  ```
  |-- SpiVG
    |-- configs
      |-- SumMe
        |-- SPELL_default.yaml
      |-- TVSum
        |-- SPELL_default.yaml
      |-- VideoXum
        |-- SPELL_default.yaml
    |-- data
      |-- annotations
        |-- SumMe
          |-- eccv16_dataset_summe_google_pool5.h5
        |-- TVSum
          |-- eccv16_dataset_tvsum_google_pool5.h5
        |-- videoxum
          |-- blip
          |-- test_videoxum.json
          |-- train_videoxum.json
          |-- val_videoxum.json
          |-- vt_clipscore
      |-- graphs
      |-- generate_temporal_graphs.py
    |-- results
      |-- SPELL_VS_SumMe_default
      |-- SPELL_VS_TVSum_default
      |-- SPELL_VS_VideXum_default
    |-- other files
  ```


## train model with:
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
