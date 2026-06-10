# Cascade-Conditioned Diffusion Model for Multi-Behavior Recommendation

This is our implementation for the paper: Cascade-Conditioned Diffusion Model for Multi-Behavior Recommendation

## Environment Settings

- torch==2.7.1
- python==3.13.4
- pandas==2.3.0

## Example to run the codes.

1. Pretrained behavior embeddings already included in `./datasets`. If you want to retrain behavior embeddings, please config the main function entry in `main.py`, use `main1()` rather than `main2()` and run `main.py` to pretrain behavior embeddings:

    ```
    python main.py
    ```

3. After getting pretrained behavior embeddings, change `main1()` to `main2()` and run `main.py` to train CCDMBR:

    ```
    python main.py
    ```

## Parameter Tuning

All the parameters are in `./config.py`

## Supplementary Experiments
We conducted significance tests between CCDMBR and the strongest baseline statistically, and ★ marked the results where CCDMBR significantly outperforms the best baseline with p-value<0.05 under t-test. The results are shown below: 
| Model | BeiBei HR@10 | BeiBei HR@20 | BeiBei NDCG@10 | BeiBei NDCG@20 | Tmall HR@10 | Tmall HR@20 | Tmall NDCG@10 | Tmall NDCG@20 | IJCAI HR@10 | IJCAI HR@20 | IJCAI NDCG@10 | IJCAI NDCG@20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRGCN** | 0.0554 | <u>0.0989</u> | 0.0268 | 0.0377 | 0.0241 | 0.0430 | 0.0125 | 0.0172 | 0.0209 | 0.0311 | 0.0112 | 0.0138 |
| **DeMBR** | <u>0.0608</u> | 0.0987 | <u>0.0306</u> | <u>0.0402</u> | <u>0.0780</u> | <u>0.0942</u> | <u>0.0526</u> | <u>0.0567</u> | <u>0.0740</u> | <u>0.0895</u> | <u>0.0495</u> | <u>0.0534</u> |
| **CCDMBR** | **0.0703**★<br>±2.7e-3 | **0.1063**★<br>±3.5e-3 | **0.0357**★<br>±1.3e-3 | **0.0451**★<br>±1.5e-3 | **0.0827**★<br>±1.6e-3 | **0.0978**★<br>±2.4e-3 | **0.0575**★<br>±8.5e-4 | **0.0613**★<br>±1e-3 | **0.0756**★<br>±1e-3 | **0.0907**★<br>±1.3e-3 | **0.0512**★<br>±7e-4 | **0.0548**★<br>±7.7e-4 |

In addition, we added finer-grained ablation experiments where each auxiliary behavior is removed individually.
The additional results are summarized below:

| Model | Tmall HR@10 | Tmall HR@20 | Tmall NDCG@10 | Tmall NDCG@20 | IJCAI HR@10 | IJCAI HR@20 | IJCAI NDCG@10 | IJCAI NDCG@20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **w/o. Beh (cart)** | 0.0774 | 0.0906 | 0.0523 | 0.0556 | 0.0607 | 0.0763 | 0.0415 | 0.0434 |
| **w/o. Beh (fav)** | 0.0645 | 0.07931 | 0.04373 | 0.04779 | 0.06408 | 0.07902 | 0.04294 | 0.0453 |
| **CCDMBR** | 0.0827 | 0.0978 | 0.0575 | 0.0613 | 0.0756 | 0.0907 | 0.0512 | 0.0548 |
