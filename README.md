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