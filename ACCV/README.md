## Train
Prepare training data in `data/train` directory as below:
```
  data
  └── train
      ├── video_1
            ├── hr
                    ├── hr0.png
                    ├── ...
                    └── hr30.png
            └── lr_x4_BI
                    ├── lr0.png
                    ├── ...
                    └── lr30.png
      ├── ...
      └── video_N
```

- Run on CPU:
```bash
python train.py --upscale_factor 4 --patch_size 32 --batch_size 16 --n_iters 300000
```

- Run on GPU:
```bash
python train.py --upscale_factor 4 --patch_size 32 --batch_size 16 --n_iters 300000 --gpu_mode True
```

## Test
We provide the pretrained model for 4x SR on BI degradation model. Note that we made some modifications to the original code and it should produce comparable or even better results.

- Run on CPU:
```bash
python demo_Vid4.py --video_name calendar --upscale_factor 4
```

- Run on GPU:
```bash
python demo_Vid4.py --video_name calendar --upscale_factor 4 --gpu_mode True
```

- Run on GPU (memory efficient):
```bash
python demo_Vid4.py --video_name calendar --upscale_factor 4 --gpu_mode True --chop_forward True
```

You can download [Vid4](https://pan.baidu.com/s/1q947P3mvPaOjTZ5f1kXoTg) dataset and unzip in `data/test` directory. Then you can test our network on other scenes.