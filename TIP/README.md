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
python train.py --scale 4 --patch_size 32 --batch_size 32 --n_iters 200000
```

- Run on GPU:
```bash
python train.py --scale 4 --patch_size 32 --batch_size 32 --n_iters 200000 --gpu_mode True
```

## Test
We provide the pretrained models (2x/3x/4x SR on BI degradation model and 4x SR on BD degradation model) for evaluation on the Vid4 dataset. 

- Generate LR test images (Matlab)

	- Run data/test/generate_LR_images.m
	
- Inference

	- Run on CPU:
	```bash
	python demo_Vid4.py --degradation BI --scale 4
	```

	- Run on GPU:
	```bash
	python demo_Vid4.py --degradation BI --scale 4 --gpu_mode True
	```

	- Run on GPU (memory efficient):
	```bash
	python demo_Vid4.py --degradation BI --scale 4 --gpu_mode True --chop_forward True
	```

- Evaluation (Matlab)

	- Run evaluation.m