We use the environment below:

* Python 3.9
* CUDA 12.4
* cudnn 11.1
* pytorch 1.20
* pillow 9.5.0
* numpy 1.21.6
* scipy 1.7.3
* imageio 2.31.2


#### Font2Font

ttf转png

```sh
python font2img.py --src_font=./simhei.ttf --dst_font=./simsong.ttf --charset=CN --sample_count=500 --sample_dir=dir1 --label=0 --filter --shuffle --mode=font2font
```


### Package
将图片转为二进制格式
```sh
python package.py --dir=dir1 --save_dir=./experiment1/data --split_ratio=0.1
```

### Train

```sh
python train.py --experiment_dir=experiment1  --gpu_ids=cuda:0  --batch_size=16  --epoch=300 --sample_steps=300  --checkpoint_steps=500
```

### Infer

```sh
python infer.py --experiment_dir experiment1 --gpu_ids cuda:0 --batch_size 16 --resume 10000  --from_txt --src_font ./simhei.ttf --src_txt 大威天龙大罗法咒世尊地藏波若诸佛 --label 0
```

