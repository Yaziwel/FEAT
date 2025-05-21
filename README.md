# FEAT


### [ArXiv Paper](https://arxiv.org/abs/2403.11050)
### Accepted by International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2025) 

FEAT: Full-Dimensional Efficient Attention Transformer for Medical Video Generation (MICCAI 2025) (Early Accept (9%))

<video controls loop>
  <source src="./video/example.mp4" type="video/mp4">
</video>


## Setup

```cmd
git clone https://github.com/xxx
cd FEAT
conda create -n FEAT python=3.10
conda activate FEAT

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Data Preparation
**Colonoscopic**:  The dataset provided by [paper](https://ieeexplore.ieee.org/abstract/document/7442848) can be found [here](http://www.depeca.uah.es/colonoscopy_dataset/). You can directly use the [processed video data](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155167044_link_cuhk_edu_hk/ES_hCHb2XWFJgsK4hrKUnNUBx3fl6QI3yyk9ImP4AkkRVw?e=LC4DU5) by *[Endo-FM](https://github.com/openmedlab/Endo-FM)* without further data processing.

**Kvasir-Capsule**:  The dataset provided by [paper](https://www.nature.com/articles/s41597-021-00920-z) can be found [here](https://datasets.simula.no/kvasir-capsule/). You can directly use the [processed video data](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155167044_link_cuhk_edu_hk/EQhyk3_yz5pAtdpKVFU93S0BfPfTNpblPFXTHaW-BIjV-Q?e=9duP5z) by *[Endo-FM](https://github.com/openmedlab/Endo-FM)* without further data processing.

Please run [`process_data.py`](process_data.py) and [`process_list.py`](process_list.py) to get the split frames and the corresponding list at first.
```cmd
CUDA_VISIBLE_DEVICES=gpu_id python process_data.py -s ./data/Colonoscopic -t ./data/Colonoscopic_frames

CUDA_VISIBLE_DEVICES=gpu_id python process_list.py -f ./data/Colonoscopic_frames -t ./data/Colonoscopic_frames/train_128_list.txt
```

The resulted file structure is as follows.
```
├── data
│   ├── Colonoscopic
│     ├── 00001.mp4
|     ├──  ...
│   ├── Kvasir-Capsule
│     ├── 00001.mp4
|     ├──  ...
│   ├── Colonoscopic_frames
│     ├── train_128_list.txt
│     ├── 00001
│           ├── 00000.jpg
|           ├── ...
|     ├──  ...
│   ├── Kvasir-Capsule_frames
│     ├── train_128_list.txt
│     ├── 00001
│           ├── 00000.jpg
|           ├── ...
|     ├──  ...
```

## Training

You can follow the steps below to train FEAT:

```bash
bash train_scripts/col/train_col.sh
bash train_scripts/kva/train_kva.sh
bash train_scripts/col/train_col_multi.sh
bash train_scripts/kva/train_kva_multi.sh
```

## Sampling

You can follow the steps below to sample videos by using FEAT:
```bash
bash sample/col.sh
bash sample/kva.sh
```
DDP sample
```bash
bash sample/col_ddp.sh
bash sample/kva_ddp.sh
```

## Evaluation

You can follow the steps below to evaluate the model.  

```cmd
CUDA_VISIBLE_DEVICES=gpu_id python process_data.py -s /path/to/generated/video -t /path/to/video/frames
cd /path/to/stylegan-v
CUDA_VISIBLE_DEVICES=gpu_id python ./src/scripts/calc_metrics_for_dataset.py \
  --fake_data_path /path/to/video/frames \
  --real_data_path /path/to/dataset/frames 
```

## Running Other Methods

As we follow the work Endora, you can run other methods the same way as how [Endora](https://github.com/CUHK-AIM-Group/Endora) described.

## Downstream Application

As we follow the work Endora, you can run the downstream task the same way as how [Endora](https://github.com/CUHK-AIM-Group/Endora) described.

|Method|Colonoscopic |
|-----|------|
|Supervised-only | 74.5  |
|LVDM | 76.2  |
|Endora| 87.0 |
|FEAT-S (ours)| 89.9 |
|FEAT-L (ours)| 91.3 |

```
## TODO List
- [X] Release code for FEAT

## Acknowledgements
Greatly appreciate the tremendous effort for the following projects!
- [Endora](https://github.com/CUHK-AIM-Group/Endora)
- [Endo-FM](https://github.com/openmedlab/Endo-FM)
- [Latte](https://github.com/Vchitect/Latte)
- [EndoGaussian](https://github.com/yifliu3/EndoGaussian)
- [CoMatch](https://github.com/salesforce/CoMatch)
- [Stylegan-v](https://github.com/universome/stylegan-v)
```
If you find FEAT useful in your research, please consider citing:

```
@article{wang2025feat,
  author    = {Huihan Wang and Zhiwen Yang and Hui Zhang and Dan Zhao and Bingzheng Wei and Yan Xu},
  title     = {FEAT: Full-Dimensional Efficient Attention Transformer for Medical Video Generation},
  journal   = {arXiv preprint arXiv:xxxx},
  year      = {2025}
}
```

