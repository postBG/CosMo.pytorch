# CoSMo.pytorch

Official Implementation of **[CoSMo: Content-Style Modulation for Image Retrieval with Text Feedback](https://openaccess.thecvf.com/content/CVPR2021/html/Lee_CoSMo_Content-Style_Modulation_for_Image_Retrieval_With_Text_Feedback_CVPR_2021_paper.html)**, Seungmin Lee*, Dongwan Kim*, Bohyung Han. *(*denotes equal contribution)*

Presented at [CVPR2021](http://cvpr2021.thecvf.com/)

![fig](readme_figs/cosmo_fig.png)

## :gear: Setup
Python: python3.7

### :package: Install required packages

Install torch and torchvision via following command (CUDA10)

```bash
pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other packages
```bash
pip install -r requirements.txt
```

### :open_file_folder: Dataset
Download the FashionIQ dataset by following the instructions on this [link](https://github.com/XiaoxiaoGuo/fashion-iq). 

We have set the default path for FashionIQ datasets in [data/fashionIQ.py](data/fashionIQ.py) as `_DEFAULT_FASHION_IQ_DATASET_ROOT = '/data/image_retrieval/fashionIQ'`. You can change this path to wherever you plan on storing the dataset.


### :books: Vocabulary file
Open up a python console and run the following lines to download NLTK punkt:
```python
import nltk
nltk.download('punkt')
```

Then, open up a Jupyter notebook and run [jupyter_files/how_to_create_fashion_iq_vocab.ipynb](jupyter_files/how_to_create_fashion_iq_vocab.ipynb). As with the dataset, the default path is set in [data/fashionIQ.py](data/fashionIQ.py).

### :chart_with_upwards_trend: Weights & Biases
We use [Weights and Biases](https://wandb.ai/) to log our experiments.

If you already have a Weights & Biases account, head over to `configs/FashionIQ_trans_g2_res50_config.json` and fill out your `wandb_account_name`. You can also change the default at `options/command_line.py`.

If you do not have a Weights & Biases account, you can either create one or change the code and logging functions to your liking.

## :running_man: Run

You can run the code by the following command:
```bash
python main.py --config_path=configs/FashionIQ_trans_g2_res50_config.json --experiment_description=test_cosmo_fashionIQDress --device_idx=0,1,2,3
```

Note that you do not need to assign `--device_idx` if you have already specified `CUDA_VISIBLE_DEVICES=0,1,2,3` in your terminal. 

We run on 4 12GB GPUs, and the main gpu `gpu:0` uses around 4GB of VRAM.


## 	:scroll: Citation
If you use our code, please cite our work:
```
@InProceedings{CoSMo2021_CVPR,
    author    = {Lee, Seungmin and Kim, Dongwan and Han, Bohyung},
    title     = {CoSMo: Content-Style Modulation for Image Retrieval With Text Feedback},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {802-812}
}
```
