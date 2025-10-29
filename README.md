## StructSR++: A Training-Free Structural Enhancement Framework for Diffusion-Based Real-World Image Super-Resolution

### Dependencies and Installation

- required packages in `requirements.txt`

```
# create an environment with python >= 3.8
conda create -n structsr++ python=3.8
conda activate structsr++
pip install -r requirements.txt
```

## 🚀 Quick Inference

#### Step 1: Download the pretrained models

- Download the pretrained SD-2-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-base).
- Download the SeeSR and DAPE models from [GoogleDrive](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO?usp=drive_link) or [OneDrive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/22042244r_connect_polyu_hk/EiUmSfWRmQFNiTGJWs7rOx0BpZn2xhoKN6tXFmTSGJ4Jfw?e=RdLbvg).

You can put the models into `preset/models`.

#### Step 2: Prepare testing data

You can put the testing images in the `preset/datasets/test_datasets`.

#### Step 3: Running testing command

```
CUDA_VISIBLE_DEVICES=0 python test_seesr.py \
    --pretrained_model_path /home/algroup/lyc/SeeSR/preset/models/stable-diffusion-2-base \
    --prompt '' \
    --seesr_model_path preset/models/seesr \
    --ram_ft_path preset/models/DAPE.pth \
    --image_path preset/datasets/test_datasets \
    --output_dir preset/datasets/output \
    --start_point lr \
    --num_inference_steps 50 \
    --guidance_scale 5.5 \
    --process_size 512 \
    --lamda_psg 2 \
    --lamda_rade 2 \
    --alpha_ema 0.9 \
    --eps 0.03
```


### Gradio Demo

Please put the all pretrained models at `preset/models`, and then run the following command to interact with the gradio website.

```
CUDA_VISIBLE_DEVICES=0 python app.py
```



### License

------

This project is licensed under [MIT License](https://github.com/LYCEXE/StructSR/blob/main/LICENSE). Redistribution and use should follow this license.

### Acknowledgement

------

This project is based on [ SeeSR](https://github.com/cswry/SeeSR/tree/main), [latent-diffusion](https://github.com/CompVis/latent-diffusion), [SPADE](https://github.com/NVlabs/SPADE), [mixture-of-diffusers](https://github.com/albarji/mixture-of-diffusers), [BasicSR](https://github.com/XPixelGroup/BasicSR) and [StructSR](https://github.com/IceClear/StableSR). Thanks for their awesome work.



