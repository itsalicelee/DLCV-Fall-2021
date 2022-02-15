# DLCV Final Project ( Food-classification-Challenge )

# How to run this code? (for TAs)

    # Download all our models and unzip (~5GB)
    bash get_checkpoints.sh
    
    # Train our ensemble models (22, 33, 38, 40), it may take long long time...
    # You can skip this!
    # bash train.sh <top food data folder>
    bash train.sh <top food data folder> 
    
    # Generate outputs our ensemble models with test time augmentation on 4 test types.
    # However it may takes 12 hours up, we have already save these results as .mat files.
    # You can also skip this!
    # bash test_TTA.sh <top food data folder> <ensemble models folder>
    bash test_TTA.sh <top food data folder> checkpoints
    
    # Do the Ensembles and generate for 4 logs for kaggle in the folder "checkpoints"
    # bash test_ensemble.sh <top food data folder> <ensemble models folder>
    bash test_ensemble.sh <top food data folder> checkpoints

## Train

    python3 train_template.py 
## BBN Train [ViT/ResNet/SWIN]

    python3 train_template_BBN.py -mode [UNIFORM/BALANCED]

[2020 CVPR] BBN is UNIFORM,our proposed model is BALANCED
## Long-Tail Train

    python3 train_template_LT.py -LT_EXP RESAMPLE/REWEIGHT/TODO
|LT_EXP  |Feature|
|-----   |--------|
|LDAM(default)|LDAM Loss[2019NIPS]|
|RESAMPLE|Balanced DataLoader with CrossEntropy|
|REVERSE|Reversed DataLoader with CrossEntropy|
|TODO|...|

Ref: https://github.com/robotframework/RIDE.git
## Test
    
    python3 test_template.py
## Test with TTA module
    
    python3 test_template_TTA.py -mode TEST/VALID
#####  Transforms
  
| Transform      | Parameters                | Values                            |
|----------------|:-------------------------:|:---------------------------------:|
| HorizontalFlip(good in our task) | -                         | -                                 |
| VerticalFlip(bad in our task QQ)   | -                         | -                                 |
| Rotate90(bad in our task QQ)       | angles                    | List\[0, 90, 180, 270]            |
| Scale          | scales<br>interpolation   | List\[float]<br>"nearest"/"linear"|
| Resize         | sizes<br>original_size<br>interpolation   | List\[Tuple\[int, int]]<br>Tuple\[int,int]<br>"nearest"/"linear"|
| Add            | values                    | List\[float]                      |
| Multiply       | factors                   | List\[float]                      |
| FiveCrops(bad in our task QQ)       | crop_height<br>crop_width | int<br>int                        |
 
#####  Aliases

  - flip_transform (horizontal + vertical flips)
  - hflip_transform (horizontal flip)
  - d4_transform (flips + rotation 0, 90, 180, 270)
  - multiscale_transform (scale transform, take scales as input parameter)
  - five_crop_transform (corner crops + center crop)
  - ten_crop_transform (five crops + five crops on horizontal flip)
  
#####  Merge modes
 - mean
 - gmean (geometric mean)
 - sum
 - max
 - min
 - tsharpen ([temperature sharpen](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/107716#latest-624046) with t=0.5)
 
More details refer to this repo,please.
Ref: https://github.com/qubvel/ttach
## Download our all models
- Download model from [here](https://drive.google.com/drive/u/3/folders/1XuJa60KacC_cbu-2Xphb3m0rAK2W4AGj)
 
## Load Swin model 
- Download model from [here](https://drive.google.com/file/d/1HFuSt0OEQzbMC65E4GmRxLlxelPL1DRT/view?usp=sharing)
- Put the file **swin_large_patch4_window12_384_22kto1k.pth** under ./model_zoo/swin/
- Remember to set the img_size to 384 for the model
- Download the **fine-tuned model (reversed sampler & gradaccum 16)** [here](https://drive.google.com/file/d/1Ee_rOaq4OpNFndOxRDoN195M87BpE6JE/view?usp=sharing)

## Load ResNeSt50/ResNeSt269 model 
- Download model from [here](https://drive.google.com/drive/u/3/folders/1XuJa60KacC_cbu-2Xphb3m0rAK2W4AGj)
- Put the file **resnest50_v1.pth** / **resnest269_v1.pth** under ./model_zoo/pytorch_resnest/
- Remember to set the img_size to 224 for the resnest50 model
- Remember to set the img_size to 320 for the resnest269 model
- Remember to pip install fvcore

## Automatic Submission to Kaggle

    export KAGGLE_USERNAME=datadinosaur
    export KAGGLE_KEY=xxxxxxxxxxxxxx
    bash test_kaggle.sh $1 $2 ($1:model_path(e.g., baseline/ ) $2:commit message)
## File structure
```
final-project-challenge-3-no_qq_no_life/
│
├── train_template.py - main script to start training
├── test_template.py - evaluation of trained model
├── test_template_TTA.py - evaluation of trained model with TTA module
├── base/ 
│   ├── dataset.py
│   ├── trainer.py
│   └── tester.py
└── model_zoo/ 
    ├── swin/*
    ├── vgg16.py
    ├── BBN/* 
    └── pytorch_pretrained_vit/* 
```
# Usage
To start working on this final project, you should clone this repository into your local machine by using the following command:

    git clone https://github.com/DLCV-Fall-2021/final-project-challenge-3-<team_name>.git
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://drive.google.com/drive/folders/13PQuQv4dllmdlA7lJNiLDiZ7gOxge2oJ?usp=sharing) to view the slides of Final Project - Food image classification. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `food_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1IYWPK8h9FWyo0p4-SCAatLGy0l5omQaw/view?usp=sharing) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> 1. Please do not disclose the dataset! Also, do not upload your get_dataset.sh to your (public) Github.
> 2. You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `food_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion

