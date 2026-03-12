# EfficientPIE
The code of the EfficientPIE is coming soon (IJCAI25).

## Setup Instructions for JAAD Dataset

### 1. Download Data and Annotations
1. **Download JAAD Videos:**
   ```bash
   wget http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip
   unzip JAAD_clips.zip -d JAAD_videos
   ```
2. **Download JAAD Annotations:**
   Clone the JAAD repository for the annotation files:
   ```bash
   git clone https://github.com/ykotseruba/JAAD.git
   ```

### 2. Organize Directory Structure
Move the `JAAD_clips` directory containing the video clips into the cloned `JAAD` repository folder so the dataset loader scripts can locate them:
```bash
mv JAAD_videos/JAAD_clips JAAD/
```

### 3. Extract Video Frames
The training pipeline requires the `.mp4` video clips to be extracted into image sequences inside `JAAD/images`.
To do this quickly, run the custom parallel video extraction python script:
```bash
python extract_images_parallel.py
```

## Summary of Code Fixes

To get the training pipeline to run properly on MacOS, the following code fixes were implemented:

1. **Updated Default Data Path:**
   Changed the `--data-path` default parameter in `train_EfficientPIE_JAAD.py` from the hardcoded path (`/home/yphe/FangQu_temporary/JAAD`) to point to the local `./JAAD` directory.
2. **Fixed `ModuleNotFoundError` in Dataset:**
   Removed an unused import statement (`from utils.adaptive_selection import adaptive_selection`) in `utils/my_dataset.py` that was crashing the dataloader initialization.
3. **Restored the `EfficientPIE` Model:**
   - The model file was missing, so `models/EfficientPIE_backup.py` was restored to `models/EfficientPIE.py`.
   - Fixed an `ImportError` in `models/EfficientPIE.py` by removing the missing `AdaptiveEncoder` import from `models/common.py`, which had been removed or deprecated.

## Running the Model
Once the frames are extracted and the environment dependencies are installed, you can start the training process. 

For example, to run the script for 5 epochs with a batch size of 16:
```bash
python train_EfficientPIE_JAAD.py --epochs 5 --batch_size 16
```
