# Voice Assisted Fan Controller

## Description
This project implements a voice recognition system on a Raspberry Pi to control a ceiling fan. The system uses a custom wake word detector trained on a dataset that includes both self-recorded samples and the Mozilla Common Voice (MCV) dataset. The project also integrates the Faster Whisper library for local speech recognition on the Pi. The goal is to implement RF-based fan control, although current limitations with the Raspberry Pi 5 have delayed this feature.

## Setup Instructions

1. Ensure Python Version 3.11
Make sure you have Python version 3.11 installed on your system. You can check your Python version by runing:
```bash
python --version
```
2. Clone or Fork the Repository
Clone the repository using the following command:
```bash
git clone https://github.com/JonahCP/VoiceAssistedFanController.git
```
3. Install dependencies from the '**requirements.txt**' file
  ```bash
  pip install -r requirements.txt
  ```
4. Run the main script
  ```bash
  python main.py
  ```
5. Using a Different Model (Optional):

A pretrained model is found in '**checkpoints/model.pth**, but it's configured for my voice. If you wish to train your own model, please see the instructions below.

## Notes on training your own model
The training scripts and procedures are provided for reference, but please note:

**Data Collection**: The dataset used for training the wake word detector is not included in this repository. It was composed of self-recorded audio samples and the Mozilla Common Voice dataset. If you wish to train the model yourself, you'll need to collect or use your own dataset.

**Creating Dataset**: To create the dataset, run the following scripts in order:

- **collect_ambient_noise.py**: Collects 300 seconds of ambient audio for a noise dataset.
- **collect_positive_set.py**: Collects 100 samples of the wake word to build the positive dataset.
- **MCV_eda.py**: A Jupyter notebook used for exploratory data analysis (EDA) of the Mozilla Common Voice dataset and to create a training dataframe.

**Training Scripts**: The scripts used to train the model can be found in the '**wakeword/model**' directory.

- **train_and_eval.py**: Main script to train and evaluate the wake word detection model.
- **train_final_model.py**: Script used to train the final model for inference.

## Future Work
- RF Signal Transmission: The next step is to implement the RF transmission for the fan controller. Due to current limitations with the Rasberry Pi 5, this feature is not yet fully implemented. The RF signal patterns have been decoded using Universal Radio Hacker (URH) and are included in the '**URH**' directory for reference. 
