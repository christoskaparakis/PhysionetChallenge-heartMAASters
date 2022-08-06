# Multi-label classification on 12, 6, 4, 3 and 2 lead electrocardiography signals using convolutional recurrent neural networks

## What's in this repository?

The official code for the following paper [Multi-Label Classification on 12, 6, 4, 3 and 2 Lead Electrocardiography Signals Using Convolutional Recurrent Neural Networks](https://ieeexplore.ieee.org/document/9662725)


## Summary

We've created a Convolutional Recurrent Neural Network (CRNN) to identify cardiac abnormalities in 12, 6, 4, 3 and 2 lead ECG data. Multi-label classification with CRNNs relies on effective data pre-processing, model architecture and hyperparameter tuning. ECG signals were first pre-processed and then zero-padded or clipped to have an equal duration of 10 seconds). 

Additionally, a wavelet-based ECG segmentation algorithm was used to extract the characteristics and locations of the PQRST complexes (features), and both PQRST fiducial points and extracted features were used as inputs to two Convolutional Recurrent Neural Networks (CRNN), respectively, each one consisting of eight layers. 

![alt text](https://github.com/ckaparakis/PhysionetChallenge-heartMAASters/blob/main/figures/PQRST.png?raw=true)

The two CRNNs were subsequently concatenated, and succeded by fully connected layers and a sigmoid activation function. The sigmoid layer ensures that the captured information from the two CRNNs are combined into one output, a prediction of which classes a specific recording belongs to. The sigmoid layer creates a probability per class; the classes that exceed the cutoff ratio are selected as predicted classes.

![alt text](https://github.com/ckaparakis/PhysionetChallenge-heartMAASters/blob/main/figures/model_scheme.png?raw=true)

## How do I run these scripts?

You can run this classifier code by installing the requirements

    pip install requirements.txt

and running

    python train_model.py training_data model
    python test_model.py model test_data test_outputs

where `training_data` is a folder of training data files, `model` is a folder for saving your models, `test_data` is a folder of test data files (you can use the training data locally for debugging and cross-validation), and `test_outputs` is a folder for saving your models' outputs. The [PhysioNet/CinC Challenge 2021 webpage](https://physionetchallenges.org/2021/) provides training databases with data files and a description of the contents and structure of these files.

After training your model and obtaining test outputs with above commands, you can evaluate the scores of your models using the [PhysioNet/CinC Challenge 2021 evaluation code](https://github.com/physionetchallenges/evaluation-2021) by running

    python evaluate_model.py labels outputs scores.csv class_scores.csv

where `labels` is a folder containing files with one or more labels for each ECG recording, such as the training database on the PhysioNet webpage; `outputs` is a folder containing files with outputs produced by your models for those recordings; `scores.csv` (optional) is a collection of scores for your models; and `class_scores.csv` (optional) is a collection of per-class scores for your models.


## How do I train, save, load, and run my model?

To train and save your models, please edit the `training_code` function in the `team_code.py` script.

To load and run your trained model, please edit the `load_twelve_lead_model`, `load_six_lead_model`, `load_three_lead_model`, and `load_two_lead_model` functions as well as the `run_twelve_lead_model`, `run_six_lead_model`, `run_three_lead_model` and `run_two_lead_model` functions in the `team_code.py` script, which takes an ECG recording as an input and returns the class labels and probabilities for the ECG recording as outputs.

## What else is in this repository?

This README has instructions for running the example code and writing and running your own code.

We also included a script, `extract_leads_wfdb.py`, for extracting reduced-lead sets from the training data. You can use this script to produce reduced-lead data that you can use with your code. You can run this script using the following commands:

    python extract_leads_wfdb.py -i twelve_lead_directory -o two_lead_directory -l II V5 
    python extract_leads_wfdb.py -i twelve_lead_directory -o three_lead_directory -l I II V2 
    python extract_leads_wfdb.py -i twelve_lead_directory -o six_lead_directory -l I II III aVL aVR aVF 

Here, the `-i` argument gives the input folder, the `-o` argument gives the output folder, and the `-l` argument gives the leads.


## How do I learn more?

Please see the [PhysioNet/CinC Challenge 2021 webpage](https://physionetchallenges.org/2021/) for more details. Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/forum/#!forum/physionet-challenges).

## Useful links

* [The PhysioNet/CinC Challenge 2021 webpage](https://physionetchallenges.org/2021/)
* [MATLAB example code for the PhysioNet/CinC Challenge 2021](https://github.com/physionetchallenges/matlab-classifier-2021)
* [Evaluation code for the PhysioNet/CinC Challenge 2021](https://github.com/physionetchallenges/evaluation-2021) 
* [2021 Challenge Frequently Asked Questions (FAQ)](https://physionetchallenges.org/2021/faq/) 
* [Frequently Asked Questions (FAQ)](https://physionetchallenges.org/faq/) 
