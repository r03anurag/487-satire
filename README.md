
# Stance Detection in Satire


A BERT Transformer model, trained on multiple examples of satirical headlines, that predicts whether unseen headlines are satirical or not.

## Description

Being able to distinguish satire from genuine news will improve a language model’s ability to analyze a text that may not directly say what they mean and correctly determine the true motivation behind it. This may be used to help readers identify the difference between satirical and genuine news reports. Some challenges of this task are that satire is inherently deceptive. Satire articles often want readers to doubletake and wonder if that could possibly be a real headline. This leads to article titles that seem very possibly real even to human readers but on closer inspection are likely to be false and satirical.

## Getting Started

### Datasets

* The dataset we used for this project is called "huggingface_multimodal_satire". As contained within the name, the dataset comes from the popular website HuggingFace, a collection of datasets for various ML tasks.
    * To replicate our exact results, uncomment the following line of code in ```main.ipynb```:
    ```
    satire_data = pd.read_csv("data_20231204.csv")  '''UNCOMMENT THIS LINE TO REPRODUCE RESULTS OBTAINED ON 12/4'''
    ```
    * This dataset is a subset of the original "huggingface_multimodal_satire" dataset, and was created by randomly sampling 5000   rows out of the original. Later on, we split the data into 4000 train / 1000 test. 
    * You will find the links to all of the datasets in the Acknowledgments section; they are very large, so we are not able to upload them to GitHub.

### Dependencies and Installations

* Code is compatible with any Operating System. Make sure that the version of Python is between 3.7-3.10.
* You will need to ```pip install``` the following packages from the Python Standard Library:
    * ```pandas```
    * ```sklearn```
    * An engine that can support reading of parquet files (such as ```pyarrow``` or ```fastparquet```)
    * ```matplotlib``` (visualization)
    * ```torch```
    * ```nltk```
    * ```numpy```
    * ```tensorflow```
    * ```tensorflow_hub```
    * ```tensorflow_text```
    * ```seaborn``` (visualization)

### Using the Code

* The ```main.ipynb``` file is set up in a way that you can run all components of the process sequentially. Refer to this file for instructions.
* As previously stated, if you want to replicate our exact results, uncomment the line of code mentioned above.
* ```prepare_data.py``` is the script that is responsible for extracting useful information from raw data. You can modify this file to use whatever datasets you want, import new datasets and/or libraries, and change how undersampling/oversampling is done. Currently, we undersample non-satirical data to balance our dataset, but this may change depending on which data you use.
## Authors

- Anurag Renduchintala (ranurag)
- Yoojin Bae (yoojinb)
- Karl Yan (karlyan)


## Acknowledgements and Links

 - [A simple README.md template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
 - HuggingFace Datasets:
    - https://huggingface.co/datasets/phosseini/multimodal_satire
    - https://huggingface.co/datasets/reza-alipour/SARC_Sarcasm
 - Kaggle Datasets:
    - https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection/data 
    - https://www.kaggle.com/datasets/danofer/sarcasm
    - https://www.kaggle.com/datasets/undefinenull/satirical-news-from-the-onion

