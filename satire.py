import pandas as pd

############################################ BEGIN PREPROCESSING ############################################
# read in all the raw data
# for parquet files, make sure you have either 'pyarrow' or 'fastparquet' engine support.
# you can download it with "pip install pyarrow" / "pip install fastparquet"
huggingface_multimodal_satire = pd.read_parquet("Data/huggingface_multimodal_satire/train-00000-of-00001-593392955b371d90.parquet", engine='pyarrow')  
huggingface_sarc = pd.read_parquet("Data/huggingface_SARC/train-00000-of-00001-61cc87ff0773a4cc.parquet", engine='pyarrow')
onion_satire = pd.read_csv("Data/The Onions Breaking News - News In Brief.csv")  
danofer_satire = pd.read_csv("Data/train-balanced-sarcasm.csv") 
json_satire = pd.read_json("Data/Sarcasm_Headlines_Dataset_v2.json") 

# preprocess the data, extract useful columns, put them together
ALL_DATA = {"text": [], "satire": []}

# from json_satire, extract "headline" and "is_sarcastic"
ALL_DATA["text"] += list(json_satire["headline"])
ALL_DATA["satire"] += list(json_satire["is_sarcastic"])

# from danofer_satire, extract "comment" and "label"
ALL_DATA["text"] += list(danofer_satire["comment"])
ALL_DATA["satire"] += list(danofer_satire["label"])

# from onion_satire, extract "title". We add a series of all 1s since all headlines are satire
ALL_DATA["text"] += list(onion_satire["Title"])
ALL_DATA["satire"] += [1 for _ in range(len(onion_satire))]

# from huggingface_multimodal_satire, extract "headline" and "is_satire"
ALL_DATA["text"] += list(huggingface_multimodal_satire["headline"])
ALL_DATA["satire"] += list(huggingface_multimodal_satire["is_satire"])

# from huggingface_sarc, extract "DoesUseSarcasm" and "text"
ALL_DATA["text"] += list(huggingface_sarc["text"])
ALL_DATA["satire"] += list(huggingface_sarc["DoesUseSarcasm"])

# convert this dict to a df
ALL_DATA = pd.DataFrame(ALL_DATA)

# function that returns information about data. Useful if you ever want to know what the data looks like
def data_info(only_total):
    huggingface_multimodal_satire_info = ("Dataset: " + "huggingface_multimodal_satire", "Rows: " + str(len(huggingface_multimodal_satire)), 
                                          "Satire/NonSatire: " + str(huggingface_multimodal_satire["is_satire"].mean()) + "/" + str(1-huggingface_multimodal_satire["is_satire"].mean()))
    huggingface_sarc_info = ("Dataset: " + "huggingface_sarc", "Rows: " + str(len(huggingface_sarc)), "Satire/NonSatire: " + str(huggingface_sarc["DoesUseSarcasm"].mean()) + "/" + str(1-huggingface_sarc["DoesUseSarcasm"].mean()))
    onion_satire_info = ("Dataset: " + "onion_satire", "Rows: " + str(len(onion_satire)), "Satire/NonSatire: " + "1/0")
    danofer_satire_info = ("Dataset: " + "danofer_satire", "Rows: " + str(len(danofer_satire)), "Satire/NonSatire: " + str(danofer_satire["label"].mean()) + "/" + str(1-danofer_satire["label"].mean()))
    json_satire_info = ("Dataset: " + "json_satire", "Rows: " + str(len(json_satire)), "Satire/NonSatire: " + str(json_satire["is_sarcastic"].mean()) + "/" + str(1-json_satire["is_sarcastic"].mean()))
    all_info = ("Dataset: ALL_DATA", "Rows: " + str(len(ALL_DATA)), "Satire/NonSatire: " + str(ALL_DATA["satire"].mean()) + "/" + str(1-ALL_DATA["satire"].mean()))
    if not only_total:
        print(huggingface_multimodal_satire_info)
        print(huggingface_sarc_info)
        print(onion_satire_info)
        print(danofer_satire_info)
        print(json_satire_info)
    print(all_info)

# enable (print_data_info = 1) this to print datasets info
print_data_info = 1
print_only_total = 1    # if we want info about all individual datasets as well, set this to 0/False
if print_data_info: data_info(print_only_total)

# lowercase all the headlines/comments
ALL_DATA["text"] = ALL_DATA["text"].str.lower()
############################################ END PREPROCESSING ############################################

