import pandas as pd
from sklearn.utils import resample
############################################ BEGIN PREPARATION ############################################
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

'''# from json_satire, extract "headline" and "is_sarcastic"
ALL_DATA["text"] += list(json_satire["headline"])
ALL_DATA["satire"] += list(json_satire["is_sarcastic"])'''

'''# from danofer_satire, extract "comment" and "label"
ALL_DATA["text"] += list(danofer_satire["comment"])
ALL_DATA["satire"] += list(danofer_satire["label"])'''

'''# from onion_satire, extract "title". We add a series of all 1s since all headlines are satire
ALL_DATA["text"] += list(onion_satire["Title"])
ALL_DATA["satire"] += [1 for _ in range(len(onion_satire))]'''

# from huggingface_multimodal_satire, extract "headline" and "is_satire"
ALL_DATA["text"] += list(huggingface_multimodal_satire["headline"])
ALL_DATA["satire"] += list(huggingface_multimodal_satire["is_satire"])

'''# from huggingface_sarc, extract "DoesUseSarcasm" and "text"
ALL_DATA["text"] += list(huggingface_sarc["text"])
ALL_DATA["satire"] += list(huggingface_sarc["DoesUseSarcasm"])'''

# convert this dict to a df
ALL_DATA = pd.DataFrame(ALL_DATA)

# function that returns information about data. Useful if you ever want to know what the data looks like
def data_info():
    huggingface_multimodal_satire_info = ("Dataset: " + "huggingface_multimodal_satire", "Rows: " + str(len(huggingface_multimodal_satire)), 
                                          "Satire/NonSatire: " + str(huggingface_multimodal_satire["is_satire"].mean()) + "/" + str(1-huggingface_multimodal_satire["is_satire"].mean()))
    huggingface_sarc_info = ("Dataset: " + "huggingface_sarc", "Rows: " + str(len(huggingface_sarc)), "Satire/NonSatire: " + str(huggingface_sarc["DoesUseSarcasm"].mean()) + "/" + str(1-huggingface_sarc["DoesUseSarcasm"].mean()))
    onion_satire_info = ("Dataset: " + "onion_satire", "Rows: " + str(len(onion_satire)), "Satire/NonSatire: " + "1/0")
    danofer_satire_info = ("Dataset: " + "danofer_satire", "Rows: " + str(len(danofer_satire)), "Satire/NonSatire: " + str(danofer_satire["label"].mean()) + "/" + str(1-danofer_satire["label"].mean()))
    json_satire_info = ("Dataset: " + "json_satire", "Rows: " + str(len(json_satire)), "Satire/NonSatire: " + str(json_satire["is_sarcastic"].mean()) + "/" + str(1-json_satire["is_sarcastic"].mean()))
    if 1:
        print(huggingface_multimodal_satire_info)
        print(huggingface_sarc_info)
        print(onion_satire_info)
        print(danofer_satire_info)
        print(json_satire_info)

# remove any rows that have blank text
ALL_DATA = ALL_DATA[~ALL_DATA["text"].isnull()]

# Resetting index after removing rows
ALL_DATA.reset_index(drop=True, inplace=True)

# enable (print_data_info = 1) this to print datasets info
print_data_info = 0
if print_data_info: data_info()

# enable this if you want to print out the data
print_all_data = 0
if print_all_data: print(ALL_DATA)

ALL_DATA = ALL_DATA.sample(n=5000, random_state=42)

################## Undersampling. Our data currently has more non-satirical examples, so we will use this technique to even it out. ######################
non_satire_df = ALL_DATA[ALL_DATA["satire"] == 0]
satire_df = ALL_DATA[ALL_DATA["satire"] == 1]

# Undersample the majority class
undersampled_non_satirical = resample(non_satire_df, replace=False, n_samples=len(satire_df), random_state=42)

# Combine the undersampled majority class with the minority class
undersampled_df = pd.concat([undersampled_non_satirical, satire_df])

# Shuffle the DataFrame to randomize the order
undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
ALL_DATA = undersampled_df.copy()
# Now, undersampled_df contains a balanced dataset with equal examples of both classes
############################################ END PREPARATION ############################################
'''
('Dataset: huggingface_multimodal_satire', 'Rows: 10000', 'Satire/NonSatire: 0.4/0.6')
('Dataset: huggingface_sarc', 'Rows: 205645', 'Satire/NonSatire: 0.5001726275863746/0.4998273724136254')
('Dataset: onion_satire', 'Rows: 6851', 'Satire/NonSatire: 1/0')
('Dataset: danofer_satire', 'Rows: 1010826', 'Satire/NonSatire: 0.5/0.5')
('Dataset: json_satire', 'Rows: 28619', 'Satire/NonSatire: 0.476396799329117/0.5236032006708831')'''