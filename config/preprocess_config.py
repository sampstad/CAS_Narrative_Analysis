"""
Central configuration for the 02_preprocess notebook.
"""

# path to the raw data TSV file generated with 01_load.ipynb
INPUT_TSV = "data\\filtered_dataset.tsv"

# columns to drop from the filtered dataframe that are not needed for the analysis
COLUMNS_TO_DROP = ["regional", 
                   "doctype", 
                   "doctype_description", 
                   "language", 
                   "dateline",
                   "subhead", #no longer needed after merging with content
                   "article_link", 
                   "content_id", 
                   "rubric_from_url" #only needed for filtering in notebook 01_load
                   ]

# publishers per publication dictionary
MEDIUM_CODE_TO_PUBLISHER = {
    "ZWAO": "TX Group",
    "AZM": "CH Media",
    "BAZ": "TX Group",
    "BZ": "TX Group",
    "BLIO": "Ringier",
    "WOZ": "infolink",
    "WEW": "Weltwoche",
    "NZZ": "NZZ-Mediengruppe",
    "LUZ": "CH Media",
    "SRF": "SRG",
    "SGT": "CH Media",
    "TA": "TX Group"
}

# configuration for sentiment analysis
SENTIMENT_MODEL_NAME = "oliverguhr/german-sentiment-bert"
SENTIMENT_DEVICE = "cuda"
SENTIMENT_BATCH_SIZE = 64
SENTIMENT_MAX_LENGTH = 512
SENTIMENT_TRUNCATION = True

# configuration for emotion analysis
EMOTION_MODEL_NAME = "MilaNLProc/xlm-emo-t"
EMOTION_DEVICE = "cuda"
EMOTION_BATCH_SIZE = 64
EMOTION_MAX_LENGTH = 512
EMOTION_TRUNCATION = True

# path to the output TSV file after preprocessing
OUTPUT_TSV = "data\\preprocessed_dataset.tsv"