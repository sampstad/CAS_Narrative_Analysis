"""
All functions for the 02_preprocess notebook
"""

from config import preprocess_config as cfg

from tqdm import tqdm
import textstat


_sentiment_pipe = None

def _get_sentiment_pipe():
    from transformers import pipeline
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline(
            task="sentiment-analysis",
            model=cfg.SENTIMENT_MODEL_NAME,
            device=cfg.SENTIMENT_DEVICE,
            batch_size=cfg.SENTIMENT_BATCH_SIZE,
            truncation=cfg.SENTIMENT_TRUNCATION,
            max_length=cfg.SENTIMENT_MAX_LENGTH,
        )
    return _sentiment_pipe

def score_sentiment(df):
    pipe = _get_sentiment_pipe()
    texts = (df["head"].fillna("") + " " + df["content"].fillna("")).tolist()
    scores = []
    batch_size = cfg.SENTIMENT_BATCH_SIZE
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring sentiment"):
        batch = texts[i:i+batch_size]
        results = pipe(batch, top_k=None)  # get all class probabilities
        for result in results:
            label_scores = {r["label"].lower(): r["score"] for r in result}
            score = label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0)
            scores.append(score)
    return scores


_emotion_pipe = None

def _get_emotion_pipe():
    global _emotion_pipe
    if _emotion_pipe is None:
        from transformers import pipeline
        _emotion_pipe = pipeline(
            task="text-classification",
            model=cfg.EMOTION_MODEL_NAME,
            device=cfg.EMOTION_DEVICE,
            batch_size=cfg.EMOTION_BATCH_SIZE,
            truncation=cfg.EMOTION_TRUNCATION,
            max_length=cfg.EMOTION_MAX_LENGTH,
            top_k=None,  # return all 4 emotion scores
        )
    return _emotion_pipe

def score_emotions(df):
    pipe = _get_emotion_pipe()
    texts = (df["head"].fillna("") + " " + df["content"].fillna("")).tolist()
    
    anger, joy, sadness, fear = [], [], [], []
    
    for i in tqdm(range(0, len(texts), cfg.EMOTION_BATCH_SIZE), desc="Scoring emotions"):
        batch = texts[i:i+cfg.EMOTION_BATCH_SIZE]
        results = pipe(batch)
        for result in results:
            scores = {r["label"]: r["score"] for r in result}
            anger.append(scores.get("anger", 0.0))
            joy.append(scores.get("joy", 0.0))
            sadness.append(scores.get("sadness", 0.0))
            fear.append(scores.get("fear", 0.0))
    
    return anger, joy, sadness, fear


def flesch_reading_ease_de(text):

    if not text or not str(text).strip():
        return None
    textstat.set_lang("de")
    return textstat.flesch_reading_ease(text)