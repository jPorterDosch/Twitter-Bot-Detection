import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np

bot_df = pd.read_parquet('data/revised_bot_data.parquet')
human_df = pd.read_parquet('data/revised_user_data.parquet')

text_df = pd.concat([bot_df, human_df], ignore_index=True)

text_df.head()

print(f"cuda is available: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)
model.eval() # Turn off dropout layers for inference

def embed_texts(texts, max_len=128, batch_size=256):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]

        # tokenize values
        enc = tokenizer(
            batch,
            padding='longest',
            truncation=True,
            max_length=max_len,
            return_tensors='pt', # return torch tensors
        )

        # Make sure any tensors returned by the tokenizer are on correct device to prevent mismatch
        enc = {k: v.to(device) for k, v in enc.items() }

        with torch.inference_mode():
            out = model(**enc)
        # Take out [CLS] token from last hidden state
        cls_vectors = out.last_hidden_state[:, 0, :].cpu()
        embs.append(cls_vectors)
    return torch.cat(embs, dim=0).numpy()

tweet_cols = [f'{i}' for i in range(1915)]

# Average user tweet embeddings to get one embedding per user
def embed_user_tweets(row):
    tweets = []
    for c in tweet_cols:
        txt = row[c]
        if pd.notna(txt) and isinstance(txt, str) and txt.lower() != 'none' and txt.strip():
            tweets.append(txt)
    if not tweets:
        # 0 vector
        return np.zeros(model.config.hidden_size, dtype=np.float32)

    tweet_embs = embed_texts(tweets)

    # Average tweet values
    return tweet_embs.mean(axis=0)

text_df['tweet_vectors'] = text_df.apply(embed_user_tweets, axis = 1)

print(f"user_vectors.type: {type(text_df['tweet_vectors'])}")
print("user_vectors.head():")

print(text_df.head())

print("="*25, "Converting to parquet", "="*25)
text_df.to_parquet('/home/jdosch1/personal/COSC325_Final/data/tweet_vectors.parquet')