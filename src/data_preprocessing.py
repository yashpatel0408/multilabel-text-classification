import pandas as pd

LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_comment(text):
    # ek comment ko saaf karta hai: lowercase aur new line hatana
    text = str(text)
    text = text.replace('\n', ' ')
    text = text.lower()
    return text


def clean_data(df):
    df = df.copy()
    df = df.drop('id', axis=1)
    df['comment_text'] = df['comment_text'].fillna('')
    df['comment_text'] = df['comment_text'].apply(clean_comment)
    return df