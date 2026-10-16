import joblib

from data_preprocessing import clean_comment

MODEL_PATH = '../models/toxic_model.pkl'


def predict_comments(texts):
    saved = joblib.load(MODEL_PATH)
    model = saved['model']
    vectorizer = saved['vectorizer']
    labels = saved['labels']

    cleaned = []
    for text in texts:
        cleaned.append(clean_comment(text))

    X = vectorizer.transform(cleaned)
    pred = model.predict(X)

    results = []
    for i in range(len(texts)):
        found_labels = []
        for j in range(len(labels)):
            if pred[i][j] == 1:
                found_labels.append(labels[j])
        results.append(found_labels)
    return results


if __name__ == '__main__':
    comments = [
        'Thank you for the helpful edit, great work.',
        'You are so stupid, just shut up.',
        'The article needs more references and sources.',
    ]

    results = predict_comments(comments)

    for i in range(len(comments)):
        if len(results[i]) == 0:
            print(comments[i], '->', 'CLEAN (no label)')
        else:
            print(comments[i], '->', results[i])