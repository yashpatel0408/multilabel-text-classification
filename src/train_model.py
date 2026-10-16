import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, hamming_loss, roc_auc_score

from data_preprocessing import load_data, clean_data, LABEL_COLS

DATA_PATH = '../data/train.csv'
MODEL_PATH = '../models/toxic_model.pkl'


def main():
    df = clean_data(load_data(DATA_PATH))

    X = df['comment_text']
    y = df[LABEL_COLS]

    # toxic ke hisaab se stratify, taaki dono parts mein toxic ka ratio same rahe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['toxic']
    )

    # text ko numbers mein badalna (TF-IDF), sirf train pe fit karna hai
    vectorizer = TfidfVectorizer(
        stop_words='english', max_features=30000, ngram_range=(1, 2), min_df=5
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        'Logistic Regression': OneVsRestClassifier(
            LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
        ),
        'Linear SVM': OneVsRestClassifier(
            LinearSVC(class_weight='balanced')
        ),
    }

    best_name = None
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        pred = model.predict(X_test_tfidf)

        macro_f1 = f1_score(y_test, pred, average='macro', zero_division=0)
        auc = roc_auc_score(y_test, model.decision_function(X_test_tfidf), average='macro')

        print('Model:', name)
        print(classification_report(y_test, pred, target_names=LABEL_COLS, zero_division=0))
        print('Macro F1:', macro_f1)
        print('Macro ROC-AUC:', auc)
        print('Hamming Loss:', hamming_loss(y_test, pred))
        print('-' * 40)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name
            best_model = model

    print('Best model:', best_name)

    joblib.dump(
        {'model': best_model, 'vectorizer': vectorizer, 'labels': LABEL_COLS},
        MODEL_PATH,
        compress=3
    )
    print('Model saved at', MODEL_PATH)


if __name__ == '__main__':
    main()