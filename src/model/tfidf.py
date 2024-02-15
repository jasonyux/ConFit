from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer


def _dummy_fun(doc):
    return doc


def tfidf_train(tokenized_corpus: List[List[str]], max_features: int = 100):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        analyzer='word',
        tokenizer=_dummy_fun,  # because we tokenize by ourself
        preprocessor=_dummy_fun,
        token_pattern=None
    )
    x = vectorizer.fit_transform(tokenized_corpus)
    return x, vectorizer


def tfidf_predict(tokenized_corpus: List[List[str]], vectorizer):
    x = vectorizer.transform(tokenized_corpus)
    return x