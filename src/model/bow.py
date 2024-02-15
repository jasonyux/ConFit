from typing import List
from sklearn.feature_extraction.text import CountVectorizer


def _dummy_fun(doc):
    return doc


def bow_train(tokenized_corpus: List[List[str]], max_features: int = 100):
    vectorizer = CountVectorizer(
        max_features=max_features,
        analyzer='word',
        tokenizer=_dummy_fun,  # because we tokenize by ourself
        preprocessor=_dummy_fun,
        token_pattern=None
    )
    x = vectorizer.fit_transform(tokenized_corpus)
    return x, vectorizer


def bow_predict(tokenized_corpus: List[List[str]], vectorizer):
    x = vectorizer.transform(tokenized_corpus)
    return x