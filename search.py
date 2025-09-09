
import json
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class SearchEngine:
    def __init__(self, data_path='leetcode_problems.json'):
        self.documents = self._load_documents(data_path)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.tf = {}
        self.idf = {}
        self.tfidf = {}
        self._precompute_tfidf()

    def _load_documents(self, data_path):
        with open(data_path, 'r') as f:
            return json.load(f)

    def _preprocess(self, text):
        words = word_tokenize(text.lower())
        return [self.stemmer.stem(w) for w in words if w.isalnum() and w not in self.stop_words]

    def _precompute_tfidf(self):
        num_documents = len(self.documents)
        all_words = set()
        doc_tokens = []

        for doc in self.documents:
            tokens = self._preprocess(doc['title'] + ' ' + doc['description'])
            doc_tokens.append(tokens)
            for word in tokens:
                all_words.add(word)

        # Compute TF
        for i, tokens in enumerate(doc_tokens):
            doc_tf = {}
            for word in tokens:
                doc_tf[word] = doc_tf.get(word, 0) + 1
            for word, freq in doc_tf.items():
                doc_tf[word] = freq / len(tokens)
            self.tf[i] = doc_tf

        # Compute IDF
        for word in all_words:
            doc_count = sum(1 for tokens in doc_tokens if word in tokens)
            self.idf[word] = math.log(num_documents / (1 + doc_count))

        # Compute TF-IDF
        for i, doc_tf in self.tf.items():
            doc_tfidf = {}
            for word, tf_val in doc_tf.items():
                doc_tfidf[word] = tf_val * self.idf.get(word, 0)
            self.tfidf[i] = doc_tfidf

    def search(self, query, top_n=20):
        query_tokens = self._preprocess(query)
        query_tfidf = {}

        # TF for query
        query_tf = {}
        for word in query_tokens:
            query_tf[word] = query_tf.get(word, 0) + 1
        for word, freq in query_tf.items():
            query_tf[word] = freq / len(query_tokens)

        # TF-IDF for query
        for word, tf_val in query_tf.items():
            query_tfidf[word] = tf_val * self.idf.get(word, 0)

        # Cosine Similarity
        scores = {}
        for i, doc_tfidf in self.tfidf.items():
            dot_product = 0
            doc_norm = 0
            query_norm = 0

            all_words = set(doc_tfidf.keys()) | set(query_tfidf.keys())

            for word in all_words:
                dot_product += doc_tfidf.get(word, 0) * query_tfidf.get(word, 0)
                doc_norm += doc_tfidf.get(word, 0) ** 2
                query_norm += query_tfidf.get(word, 0) ** 2
            
            doc_norm = math.sqrt(doc_norm)
            query_norm = math.sqrt(query_norm)

            if doc_norm > 0 and query_norm > 0:
                scores[i] = dot_product / (doc_norm * query_norm)
            else:
                scores[i] = 0

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for i, score in sorted_docs[:top_n]:
            results.append(self.documents[i])
        return results
