import spacy
nlp = spacy.load('pt_core_news_sm')

from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelmax
from itertools import islice
import numpy as np


def window(seq, n=3):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n: 
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def get_local_maxima(depth_scores, order=1):
    maxima_ids = argrelmax(depth_scores, order=order)[0]
    filtered_scores = np.zeros(len(depth_scores))
    filtered_scores[maxima_ids] = depth_scores[maxima_ids]
    return filtered_scores

def compute_threshold(scores):
    s = scores[np.nonzero(scores)]
    threshold = np.mean(s) - (np.std(s) / 2)
    # threshold = np.mean(s) - (np.std(s))
    return threshold

def get_threshold_segments(scores, threshold=0.1):
     segment_ids = np.where(scores >= threshold)[0]
     return segment_ids

def climb(seq, i, mode='left'):
    if mode == 'left':
        while True:
            curr = seq[i]
            if i == 0:
                return curr
            i = i-1
            if not seq[i] > curr:
                return curr
    if mode == 'right':
        while True:
            curr = seq[i]
            if i == (len(seq)-1):
                return curr
            i = i+1
            if not seq[i] > curr:
                return curr

def get_depths(scores):
    depths = []
    for i in range(len(scores)):
        score = scores[i]
        l_peak = climb(scores, i, mode='left')
        r_peak = climb(scores, i, mode='right')
        depth = 0.5 * (l_peak + r_peak - (2*score))
        depths.append(depth)
    return np.array(depths)



with open('text/exemplo.txt', 'r', encoding="utf-8") as f:
	data = [i for i in (f.read().splitlines()) if i != '']
data
sents = []
for text in data:
    doc = nlp(text)
    for sent in doc.sents:
        sents.append(sent)

from sentence_transformers import SentenceTransformer
# MODEL_STR = "sentence-transformers/all-MiniLM-L6-v2"
# model = SentenceTransformer(MODEL_STR)
MODEL_STR = "bert-base-multilingual-cased"
model = SentenceTransformer(MODEL_STR)


WINDOW_SIZE = 5
window_sent = list(window(sents, WINDOW_SIZE))
window_sent = [' '.join([sent.text for sent in window]) for window in window_sent]


encoded_sent = model.encode(window_sent)

coherence_scores = [cosine_similarity([pair[0]], [pair[1]])[0][0] for pair in zip(encoded_sent, encoded_sent[1:])]

depth_scores = get_depths(coherence_scores)

filtered_scores = get_local_maxima(depth_scores, order=1)

threshold = compute_threshold(filtered_scores)
segments_ids = get_threshold_segments(filtered_scores, threshold)

segment_indices = segments_ids + WINDOW_SIZE
segment_indices = [0] + segment_indices.tolist() + [len(sents)]
slices = list(zip(segment_indices[:-1], segment_indices[1:]))

segmented = [sents[s[0]: s[1]] for s in slices]

