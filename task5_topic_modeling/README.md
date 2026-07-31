# Task 5: Topic Modeling on News Articles

Unsupervised topic discovery on 2,225 BBC News articles using Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF). The workflow lowercases and tokenizes text, removes stopwords, stems words, extracts five latent topics, and reports the most significant words per topic.

## Approaches

- **LDA:** trained on document-term counts, with an interactive pyLDAvis visualization.
- **NMF:** trained on TF-IDF features as the bonus comparison.
- **Comparison:** Gensim c_v coherence measures topic interpretability, topic diversity measures vocabulary overlap, and normalized mutual information (NMI) compares discovered document topics with the five known BBC categories only for external evaluation, never for training.

LDA perplexity and NMF reconstruction error are model-specific and should not be compared directly. Coherence, topic diversity, and category NMI use the same scale for both models; higher is better.

## Results

| Model | c_v coherence | Topic diversity | Category NMI |
| --- | ---: | ---: | ---: |
| LDA | 0.5616 | 0.8167 | 0.7918 |
| **NMF** | **0.7317** | **0.9333** | **0.8202** |

Both models recovered politics, technology, business, entertainment, and sport. NMF produced the more coherent, less repetitive topics and aligned more closely with the withheld BBC category labels.

## Setup and run

```bash
source .venv/bin/activate
pip install -r task5_topic_modeling/requirements.txt
python -m task5_topic_modeling.src.download_data
python -m task5_topic_modeling.src.train
```

## Dataset

The project uses the original [BBC News dataset](https://derekgreene.com/bbc/) from the UCD Machine Learning Group: 2,225 articles from 2004–2005 across business, entertainment, politics, sport, and technology. The downloader reads the original text archive, validates the article count and categories, and creates a local CSV excluded from Git.

## Generated outputs

- `metrics.json` — LDA/NMF quality and dataset statistics
- `lda_topic_words.csv` and `nmf_topic_words.csv`
- `lda_topic_words.png` and `nmf_topic_words.png`
- `lda_visualization.html` — interactive pyLDAvis topic explorer
- `document_topics.csv` — dominant topic for every article
