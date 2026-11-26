# Análise de Sentimento em Tweets em Português

Projeto desenvolvido para a disciplina de **Processamento de Linguagem Natural** (Engenharia da Computação — IFPB), com foco na comparação sistemática de pipelines clássicos de PLN para classificação de sentimento em tweets em português.

O repositório implementa um fluxo completo de pré-processamento, análise exploratória, benchmark declarativo com ajuste de hiperparâmetros, comparação contra um baseline léxico (LeIA), visualização de resultados e serialização dos modelos treinados.

---

## Dataset

* **Fonte:** [`augustop/portuguese-tweets-for-sentiment-analysis`](https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis) (`NoThemeTweets.csv`), baixado automaticamente via `kagglehub`.
* **Amostragem:** subamostragem estratificada configurável (`LIMITE_TWEETS = 10000`, `RANDOM_STATE = 42`), com remoção de duplicatas antes e após a limpeza.
* **Divisão treino/teste:** split único estratificado (80% treino / 20% teste) compartilhado entre todos os modelos de ML e o baseline léxico, com checagem explícita de sobreposição para evitar vazamento de dados.
* **Controle de *data leakage*:** como o corpus utiliza supervisão distante baseada em emoticons, o pipeline permite controlar a remoção de emoticons ASCII e emojis Unicode (`PRESERVAR_EMOJIS_UNICODE = False` por padrão).

---

## Pré-processamento

O texto passa pelas seguintes etapas antes da vetorização:

1. Normalização Unicode (`NFKC`), conversão para minúsculas e remoção de URLs e menções (`@usuario`).
2. Tokenização e lematização com **spaCy** (`pt_core_news_sm`, com `parser` e `ner` desabilitados para maior velocidade).
3. Remoção de pontuação, números, espaços e tokens de 1 caractere.
4. Remoção de *stopwords* em português (**NLTK**), preservando explicitamente termos de negação e intensificadores (ex.: `não`, `nunca`, `jamais`, `mas`, `muito`, `pouco`, `bom`, `ruim`).

---

## Modelos e Experimentos Avaliados

O benchmark executa uma suíte declarativa comparando vetorizadores esparsos, classificadores lineares/probabilísticos e um baseline baseado em léxico:

| Experimento | Vetorizador | Classificador | Tuning |
| :--- | :--- | :--- | :--- |
| TF-IDF + Reg. Logística (Tuned) | TF-IDF (word 1–2) | LogisticRegression | GridSearchCV (5-fold) |
| BoW (1,2) + Reg. Logística | CountVectorizer (binário, 1–2) | LogisticRegression | Parâmetros fixos |
| TF-IDF + LinearSVC (Tuned) | TF-IDF (word 1–2) | LinearSVC | GridSearchCV (5-fold) |
| TF-IDF + Naive Bayes (Tuned) | TF-IDF (word 1–2) | MultinomialNB | GridSearchCV (5-fold) |
| TF-IDF + ComplementNB (Tuned) | TF-IDF (word 1–2) | ComplementNB | GridSearchCV (5-fold) |
| BoW (1,2) + Naive Bayes | CountVectorizer (word 1–2) | MultinomialNB | Parâmetros fixos |
| TF-IDF Char (3,5) + LinearSVC | TF-IDF (`char_wb` 3–5) | LinearSVC | Parâmetros fixos |
| LeIA (Baseline Léxico) | Léxico | `SentimentIntensityAnalyzer` | — |

---

## Avaliação e Métricas

* **Validação cruzada:** `StratifiedKFold(n_splits=5, shuffle=True)` otimizando **F1-Macro**.
* **Métricas de classificação:** Acurácia, Precision, Recall e F1 (macro e por classe), além de matrizes de confusão dos 4 melhores pipelines e inspeção qualitativa de erros.
* **Desempenho operacional:** tempo de treino/tuning, tempo de inferência em lote, latência unitária média (*modelo puro* vs. *end-to-end* incluindo spaCy em amostra aleatória de 100 tweets) e tamanho do arquivo `.pkl` gerado.

---

## Como Executar

### 1. Clonar o repositório e criar o ambiente virtual

```bash
git clone [https://github.com/](https://github.com/)<seu-usuario>/<seu-repo>.git
cd <seu-repo>
python -m venv .venv