# Análise de Sentimento em Tweets em Português

Projeto da disciplina de **Inteligência Artificial** (Engenharia da Computação — IFPB) voltado à comparação sistemática de pipelines clássicos de PLN e de um baseline léxico (LeIA) na classificação de polaridade em tweets em português.

**Autores:** Amanda Gabriela dos Santos Vasconcelos, Arthur Ramalho Souza Alves e Romero Galdino de Albuquerque

---

## Estrutura do Repositório

```text
.
├── pipeline_nlp_sentimentos.py          # Pipeline completo (EDA, benchmark, gráficos e inferência)
├── pipeline_nlp_sentimentos_colab.ipynb # Versão em notebook para Google Colab (opcional)
├── graficos/                            # Gráficos em 300 DPI e CSV de benchmarks (gerado na execução)
├── modelos_pkl/                         # Pipelines serializados (.pkl) com metadados (gerado na execução)
└── README.md
```

---

## Dataset

- **Fonte e Licença:** [`augustop/portuguese-tweets-for-sentiment-analysis`](https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis) (`NoThemeTweets.csv`, licença **CC BY-NC-SA 4.0**), obtido via `kagglehub`.
- **Amostragem:** subamostragem estratificada configurável (10.000 tweets, `RANDOM_STATE = 42`), com remoção de nulos e duplicatas antes e após a limpeza.
- **Divisão treino/teste:** split único estratificado (80% / 20%) compartilhado entre todos os modelos e o baseline léxico, com checagem explícita de sobreposição zero.
- **Controle de *target leakage*:** como os rótulos derivam de supervisão distante por emoticons, o pipeline remove emoticons/emojis por padrão para evitar vazamento e inclui teste de hipótese comparando cenários com e sem emoticons.

---

## Pré-processamento

1. Normalização Unicode (`NFKC`), conversão para minúsculas e remoção de URLs e menções (`@usuario`).
2. Tokenização e lematização em lote com **spaCy** (`pt_core_news_sm`, sem `parser` e `ner`).
3. Remoção de pontuação, números, espaços e tokens de 1 caractere.
4. Remoção de *stopwords* em português (**NLTK**), preservando negações e intensificadores (`não`, `nunca`, `jamais`, `mas`, `muito`, `pouco`, `bom`, `ruim`).

---

## Modelos Avaliados

| Experimento | Vetorizador | Classificador | Tuning |
| :--- | :--- | :--- | :--- |
| TF-IDF + Reg. Logística (Tuned) | TF-IDF (`word` 1–2) | `LogisticRegression` | `GridSearchCV` (5-fold) |
| BoW (1,2) + Reg. Logística | `CountVectorizer` (binário, 1–2) | `LogisticRegression` | Parâmetros fixos |
| TF-IDF + LinearSVC (Tuned) | TF-IDF (`word` 1–2) | `LinearSVC` | `GridSearchCV` (5-fold) |
| TF-IDF + Naive Bayes (Tuned) | TF-IDF (`word` 1–2) | `MultinomialNB` | `GridSearchCV` (5-fold) |
| TF-IDF + ComplementNB (Tuned) | TF-IDF (`word` 1–2) | `ComplementNB` | `GridSearchCV` (5-fold) |
| BoW (1,2) + Naive Bayes | `CountVectorizer` (`word` 1–2) | `MultinomialNB` | Parâmetros fixos |
| TF-IDF Char (3,5) + LinearSVC | TF-IDF (`char_wb` 3–5) | `LinearSVC` | Parâmetros fixos |
| LeIA (Baseline Léxico) | Léxico | `SentimentIntensityAnalyzer` | — |

---

## Avaliação

- **Validação cruzada:** `StratifiedKFold(n_splits=5, shuffle=True)` otimizando **F1-Macro**.
- **Qualidade preditiva:** Acurácia, Precision, Recall e F1 (macro e por classe), matrizes de confusão dos 4 melhores pipelines e inspeção qualitativa de erros.
- **Desempenho operacional:** tempo de treino/tuning, inferência em lote, latência unitária média (*modelo puro* vs. *end-to-end* com spaCy em 100 tweets) e tamanho do `.pkl`.

---

## Como Executar

### 1. Clonar e criar o ambiente virtual

```bash
git clone [https://github.com/](https://github.com/)<seu-usuario>/<seu-repo>.git
cd <seu-repo>
python -m venv .venv
```

Ative o ambiente:

- **Windows (PowerShell):**
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Instalar dependências e modelo do spaCy

```bash
pip install kagglehub matplotlib numpy pandas nltk spacy scikit-learn leia-br
python -m spacy download pt_core_news_sm
```

### 3. Executar o pipeline

```bash
python pipeline_nlp_sentimentos.py
```

---

## Resultados

> Com os valores de `graficos/resultados_benchmarks.csv`.

| Experimento | Acurácia | F1-Macro | Tempo Treino (s) | Runtime E2E (ms) | Tamanho (.pkl) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Char (3,5) + LinearSVC** | 0.7501 | **0.7305** | 0.51 | 3.92 | 0.82 MB |
| TF-IDF + ComplementNB (Tuned) | 0.7564 | 0.7299 | 1.22 | 3.23 | 0.64 MB |
| TF-IDF + Reg. Logística (Tuned) | 0.7397 | 0.7206 | 12.41 | 3.23 | 0.38 MB |
| TF-IDF + LinearSVC (Tuned) | 0.7371 | 0.7182 | 1.86 | 2.96 | 0.25 MB |
| BoW (1,2) + Reg. Logística | 0.7413 | 0.7176 | 0.25 | 2.92 | 0.32 MB |
| BoW (1,2) + Naive Bayes | 0.7616 | 0.7152 | 0.19 | 2.98 | 0.51 MB |
| TF-IDF + Naive Bayes (Tuned) | 0.7480 | 0.6900 | 1.12 | 3.28 | 0.58 MB |
| LeIA (Léxico s/ emoticons) | 0.5273 | 0.5269 | 0.00 | 0.11 | — |