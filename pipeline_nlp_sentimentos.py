# -*- coding: utf-8 -*-
"""
Projeto 3 - Análise de Sentimentos em Comentários de Redes Sociais (PLN)
Pipeline completo adaptado para execução local (Windows 11 / Linux / macOS).

Etapas incluídas:
  1. Configuração do ambiente, cache e carregamento do corpus (NoThemeTweets.csv)
  2. Pré-processamento e limpeza textual com spaCy + prevenção real de Target Leakage
  3. Análise Exploratória (EDA) com gráficos individuais em alta resolução
  4. Modelagem modular: BoW, TF-IDF (word e char), Reg. Logística, LinearSVC,
     MultinomialNB, ComplementNB, Tuning (GridSearchCV) e Baseline Léxico (LeIA)
  5. Execução de benchmarks, medição de tempos (treino, teste e runtime),
     serialização em .pkl, gráficos desacoplados e análise qualitativa de erros
  6. Experimento de hipótese: impacto dos emoticons (Target Leakage vs. Generalização)
"""

import hashlib
import html
import os
import pickle
import re
import sys
import time
import unicodedata

# ==============================================================================
# CONFIGURAÇÕES GERAIS (edite aqui conforme sua necessidade)
# ==============================================================================
LIMITE_AMOSTRA = 10000               # Use None para processar o dataset completo
USAR_LEMA = True                    # True = lematização com spaCy; False = apenas tokens limpos
REMOVER_EMOTICONS_PRINCIPAL = True  # True = evita Target Leakage da supervisão distante (:), :(
NORMALIZAR_ALONGAMENTOS = True      # True = reduz repetições extremas (ex.: kkkkkk -> kk, muuuuito -> muuito)
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5                        # Se ficar lento com dataset completo, reduza para 3

# Paralelismo (otimizado para Windows 11 + CPU multi-core sem estourar RAM)
N_PROCESS_SPACY = 2                 # Usado automaticamente apenas para lotes >= 20k textos
N_JOBS_SKLEARN = max(1, min(4, (os.cpu_count() or 2) - 1))  # Evita OOM no Windows com dataset completo
GRIDSEARCH_VERBOSE = 1              # Exibe progresso no terminal durante o GridSearchCV

# Cache de pré-processamento (evita reprocessar ~785k tweets no spaCy a cada execução)
USAR_CACHE_PREPROCESSAMENTO = True
PASTA_CACHE = "cache_dados"

# Saídas e visualização
PASTA_MODELOS = "modelos_pkl"
PASTA_GRAFICOS = "graficos"
EXIBIR_GRAFICOS = False             # False = apenas salva os PNGs (não trava o terminal); True = abre janela por gráfico
DPI_GRAFICOS = 300

# Experimento extra (Seção 6): Com vs. Sem emoticons
RODAR_EXPERIMENTO_EMOTICONS = True

# Configura backend gráfico antes de importar pyplot (evita janelas travando no Windows)
import matplotlib
if not EXIBIR_GRAFICOS:
    matplotlib.use("Agg")

import kagglehub
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import spacy
from LeIA import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from spacy.symbols import ORTH

# ==============================================================================
# CONSTANTES E REGEX PRÉ-COMPILADAS
# ==============================================================================
SPACY_MODELS = {
    "SMALL": "pt_core_news_sm",
    "MEDIUM": "pt_core_news_md",
    "LARGE": "pt_core_news_lg",
}

# Mantido vazio no pipeline principal para evitar leakage da supervisão distante (:), :(
EMOTICONS = []
EMOTICONS_SET = {emo.lower() for emo in EMOTICONS}

# Usado no experimento de hipótese (Seção 6)
LISTA_EMOTICONS_HIPOTESE = [
    ":)", ":(", ";)", ";(", "xD", "xd", ":D", ":d",
    ":/", "<3", ":-)", ":-(", "=)", "=(", ":p", ":P", ":'("
]

PALAVRAS_RELEVANTES_SENTIMENTO = {
    "não", "nao", "nem", "nunca", "jamais", "nada", "ninguém", "ninguem",
    "sem", "mas", "porém", "contudo", "entretanto", "todavia",
    "mais", "muito", "pouco", "bem", "mal", "bom", "ruim",
}

RE_URLS = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_MENCOES = re.compile(r"@\w+")
RE_RT = re.compile(r"\brt\b", re.IGNORECASE)
RE_ALONGAMENTOS = re.compile(r"(.)\1{2,}")
RE_ESPACOS = re.compile(r"\s+")

# Regex segura contra falsos positivos em horários (18:30), frações (8/10) e palavras (8pm),
# cobrindo emoticons ASCII com repetições (:(((, :))), xD, <3, ;-;, T_T e invertidos ((:, ):)
RE_EMOTICONS_LEAKAGE = re.compile(
    r"(?:[:;=][\-o\*\']?[\)\]\(\[\}\{@\|\\]+)"
    r"|(?:(?<!\w)[:;=8][\-o\*\']?[dDpP/]+(?!\w))"
    r"|(?:(?<!\w)8[\-o\*\']?[\)\]\(\[\}\{@\|\\]+(?!\w))"
    r"|(?:(?<!\w)[\(\)\[\]]+[\-o\*\']?[:;=](?!\w))"
    r"|(?:(?<!\w)<3+(?!\w))"
    r"|(?:(?<!\w)[xX][dD]+(?!\w))"
    r"|(?:(?<!\w)(?:[;:\^][_\-\.][;:\^]|[tT]_[tT])(?!\w))"
)

VETORIZADORES = {
    "tfidf": TfidfVectorizer,
    "bow": CountVectorizer,
}

CLASSIFICADORES = {
    "logistic_regression": LogisticRegression,
    "naive_bayes": MultinomialNB,
    "complement_nb": ComplementNB,
    "linear_svc": LinearSVC,
}


# ==============================================================================
# 1. CONFIGURAÇÃO DE RECURSOS NLP
# ==============================================================================
def configurar_console_utf8():
    """Evita UnicodeEncodeError ao imprimir emojis ou acentos no terminal do Windows."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def carregar_spacy(
    modelo: str = SPACY_MODELS["SMALL"],
    emoticons_especiais: list[str] = None,
    usar_lema: bool = True,
):
    """
    Carrega o modelo spaCy desativando componentes não utilizados.
    Quando `usar_lema=False`, desabilita também o pipeline morfológico para máxima velocidade.
    """
    desabilitar = ["parser", "ner"]
    if not usar_lema:
        desabilitar.extend(["tok2vec", "morphologizer", "attribute_ruler", "lemmatizer"])

    try:
        nlp_model = spacy.load(modelo, disable=desabilitar)
    except OSError as exc:
        raise RuntimeError(
            f"Modelo spaCy '{modelo}' não encontrado. "
            f"Execute no terminal: python -m spacy download {modelo}"
        ) from exc

    if emoticons_especiais:
        for emo in {e.lower() for e in emoticons_especiais}:
            nlp_model.tokenizer.add_special_case(emo, [{ORTH: emo}])
    return nlp_model


def carregar_stopwords_pt() -> set[str]:
    """Baixa (se necessário) e monta a lista de stopwords preservando termos de polaridade."""
    nltk.download("stopwords", quiet=True)
    nltk_stopwords = {s.lower() for s in stopwords.words("portuguese")}
    return nltk_stopwords - PALAVRAS_RELEVANTES_SENTIMENTO


def carregar_corpus() -> pd.DataFrame:
    """Baixa/carrega em cache o dataset NoThemeTweets.csv via kagglehub."""
    caminho_dataset = kagglehub.dataset_download("augustop/portuguese-tweets-for-sentiment-analysis")
    arquivo_csv = os.path.join(caminho_dataset, "NoThemeTweets.csv")
    return pd.read_csv(arquivo_csv)


# ==============================================================================
# 2. PRÉ-PROCESSAMENTO E LIMPEZA TEXTUAL
# ==============================================================================
def pre_limpar_string(
    texto: str,
    remover_emoticons: bool = REMOVER_EMOTICONS_PRINCIPAL,
    normalizar_alongamentos: bool = NORMALIZAR_ALONGAMENTOS,
) -> str:
    """
    Decodifica entidades HTML, normaliza Unicode (NFKC), remove URLs, menções (@),
    marcadores de RT, (opcionalmente) emoticons ASCII de supervisão distante
    e normaliza alongamentos excessivos.
    """
    texto = html.unescape(texto)
    texto = unicodedata.normalize("NFKC", texto)
    texto = RE_URLS.sub(" ", texto)
    texto = RE_MENCOES.sub(" ", texto)
    texto = RE_RT.sub(" ", texto)
    if remover_emoticons:
        texto = RE_EMOTICONS_LEAKAGE.sub(" ", texto)
    texto = texto.lower()
    if normalizar_alongamentos:
        texto = RE_ALONGAMENTOS.sub(r"\1\1", texto)
    return RE_ESPACOS.sub(" ", texto).strip()


def limpar_texto_para_leia(texto: str, remover_emoticons: bool = True) -> str:
    """Limpeza leve específica para o léxico LeIA (preserva pontuação e caixa alta)."""
    s = html.unescape(str(texto))
    s = RE_URLS.sub(" ", s)
    s = RE_MENCOES.sub(" ", s)
    s = RE_RT.sub(" ", s)
    if remover_emoticons:
        s = RE_EMOTICONS_LEAKAGE.sub(" ", s)
    return RE_ESPACOS.sub(" ", s).strip()


def is_emoji(texto: str) -> bool:
    """Identifica emojis reais pela categoria Unicode 'So' (Symbol, Other)."""
    return any(unicodedata.category(c) == "So" for c in texto)


def processar_tokens(
    doc,
    stopwords_set: set[str],
    usar_lema: bool = True,
    emoticons_set: set[str] = None,
) -> list[str]:
    """Filtra tokens preservando emojis/emoticons configurados e aplicando stopwords."""
    emo_set = emoticons_set if emoticons_set is not None else EMOTICONS_SET
    tokens_processados = []

    for token in doc:
        if token.is_space:
            continue

        termo = token.text.strip()

        if is_emoji(termo) or termo in emo_set:
            tokens_processados.append(termo)
            continue

        if token.is_punct or token.like_num:
            continue

        palavra = token.lemma_.lower().strip() if usar_lema else termo
        palavra = palavra.lstrip("#")

        # Descarta tokens sem nenhuma letra (ex.: horários '18:30', frações '8/10', símbolos residuais)
        if not any(c.isalpha() for c in palavra):
            continue

        if palavra and palavra not in stopwords_set and len(palavra) > 1:
            tokens_processados.append(palavra)

    return tokens_processados


def limpar_texto(
    texto: str,
    stopwords_set: set[str],
    nlp_model,
    usar_lema: bool = True,
    emoticons_set: set[str] = None,
    remover_emoticons: bool = REMOVER_EMOTICONS_PRINCIPAL,
    normalizar_alongamentos: bool = NORMALIZAR_ALONGAMENTOS,
) -> str:
    """Limpa um texto individual (usado em runtime e na inferência de frases inéditas)."""
    if not isinstance(texto, str):
        return ""
    pre_limpo = pre_limpar_string(
        texto,
        remover_emoticons=remover_emoticons,
        normalizar_alongamentos=normalizar_alongamentos,
    )
    doc = nlp_model(pre_limpo)
    return " ".join(
        processar_tokens(doc, stopwords_set, usar_lema=usar_lema, emoticons_set=emoticons_set)
    )


def _obter_caminho_cache(
    limite: int,
    usar_lema: bool,
    remover_emoticons: bool,
    normalizar_alongamentos: bool,
    random_state: int,
    stopwords_set: set[str],
    pasta_cache: str = PASTA_CACHE,
) -> str:
    """Gera nome de cache determinístico baseado nas configurações de pré-processamento."""
    assinatura = (
        f"lim={limite}|lema={usar_lema}|rem_emo={remover_emoticons}|"
        f"along={normalizar_alongamentos}|rs={random_state}|"
        f"sw={sorted(stopwords_set)}"
    )
    chave = hashlib.md5(assinatura.encode("utf-8")).hexdigest()[:10]
    os.makedirs(pasta_cache, exist_ok=True)
    return os.path.join(pasta_cache, f"corpus_limpo_{limite or 'full'}_{chave}.pkl")


def limpar_dataset(
    dataframe: pd.DataFrame,
    stopwords_set: set[str],
    nlp_model,
    coluna_texto: str = "tweet_text",
    coluna_alvo: str = "sentiment",
    limite: int = None,
    usar_lema: bool = True,
    remover_emoticons: bool = REMOVER_EMOTICONS_PRINCIPAL,
    normalizar_alongamentos: bool = NORMALIZAR_ALONGAMENTOS,
    emoticons_set: set[str] = None,
    batch_size: int = 1000,
    n_process: int = 1,
    random_state: int = 42,
    usar_cache: bool = USAR_CACHE_PREPROCESSAMENTO,
) -> pd.DataFrame:
    """
    Limpa o corpus em lote usando nlp.pipe (com suporte a cache em disco).
    Quando `limite` é definido, aplica amostragem estratificada por `coluna_alvo`.
    """
    stopwords_norm = {s.lower() for s in stopwords_set}
    caminho_cache = None

    if usar_cache:
        caminho_cache = _obter_caminho_cache(
            limite=limite,
            usar_lema=usar_lema,
            remover_emoticons=remover_emoticons,
            normalizar_alongamentos=normalizar_alongamentos,
            random_state=random_state,
            stopwords_set=stopwords_norm,
        )
        if os.path.exists(caminho_cache):
            print(f"[Cache encontrado] Carregando corpus pré-processado de: {caminho_cache}")
            return pd.read_pickle(caminho_cache)

    colunas_validacao = [c for c in [coluna_texto, coluna_alvo] if c in dataframe.columns]
    df_base = dataframe.dropna(subset=colunas_validacao).drop_duplicates(subset=[coluna_texto], keep="first")

    if limite is not None and limite < len(df_base):
        if coluna_alvo in df_base.columns and df_base[coluna_alvo].value_counts().min() >= 2:
            df_copia, _ = train_test_split(
                df_base,
                train_size=limite,
                random_state=random_state,
                stratify=df_base[coluna_alvo],
            )
            df_copia = df_copia.copy()
        else:
            df_copia = df_base.sample(n=limite, random_state=random_state).copy()
    else:
        df_copia = df_base.copy()

    textos_pre_limpos = [
        pre_limpar_string(
            str(t),
            remover_emoticons=remover_emoticons,
            normalizar_alongamentos=normalizar_alongamentos,
        )
        for t in df_copia[coluna_texto]
    ]

    # Evita overhead de spawn no Windows para lotes pequenos
    n_proc_efetivo = 1 if len(textos_pre_limpos) < 20000 else max(1, n_process)

    tokens_limpos_lote = [
        " ".join(
            processar_tokens(doc, stopwords_norm, usar_lema=usar_lema, emoticons_set=emoticons_set)
        )
        for doc in nlp_model.pipe(textos_pre_limpos, batch_size=batch_size, n_process=n_proc_efetivo)
    ]

    coluna_limpa = f"{coluna_texto}_limpo"
    df_copia[coluna_limpa] = tokens_limpos_lote

    # Remove textos vazios após a limpeza e deduplica
    df_copia = df_copia[df_copia[coluna_limpa].str.strip() != ""].reset_index(drop=True)
    total_antes = len(df_copia)
    df_copia = df_copia.drop_duplicates(subset=[coluna_limpa], keep="first").reset_index(drop=True)
    total_depois = len(df_copia)
    removidos = total_antes - total_depois

    print(f"Registros antes da deduplicação pós-limpeza: {total_antes:,}".replace(",", "."))
    print(f"Registros duplicados removidos: {removidos:,} ({removidos / max(total_antes, 1) * 100:.2f}%)".replace(",", "."))
    print(f"Registros únicos mantidos: {total_depois:,}".replace(",", "."))

    if usar_cache and caminho_cache:
        df_copia.to_pickle(caminho_cache)
        print(f"[Cache salvo] {caminho_cache}")

    return df_copia


# ==============================================================================
# 3. UTILITÁRIOS VISUAIS E ANÁLISE EXPLORATÓRIA (IMAGENS SEPARADAS)
# ==============================================================================
def _estilizar_eixo(ax, grade_eixo: str = "x"):
    """Remove bordas desnecessárias e aplica grade discreta para leitura limpa."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    if grade_eixo:
        ax.grid(axis=grade_eixo, linestyle="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)


def _exibir_e_salvar(
    fig,
    nome_arquivo: str = None,
    pasta_saida: str = PASTA_GRAFICOS,
    dpi: int = DPI_GRAFICOS,
    exibir: bool = EXIBIR_GRAFICOS,
):
    """Ajusta margens, salva a imagem individual em alta resolução e exibe se configurado."""
    fig.tight_layout()
    if pasta_saida and nome_arquivo:
        os.makedirs(pasta_saida, exist_ok=True)
        caminho = os.path.join(pasta_saida, nome_arquivo)
        fig.savefig(caminho, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"   [Gráfico salvo] {caminho}")
    if exibir:
        plt.show()
    plt.close(fig)


def obter_top_termos(serie_textos: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Conta os termos mais frequentes preservando emojis e emoticons."""
    vectorizer = CountVectorizer(token_pattern=r"\S+", max_features=top_n)
    matriz = vectorizer.fit_transform(serie_textos.astype(str))
    somas = matriz.sum(axis=0)
    freq = [(termo, int(somas[0, idx])) for termo, idx in vectorizer.vocabulary_.items()]
    return pd.DataFrame(freq, columns=["Termo", "Frequência"]).sort_values(by="Frequência", ascending=True)


def plotar_analise_exploratoria(
    dataset_limpo: pd.DataFrame,
    n_termos: int = 10,
    pasta_saida: str = PASTA_GRAFICOS,
):
    """Imprime estatísticas da EDA e gera gráficos individuais bem espaçados."""
    df_eda = dataset_limpo.copy()
    contagem = df_eda["sentiment"].value_counts()
    percentual = df_eda["sentiment"].value_counts(normalize=True) * 100

    resumo_classes = pd.DataFrame({
        "Quantidade": contagem,
        "Percentual (%)": percentual.round(2),
    })
    print("\n=== Distribuição de Classes ===")
    print(resumo_classes)

    df_eda["qtd_caracteres"] = df_eda["tweet_text_limpo"].astype(str).str.len()
    df_eda["qtd_palavras"] = df_eda["tweet_text_limpo"].astype(str).str.split().str.len()

    print("\n=== Estatísticas Gerais de Tamanho ===")
    print(df_eda[["qtd_caracteres", "qtd_palavras"]].describe().round(2))
    print("\n=== Tamanho Médio (Palavras) por Classe ===")
    print(df_eda.groupby("sentiment")["qtd_palavras"].describe().round(2))

    # --- Imagem 1: Distribuição de Classes ---
    paleta_base = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    cores_classes = [paleta_base[i % len(paleta_base)] for i in range(len(contagem))]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    barras = ax.bar(
        contagem.index.astype(str),
        contagem.values,
        width=0.52,
        color=cores_classes,
        edgecolor="#222222",
        linewidth=0.8,
    )
    ax.set_title("Distribuição de Classes no Corpus Limpo", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Sentimento", fontsize=11)
    ax.set_ylabel("Quantidade de Tweets", fontsize=11)
    ax.set_ylim(0, contagem.max() * 1.18)
    _estilizar_eixo(ax, grade_eixo="y")

    for barra, qtd, pct in zip(barras, contagem.values, percentual.values):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + contagem.max() * 0.025,
            f"{qtd:,}\n({pct:.1f}%)".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    _exibir_e_salvar(fig, "01_eda_distribuicao_classes.png", pasta_saida)

    # --- Imagem 2: Distribuição do Tamanho das Mensagens (recorte no P99 para evitar distorção por outliers) ---
    qtd_palavras = df_eda["qtd_palavras"]
    media = qtd_palavras.mean()
    mediana = qtd_palavras.median()
    limite_p99 = max(int(np.percentile(qtd_palavras, 99)), 5)
    dados_vis = qtd_palavras[qtd_palavras <= limite_p99]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.arange(1, limite_p99 + 2) - 0.5
    ax.hist(dados_vis, bins=bins, color="#2A9D8F", edgecolor="white", alpha=0.9)
    ax.axvline(media, color="#E76F51", linestyle="--", linewidth=2, label=f"Média: {media:.1f} palavras")
    ax.axvline(mediana, color="#264653", linestyle=":", linewidth=2, label=f"Mediana: {mediana:.0f} palavras")

    ax.set_title(f"Distribuição do Tamanho dos Tweets Após Limpeza (até P99 = {limite_p99} palavras)", fontsize=12.5, fontweight="bold", pad=14)
    ax.set_xlabel("Número de Palavras por Tweet", fontsize=11)
    ax.set_ylabel("Frequência (Nº de Tweets)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10)
    _estilizar_eixo(ax, grade_eixo="y")
    _exibir_e_salvar(fig, "02_eda_tamanho_mensagens.png", pasta_saida)

    # --- Imagens 3+: Top N Termos — uma imagem separada por classe ---
    classes_eda = sorted(df_eda["sentiment"].unique())
    paleta_termos = ["#C44E52", "#4C72B0", "#55A868", "#DD8452", "#8172B3"]

    for idx, classe in enumerate(classes_eda, start=1):
        cor = paleta_termos[(idx - 1) % len(paleta_termos)]
        textos_classe = df_eda[df_eda["sentiment"] == classe]["tweet_text_limpo"]
        df_top = obter_top_termos(textos_classe, n_termos)

        fig, ax = plt.subplots(figsize=(8.5, max(5.0, n_termos * 0.48)))
        barras = ax.barh(
            df_top["Termo"],
            df_top["Frequência"],
            height=0.65,
            color=cor,
            edgecolor="#222222",
            linewidth=0.6,
        )
        max_freq = df_top["Frequência"].max()
        ax.set_xlim(0, max_freq * 1.18)
        ax.set_title(f"Top {n_termos} Termos Mais Frequentes — Classe: {classe}", fontsize=13, fontweight="bold", pad=14)
        ax.set_xlabel("Frequência no Corpus", fontsize=11)
        ax.tick_params(axis="y", labelsize=10.5)
        _estilizar_eixo(ax, grade_eixo="x")

        for b in barras:
            val = int(b.get_width())
            ax.text(
                val + max_freq * 0.015,
                b.get_y() + b.get_height() / 2,
                f"{val:,}".replace(",", "."),
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
            )

        slug_classe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(classe).lower()).strip("_")
        _exibir_e_salvar(fig, f"03_{idx}_eda_top_termos_{slug_classe}.png", pasta_saida)


# ==============================================================================
# 4. MODELAGEM, BENCHMARKS, TUNING E SERIALIZAÇÃO (.PKL)
# ==============================================================================
def criar_split_compartilhado(
    dataframe: pd.DataFrame,
    coluna_texto_limpo: str = "tweet_text_limpo",
    coluna_alvo: str = "sentiment",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Realiza UM único split estratificado por índices, garantindo que todos os
    pipelines (ML e LeIA) sejam avaliados exatamente nos mesmos tweets.
    """
    df_unico = dataframe.drop_duplicates(subset=[coluna_texto_limpo], keep="first").reset_index(drop=True)

    idx_treino, idx_teste = train_test_split(
        df_unico.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df_unico[coluna_alvo],
    )

    df_treino = df_unico.loc[idx_treino].reset_index(drop=True)
    df_teste = df_unico.loc[idx_teste].reset_index(drop=True)

    sobreposicao = set(df_treino[coluna_texto_limpo]).intersection(set(df_teste[coluna_texto_limpo]))
    assert len(sobreposicao) == 0, f"Vazamento detectado: {len(sobreposicao)} textos em ambos os conjuntos!"

    return df_treino, df_teste


def construir_pipeline(
    nome_vetorizador: str,
    nome_classificador: str,
    params_vec: dict = None,
    params_clf: dict = None,
) -> Pipeline:
    """Monta dinamicamente um sklearn.Pipeline (vetorizador + classificador)."""
    if nome_vetorizador not in VETORIZADORES:
        raise ValueError(f"Vetorizador inválido: '{nome_vetorizador}'. Opções: {list(VETORIZADORES)}")
    if nome_classificador not in CLASSIFICADORES:
        raise ValueError(f"Classificador inválido: '{nome_classificador}'. Opções: {list(CLASSIFICADORES)}")

    vec_kwargs = dict(params_vec or {})
    if nome_vetorizador == "tfidf" and "dtype" not in vec_kwargs:
        vec_kwargs["dtype"] = np.float32  # Reduz uso de RAM pela metade sem perda prática

    vec = VETORIZADORES[nome_vetorizador](**vec_kwargs)
    clf = CLASSIFICADORES[nome_classificador](**(params_clf or {}))
    return Pipeline([("vec", vec), ("clf", clf)])


def medir_latencia_runtime_ml(
    pipeline: Pipeline,
    textos_brutos_amostra: list[str],
    stopwords_set: set[str],
    nlp_model,
    usar_lema: bool = True,
    remover_emoticons: bool = REMOVER_EMOTICONS_PRINCIPAL,
    normalizar_alongamentos: bool = NORMALIZAR_ALONGAMENTOS,
) -> dict:
    """
    Mede latência unitária (1 tweet por vez):
      - runtime_modelo_ms: apenas vetorização + predict do pipeline
      - runtime_e2e_ms: ponta a ponta (pré-processamento spaCy + pipeline)
    """
    if not textos_brutos_amostra:
        return {"runtime_modelo_ms": 0.0, "runtime_e2e_ms": 0.0}

    textos_limpos = [
        limpar_texto(
            t,
            stopwords_set,
            nlp_model,
            usar_lema=usar_lema,
            remover_emoticons=remover_emoticons,
            normalizar_alongamentos=normalizar_alongamentos,
        )
        for t in textos_brutos_amostra
    ]

    t0 = time.perf_counter()
    for txt in textos_limpos:
        pipeline.predict([txt])
    ms_apenas_modelo = ((time.perf_counter() - t0) / len(textos_limpos)) * 1000

    t0 = time.perf_counter()
    for txt in textos_brutos_amostra:
        limpo = limpar_texto(
            txt,
            stopwords_set,
            nlp_model,
            usar_lema=usar_lema,
            remover_emoticons=remover_emoticons,
            normalizar_alongamentos=normalizar_alongamentos,
        )
        pipeline.predict([limpo])
    ms_ponta_a_ponta = ((time.perf_counter() - t0) / len(textos_brutos_amostra)) * 1000

    return {
        "runtime_modelo_ms": ms_apenas_modelo,
        "runtime_e2e_ms": ms_ponta_a_ponta,
    }


def salvar_artefato_pkl(caminho: str, artefato: dict) -> float:
    """Serializa pipeline e metadados em disco via pickle e retorna o tamanho em MB."""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump(artefato, f, protocol=pickle.HIGHEST_PROTOCOL)
    return os.path.getsize(caminho) / (1024 * 1024)


def executar_suite_benchmarks(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    experimentos: list[dict],
    stopwords_set: set[str],
    nlp_model,
    usar_lema: bool = USAR_LEMA,
    remover_emoticons: bool = REMOVER_EMOTICONS_PRINCIPAL,
    normalizar_alongamentos: bool = NORMALIZAR_ALONGAMENTOS,
    pasta_modelos: str = PASTA_MODELOS,
    n_amostras_runtime: int = 100,
    incluir_leia: bool = True,
    remover_emoticons_leia: bool = True,
    cv_folds: int = CV_FOLDS,
    n_jobs: int = N_JOBS_SKLEARN,
    random_state: int = RANDOM_STATE,
):
    """
    Executa todos os pipelines declarados (com ou sem GridSearchCV),
    coleta métricas da rubrica + tempos e serializa cada modelo em .pkl.
    """
    X_treino = df_treino["tweet_text_limpo"].astype(str)
    y_treino = df_treino["sentiment"]
    X_teste = df_teste["tweet_text_limpo"].astype(str)
    y_teste = df_teste["sentiment"]

    classes = sorted(y_teste.unique())
    amostra_runtime = df_teste["tweet_text"].astype(str).iloc[:min(n_amostras_runtime, len(df_teste))].tolist()

    resultados = []
    detalhes_modelos = {}

    for exp in experimentos:
        nome = exp["nome"]
        print(f"\n>>> Executando: {nome}")

        pipe = construir_pipeline(
            nome_vetorizador=exp["vetorizador"],
            nome_classificador=exp["classificador"],
            params_vec=exp.get("params_vec"),
            params_clf=exp.get("params_clf"),
        )

        param_grid = exp.get("param_grid")

        # 1. Treinamento (com ou sem GridSearchCV exclusivamente sobre o treino)
        t0_treino = time.perf_counter()
        if param_grid:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            busca = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=n_jobs,
                pre_dispatch="2*n_jobs",
                verbose=GRIDSEARCH_VERBOSE,
                refit=True,
            )
            busca.fit(X_treino, y_treino)
            melhor_pipeline = busca.best_estimator_
            melhores_params = busca.best_params_
            f1_cv = busca.best_score_
        else:
            melhor_pipeline = pipe.fit(X_treino, y_treino)
            melhores_params = {
                **{f"vec__{k}": v for k, v in (exp.get("params_vec") or {}).items()},
                **{f"clf__{k}": v for k, v in (exp.get("params_clf") or {}).items()},
            }
            f1_cv = np.nan
        tempo_treino_s = time.perf_counter() - t0_treino

        # 2. Avaliação em lote no conjunto de teste
        t0_teste = time.perf_counter()
        y_pred = melhor_pipeline.predict(X_teste)
        tempo_teste_s = time.perf_counter() - t0_teste

        # 3. Métricas
        acc = accuracy_score(y_teste, y_pred)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_teste, y_pred, average="macro", zero_division=0)
        _, _, f1_c, _ = precision_recall_fscore_support(y_teste, y_pred, labels=classes, average=None, zero_division=0)
        cm = confusion_matrix(y_teste, y_pred, labels=classes)
        report = classification_report(y_teste, y_pred, zero_division=0)

        # 4. Latência em runtime (1 tweet por vez)
        latencias = medir_latencia_runtime_ml(
            melhor_pipeline,
            amostra_runtime,
            stopwords_set,
            nlp_model,
            usar_lema=usar_lema,
            remover_emoticons=remover_emoticons,
            normalizar_alongamentos=normalizar_alongamentos,
        )

        # 5. Serialização em .pkl com metadados completos de pré-processamento
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", nome.lower()).strip("_")
        caminho_pkl = os.path.join(pasta_modelos, f"{slug}.pkl")
        artefato = {
            "nome": nome,
            "pipeline": melhor_pipeline,
            "classes": classes,
            "hiperparametros": melhores_params,
            "stopwords": stopwords_set,
            "usar_lema": usar_lema,
            "remover_emoticons": remover_emoticons,
            "normalizar_alongamentos": normalizar_alongamentos,
            "metricas_teste": {
                "acuracia": acc,
                "precision_macro": prec_m,
                "recall_macro": rec_m,
                "f1_macro": f1_m,
            },
        }
        tamanho_mb = salvar_artefato_pkl(caminho_pkl, artefato)

        linha = {
            "Experimento": nome,
            "Vetorizador": exp["vetorizador"],
            "Classificador": exp["classificador"],
            "Acurácia": acc,
            "Precision_Macro": prec_m,
            "Recall_Macro": rec_m,
            "F1_Macro": f1_m,
            "F1_CV_Treino": f1_cv,
            "Tempo_Treino_s": tempo_treino_s,
            "Tempo_Teste_Lote_s": tempo_teste_s,
            "Runtime_Modelo_ms": latencias["runtime_modelo_ms"],
            "Runtime_E2E_ms": latencias["runtime_e2e_ms"],
            "Tamanho_PKL_MB": tamanho_mb,
            "Caminho_PKL": caminho_pkl,
        }
        for idx_c, cls in enumerate(classes):
            linha[f"F1_{cls}"] = f1_c[idx_c]

        resultados.append(linha)
        detalhes_modelos[nome] = {
            "pipeline": melhor_pipeline,
            "y_pred": y_pred,
            "confusion_matrix": cm,
            "classification_report": report,
            "melhores_params": melhores_params,
            "caminho_pkl": caminho_pkl,
        }
        print(
            f"   F1-Macro: {f1_m:.4f} | Acurácia: {acc:.4f} | "
            f"Treino: {tempo_treino_s:.2f}s | Runtime E2E: {latencias['runtime_e2e_ms']:.2f}ms | "
            f"Salvo: {caminho_pkl} ({tamanho_mb:.2f} MB)"
        )

    # 6. Baseline Léxico (LeIA) avaliado exatamente no mesmo conjunto de teste
    if incluir_leia:
        print("\n>>> Executando baseline léxico: LeIA")
        sia = SentimentIntensityAnalyzer()
        textos_leia = [
            limpar_texto_para_leia(t, remover_emoticons=remover_emoticons_leia)
            for t in df_teste["tweet_text"].astype(str)
        ]

        rotulo_pos = "Positivo" if "Positivo" in classes else classes[-1]
        rotulo_neg = "Negativo" if "Negativo" in classes else classes[0]

        t0_teste = time.perf_counter()
        scores_leia = [sia.polarity_scores(t)["compound"] for t in textos_leia]
        y_pred_leia = [rotulo_pos if s >= 0.0 else rotulo_neg for s in scores_leia]
        tempo_teste_s = time.perf_counter() - t0_teste

        qtd_neutros = sum(1 for s in scores_leia if s == 0.0)
        pct_neutros = (qtd_neutros / max(len(scores_leia), 1)) * 100

        t0_rt = time.perf_counter()
        for t in amostra_runtime:
            txt = limpar_texto_para_leia(t, remover_emoticons=remover_emoticons_leia)
            _ = rotulo_pos if sia.polarity_scores(txt)["compound"] >= 0.0 else rotulo_neg
        runtime_leia_ms = ((time.perf_counter() - t0_rt) / max(len(amostra_runtime), 1)) * 1000

        acc = accuracy_score(y_teste, y_pred_leia)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_teste, y_pred_leia, average="macro", zero_division=0)
        _, _, f1_c, _ = precision_recall_fscore_support(y_teste, y_pred_leia, labels=classes, average=None, zero_division=0)
        cm = confusion_matrix(y_teste, y_pred_leia, labels=classes)

        nome_leia = "LeIA (Léxico s/ emoticons)" if remover_emoticons_leia else "LeIA (Léxico bruto)"
        linha_leia = {
            "Experimento": nome_leia,
            "Vetorizador": "léxico",
            "Classificador": "LeIA",
            "Acurácia": acc,
            "Precision_Macro": prec_m,
            "Recall_Macro": rec_m,
            "F1_Macro": f1_m,
            "F1_CV_Treino": np.nan,
            "Tempo_Treino_s": 0.0,
            "Tempo_Teste_Lote_s": tempo_teste_s,
            "Runtime_Modelo_ms": runtime_leia_ms,
            "Runtime_E2E_ms": runtime_leia_ms,
            "Tamanho_PKL_MB": 0.0,
            "Caminho_PKL": None,
        }
        for idx_c, cls in enumerate(classes):
            linha_leia[f"F1_{cls}"] = f1_c[idx_c]

        resultados.append(linha_leia)
        detalhes_modelos[nome_leia] = {
            "pipeline": None,
            "y_pred": np.array(y_pred_leia),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_teste, y_pred_leia, zero_division=0),
            "melhores_params": {
                "remover_emoticons": remover_emoticons_leia,
                "tweets_neutros_compound_zero_pct": round(pct_neutros, 2),
            },
            "caminho_pkl": None,
        }
        print(
            f"   F1-Macro: {f1_m:.4f} | Acurácia: {acc:.4f} | Runtime: {runtime_leia_ms:.2f}ms | "
            f"Neutros (compound=0.0): {pct_neutros:.1f}%"
        )

    df_resultados = pd.DataFrame(resultados).sort_values(by="F1_Macro", ascending=False).reset_index(drop=True)
    return df_resultados, detalhes_modelos, classes


# ==============================================================================
# 5. GRÁFICOS DESACOPLADOS PARA RELATÓRIO, ERROS E INFERÊNCIA (.PKL)
# ==============================================================================
def plotar_relatorio_benchmarks(
    df_bench: pd.DataFrame,
    detalhes: dict,
    classes: list[str],
    top_n_matrizes: int = 4,
    pasta_saida: str = PASTA_GRAFICOS,
):
    """Gera cada visualização de benchmark em uma imagem individual e limpa."""
    df = df_bench.sort_values(by="F1_Macro", ascending=False).reset_index(drop=True)
    n_modelos = len(df)

    # --- Imagem 1: Ranking Principal (F1-Macro) ---
    fig, ax = plt.subplots(figsize=(10.5, max(5.0, n_modelos * 0.68)))
    cores_barras = []
    for i, row in df.iterrows():
        if row["Vetorizador"] == "léxico":
            cores_barras.append("#8C8C8C")
        elif i == 0:
            cores_barras.append("#1B9E77")
        else:
            cores_barras.append("#4C72B0")

    y_pos = np.arange(n_modelos)
    valores_f1 = df["F1_Macro"] * 100
    barras = ax.barh(y_pos, valores_f1, height=0.62, color=cores_barras, edgecolor="#222222", linewidth=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Experimento"], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, min(109, max(100, valores_f1.max() + 12)))
    ax.set_xlabel("F1-Macro (%)", fontsize=11)
    ax.set_title("Ranking Geral dos Pipelines — F1-Macro no Conjunto de Teste", fontsize=13, fontweight="bold", pad=14)
    _estilizar_eixo(ax, grade_eixo="x")

    for b, val in zip(barras, valores_f1):
        ax.text(val + 0.8, b.get_y() + b.get_height() / 2, f"{val:.2f}%", va="center", ha="left", fontsize=10, fontweight="bold")

    legenda_ranking = [
        mpatches.Patch(color="#1B9E77", label="Melhor Pipeline (ML)"),
        mpatches.Patch(color="#4C72B0", label="Pipelines Supervisionados (ML)"),
        mpatches.Patch(color="#8C8C8C", label="Baseline Léxico (LeIA)"),
    ]
    ax.legend(handles=legenda_ranking, loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC")
    _exibir_e_salvar(fig, "04_benchmark_ranking_f1_macro.png", pasta_saida)

    # --- Imagem 2: Heatmap Completo de Métricas ---
    colunas_metricas = ["Acurácia", "Precision_Macro", "Recall_Macro", "F1_Macro"]
    colunas_metricas += [f"F1_{c}" for c in classes if f"F1_{c}" in df.columns]
    rotulos_colunas = [c.replace("_Macro", "\n(Macro)").replace("F1_", "F1\n") for c in colunas_metricas]
    matriz_vals = df[colunas_metricas].values * 100

    fig, ax = plt.subplots(figsize=(max(9.0, len(colunas_metricas) * 1.45), max(5.0, n_modelos * 0.65)))
    vmin = max(0, np.nanmin(matriz_vals) - 5)
    vmax = min(100, np.nanmax(matriz_vals) + 2)
    im = ax.imshow(matriz_vals, cmap="YlGnBu", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(colunas_metricas)))
    ax.set_xticklabels(rotulos_colunas, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(n_modelos))
    ax.set_yticklabels(df["Experimento"], fontsize=10)
    ax.set_title("Comparativo Detalhado de Métricas por Pipeline (%)", fontsize=13, fontweight="bold", pad=14)

    limiar_cor = (vmin + vmax) / 2
    for i in range(matriz_vals.shape[0]):
        for j in range(matriz_vals.shape[1]):
            val = matriz_vals[i, j]
            cor_txt = "white" if val > limiar_cor else "#111111"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=cor_txt, fontsize=9.5, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Desempenho (%)", fontsize=10)
    _exibir_e_salvar(fig, "05_benchmark_heatmap_metricas.png", pasta_saida)

    # --- Imagem 3: Trade-off F1-Macro vs. Tempo de Treino ---
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    paleta_pts = plt.get_cmap("tab10").colors if n_modelos <= 10 else plt.get_cmap("tab20").colors

    tempos_pos = df["Tempo_Treino_s"][df["Tempo_Treino_s"] > 0]
    usar_symlog = len(tempos_pos) > 0 and (tempos_pos.max() / max(tempos_pos.median(), 1e-3) > 8)

    for idx, row in df.iterrows():
        cor = paleta_pts[idx % len(paleta_pts)]
        label_leg = f"[{idx + 1}] {row['Experimento']} ({row['F1_Macro']*100:.1f}% | {row['Tempo_Treino_s']:.2f}s)"
        ax.scatter(
            row["Tempo_Treino_s"],
            row["F1_Macro"] * 100,
            s=220,
            color=cor,
            edgecolor="#111111",
            linewidth=1.0,
            zorder=4,
            label=label_leg,
        )
        ax.text(
            row["Tempo_Treino_s"],
            row["F1_Macro"] * 100,
            str(idx + 1),
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            fontweight="bold",
            zorder=5,
        )

    if usar_symlog:
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("Tempo de Treinamento + Tuning (s) — Escala Simétrica Logarítmica", fontsize=11)
    else:
        ax.set_xlabel("Tempo de Treinamento + Tuning (s)", fontsize=11)

    f1_min, f1_max = (df["F1_Macro"].min() * 100), (df["F1_Macro"].max() * 100)
    margem_y = max(2.5, (f1_max - f1_min) * 0.2)
    ax.set_ylim(max(0, f1_min - margem_y), min(102, f1_max + margem_y))
    ax.set_ylabel("F1-Macro (%)", fontsize=11)
    ax.set_title("Trade-off: Desempenho (F1-Macro) vs. Custo de Treinamento", fontsize=13, fontweight="bold", pad=14)
    _estilizar_eixo(ax, grade_eixo="both")

    ax.legend(
        title="Pipelines (Ranking | F1 | Treino)",
        title_fontsize=9.5,
        fontsize=8.8,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
    )
    _exibir_e_salvar(fig, "06_benchmark_tradeoff_f1_vs_treino.png", pasta_saida)

    # --- Imagem 4: Latência Ponta a Ponta (Modelo + spaCy) ---
    df_e2e = df.sort_values(by="Runtime_E2E_ms", ascending=True).reset_index(drop=True)
    y_e2e = np.arange(len(df_e2e))
    tempo_modelo = df_e2e["Runtime_Modelo_ms"].values
    overhead_spacy = np.maximum(df_e2e["Runtime_E2E_ms"].values - tempo_modelo, 0.0)
    total_e2e = df_e2e["Runtime_E2E_ms"].values

    fig, ax = plt.subplots(figsize=(10.5, max(5.0, n_modelos * 0.65)))
    ax.barh(y_e2e, tempo_modelo, height=0.6, color="#2A9D8F", edgecolor="#222222", linewidth=0.6, label="Inferência do Modelo")
    ax.barh(y_e2e, overhead_spacy, left=tempo_modelo, height=0.6, color="#E76F51", edgecolor="#222222", linewidth=0.6, label="Pré-processamento (spaCy)")

    ax.set_yticks(y_e2e)
    ax.set_yticklabels(df_e2e["Experimento"], fontsize=10)
    ax.set_xlim(0, total_e2e.max() * 1.22 if total_e2e.max() > 0 else 1)
    ax.set_xlabel("Latência Média por Tweet (ms)", fontsize=11)
    ax.set_title("Latência Ponta a Ponta em Runtime (1 Tweet por Vez)", fontsize=13, fontweight="bold", pad=14)
    _estilizar_eixo(ax, grade_eixo="x")

    for idx_y, tot in enumerate(total_e2e):
        ax.text(tot + total_e2e.max() * 0.015, idx_y, f"{tot:.2f} ms", va="center", ha="left", fontsize=9.5, fontweight="bold")

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC")
    _exibir_e_salvar(fig, "07_benchmark_latencia_ponta_a_ponta.png", pasta_saida)

    # --- Imagem 5: Latência Exclusiva do Modelo ---
    df_mod = df.sort_values(by="Runtime_Modelo_ms", ascending=True).reset_index(drop=True)
    y_mod = np.arange(len(df_mod))
    vals_mod = df_mod["Runtime_Modelo_ms"].values

    fig, ax = plt.subplots(figsize=(10.5, max(5.0, n_modelos * 0.65)))
    barras_mod = ax.barh(y_mod, vals_mod, height=0.6, color="#2A9D8F", edgecolor="#222222", linewidth=0.6)

    ax.set_yticks(y_mod)
    ax.set_yticklabels(df_mod["Experimento"], fontsize=10)
    ax.set_xlim(0, vals_mod.max() * 1.24 if vals_mod.max() > 0 else 1)
    ax.set_xlabel("Latência Média Exclusiva do Classificador/Vetorizador (ms)", fontsize=11)
    ax.set_title("Latência Isolada do Modelo em Runtime (Sem Pré-processamento)", fontsize=13, fontweight="bold", pad=14)
    _estilizar_eixo(ax, grade_eixo="x")

    for b, val in zip(barras_mod, vals_mod):
        ax.text(val + vals_mod.max() * 0.015, b.get_y() + b.get_height() / 2, f"{val:.3f} ms", va="center", ha="left", fontsize=9.5, fontweight="bold")
    _exibir_e_salvar(fig, "08_benchmark_latencia_apenas_modelo.png", pasta_saida)

    # --- Imagens 6+: Matrizes de Confusão Individuais ---
    top_df = df.head(top_n_matrizes)
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        nome = row["Experimento"]
        cm = detalhes[nome]["confusion_matrix"]
        somas_linha = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, somas_linha, out=np.zeros_like(cm, dtype=float), where=somas_linha != 0) * 100

        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=11)
        ax.set_yticklabels(classes, fontsize=11)
        ax.set_xlabel("Classe Predita", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("Classe Real", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_title(
            f"Matriz de Confusão (#{rank}) — {nome}\n"
            f"F1-Macro: {row['F1_Macro']*100:.2f}%  |  Acurácia: {row['Acurácia']*100:.2f}%",
            fontsize=11.5,
            fontweight="bold",
            pad=14,
        )

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cor_texto = "white" if cm_pct[i, j] > 55 else "#111111"
                ax.text(
                    j, i,
                    f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)".replace(",", "."),
                    ha="center",
                    va="center",
                    color=cor_texto,
                    fontsize=11,
                    fontweight="bold",
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Taxa por Classe Real (%)", fontsize=9.5)

        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", nome.lower()).strip("_")
        _exibir_e_salvar(fig, f"09_{rank}_matriz_confusao_{slug}.png", pasta_saida)


def inspecionar_erros(
    df_teste: pd.DataFrame,
    detalhes: dict,
    nome_experimento: str,
    n: int = 10,
    pasta_saida: str = PASTA_GRAFICOS,
) -> pd.DataFrame:
    """Extrai exemplos reais classificados incorretamente e salva amostra para análise qualitativa."""
    y_pred = detalhes[nome_experimento]["y_pred"]
    df_analise = df_teste[["tweet_text", "tweet_text_limpo", "sentiment"]].copy()
    df_analise["predito"] = y_pred
    erros = df_analise[df_analise["sentiment"] != df_analise["predito"]]
    print(f"Total de erros ({nome_experimento}): {len(erros):,} de {len(df_analise):,} ({len(erros) / len(df_analise) * 100:.2f}%)".replace(",", "."))

    if pasta_saida:
        os.makedirs(pasta_saida, exist_ok=True)
        caminho_erros = os.path.join(pasta_saida, "erros_melhor_modelo.csv")
        erros.head(200).to_csv(caminho_erros, index=False, encoding="utf-8-sig")
        print(f"   [Erros salvos] {caminho_erros}")

    return erros.head(n)


def carregar_modelo_e_prever(caminho_pkl: str, texto_usuario: str, nlp_model) -> dict:
    """Carrega um artefato .pkl salvo e classifica uma frase inédita medindo latência."""
    with open(caminho_pkl, "rb") as f:
        artefato = pickle.load(f)

    t0 = time.perf_counter()
    texto_limpo = limpar_texto(
        texto_usuario,
        artefato["stopwords"],
        nlp_model,
        usar_lema=artefato.get("usar_lema", True),
        remover_emoticons=artefato.get("remover_emoticons", True),
        normalizar_alongamentos=artefato.get("normalizar_alongamentos", True),
    )
    pipeline = artefato["pipeline"]
    predicao = pipeline.predict([texto_limpo])[0]

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([texto_limpo])[0]
        confianca = round(float(np.max(probs)), 4)
    elif hasattr(pipeline, "decision_function"):
        margem = pipeline.decision_function([texto_limpo])
        confianca = f"margem={float(np.max(np.abs(margem))):.4f}"
    else:
        confianca = "N/A"

    latencia_ms = (time.perf_counter() - t0) * 1000
    return {
        "modelo": artefato["nome"],
        "texto_original": texto_usuario,
        "texto_limpo": texto_limpo,
        "sentimento_previsto": predicao,
        "confianca": confianca,
        "latencia_ms": round(latencia_ms, 2),
    }


# ==============================================================================
# 6. EXPERIMENTO DE HIPÓTESE: COM vs. SEM EMOTICONS (LEAKAGE vs. GENERALIZAÇÃO)
# ==============================================================================
def executar_experimento_hipotese_emoticons(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    stopwords_set: set[str],
    nlp_sem_emo,
    usar_lema: bool = USAR_LEMA,
    n_process: int = N_PROCESS_SPACY,
    pasta_saida: str = PASTA_GRAFICOS,
) -> pd.DataFrame:
    """Testa Target Leakage vs. Generalização real nos 3 cenários (COM->COM, SEM->SEM, COM->SEM)."""
    print("\n" + "=" * 70)
    print("=== 6. EXPERIMENTO DE HIPÓTESE: COM vs. SEM EMOTICONS ===")
    print("=" * 70)

    emo_set = {e.lower() for e in LISTA_EMOTICONS_HIPOTESE}
    nlp_com_emo = carregar_spacy(
        SPACY_MODELS["SMALL"],
        emoticons_especiais=LISTA_EMOTICONS_HIPOTESE,
        usar_lema=usar_lema,
    )

    def _limpar_lote(textos, manter_emoticons: bool):
        modelo_nlp = nlp_com_emo if manter_emoticons else nlp_sem_emo
        emo_atual = emo_set if manter_emoticons else set()
        textos_pre = [
            pre_limpar_string(
                str(t),
                remover_emoticons=not manter_emoticons,
                normalizar_alongamentos=NORMALIZAR_ALONGAMENTOS,
            )
            for t in textos
        ]
        n_proc_efetivo = 1 if len(textos_pre) < 20000 else max(1, n_process)

        return [
            " ".join(processar_tokens(doc, stopwords_set, usar_lema=usar_lema, emoticons_set=emo_atual))
            for doc in modelo_nlp.pipe(textos_pre, batch_size=1000, n_process=n_proc_efetivo)
        ]

    print("Preparando versões COM e SEM emoticons sobre o mesmo split...")
    X_tr_com = _limpar_lote(df_treino["tweet_text"], manter_emoticons=True)
    X_te_com = _limpar_lote(df_teste["tweet_text"], manter_emoticons=True)

    # Reutiliza a versão SEM emoticons já calculada no pipeline principal quando aplicável
    if REMOVER_EMOTICONS_PRINCIPAL and "tweet_text_limpo" in df_treino.columns:
        X_tr_sem = df_treino["tweet_text_limpo"].astype(str).tolist()
        X_te_sem = df_teste["tweet_text_limpo"].astype(str).tolist()
    else:
        X_tr_sem = _limpar_lote(df_treino["tweet_text"], manter_emoticons=False)
        X_te_sem = _limpar_lote(df_teste["tweet_text"], manter_emoticons=False)

    y_tr = df_treino["sentiment"]
    y_te = df_teste["sentiment"]

    modelos_teste = {
        "Reg. Logística": lambda: LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "LinearSVC": lambda: LinearSVC(C=0.5, max_iter=2000, dual="auto", class_weight="balanced", random_state=RANDOM_STATE),
        "Naive Bayes": lambda: MultinomialNB(alpha=0.5),
    }

    linhas = []
    pipelines_rl = {}

    for nome_clf, fabrica_clf in modelos_teste.items():
        pipe_com = Pipeline([
            ("vec", TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, max_features=20000, dtype=np.float32)),
            ("clf", fabrica_clf()),
        ]).fit(X_tr_com, y_tr)

        pipe_sem = Pipeline([
            ("vec", TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, max_features=20000, dtype=np.float32)),
            ("clf", fabrica_clf()),
        ]).fit(X_tr_sem, y_tr)

        if nome_clf == "Reg. Logística":
            pipelines_rl["COM"] = pipe_com
            pipelines_rl["SEM"] = pipe_sem

        cenarios = [
            ("1. Treino COM -> Teste COM", pipe_com.predict(X_te_com)),
            ("2. Treino SEM -> Teste SEM", pipe_sem.predict(X_te_sem)),
            ("3. Treino COM -> Teste SEM", pipe_com.predict(X_te_sem)),
        ]
        for cenario, pred in cenarios:
            linhas.append({
                "Modelo": nome_clf,
                "Cenário": cenario,
                "Acurácia": accuracy_score(y_te, pred),
                "F1_Macro": f1_score(y_te, pred, average="macro"),
            })

    # LeIA nos cenários com vs. sem emoticons
    sia = SentimentIntensityAnalyzer()
    classes_ord = sorted(y_te.unique())
    rot_pos = "Positivo" if "Positivo" in classes_ord else classes_ord[-1]
    rot_neg = "Negativo" if "Negativo" in classes_ord else classes_ord[0]

    pred_leia_com = [
        rot_pos if sia.polarity_scores(limpar_texto_para_leia(t, remover_emoticons=False))["compound"] >= 0 else rot_neg
        for t in df_teste["tweet_text"]
    ]
    pred_leia_sem = [
        rot_pos if sia.polarity_scores(limpar_texto_para_leia(t, remover_emoticons=True))["compound"] >= 0 else rot_neg
        for t in df_teste["tweet_text"]
    ]

    linhas.append({"Modelo": "LeIA (Léxico)", "Cenário": "1. Treino COM -> Teste COM", "Acurácia": accuracy_score(y_te, pred_leia_com), "F1_Macro": f1_score(y_te, pred_leia_com, average="macro")})
    linhas.append({"Modelo": "LeIA (Léxico)", "Cenário": "2. Treino SEM -> Teste SEM", "Acurácia": accuracy_score(y_te, pred_leia_sem), "F1_Macro": f1_score(y_te, pred_leia_sem, average="macro")})

    df_hip = pd.DataFrame(linhas)
    tabela_pivot = df_hip.pivot(index="Modelo", columns="Cenário", values="F1_Macro") * 100
    print("\n=== F1-MACRO (%) POR CENÁRIO ===")
    print(tabela_pivot.round(2).to_string())

    # Gráfico individual bem espaçado (sem rótulos 'nan%' para o LeIA no cenário 3)
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    cores_cenarios = ["#C44E52", "#1B9E77", "#DD8452"]
    tabela_pivot.plot(kind="bar", ax=ax, width=0.72, color=cores_cenarios, edgecolor="#222222", linewidth=0.7)

    ax.set_ylabel("F1-Macro (%)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylim(0, 114)
    ax.set_title("Impacto dos Emoticons: Target Leakage vs. Generalização Real", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10.5)
    ax.legend(title="Cenário Experimental", loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC")
    _estilizar_eixo(ax, grade_eixo="y")

    for container in ax.containers:
        rotulos = [f"{v:.1f}%" if np.isfinite(v) else "" for v in container.datavalues]
        ax.bar_label(container, labels=rotulos, padding=4, fontsize=8.8, fontweight="bold")

    _exibir_e_salvar(fig, "10_hipotese_emoticons_leakage_vs_generalizacao.png", pasta_saida)

    # Evidência dos pesos na Regressão Logística (guarda para problemas binários)
    def _mostrar_top_pesos(pipe, titulo, n=8):
        vec = pipe.named_steps["vec"]
        clf = pipe.named_steps["clf"]
        if len(clf.classes_) != 2:
            print(f"\n=== {titulo} (exibição de pesos simplificada apenas para classificação binária) ===")
            return
        termos = np.array(vec.get_feature_names_out())
        pesos = clf.coef_[0]
        top_neg = list(zip(termos[np.argsort(pesos)[:n]], np.sort(pesos)[:n].round(2)))
        top_pos = list(zip(termos[np.argsort(pesos)[-n:][::-1]], np.sort(pesos)[-n:][::-1].round(2)))
        print(f"\n=== {titulo} ===")
        print(f"Mais puxam para {clf.classes_[0]}: {top_neg}")
        print(f"Mais puxam para {clf.classes_[1]}: {top_pos}")

    _mostrar_top_pesos(pipelines_rl["COM"], "Pesos Reg. Logística — COM emoticons")
    _mostrar_top_pesos(pipelines_rl["SEM"], "Pesos Reg. Logística — SEM emoticons")
    return tabela_pivot


# ==============================================================================
# EXECUÇÃO PRINCIPAL (obrigatório usar __main__ no Windows por causa do spawn)
# ==============================================================================
def obter_grade_experimentos() -> list[dict]:
    """Define declarativamente os pipelines (vetorização + classificador + tuning)."""
    return [
        {
            "nome": "TF-IDF + Reg. Logística (Tuned)",
            "vetorizador": "tfidf",
            "classificador": "logistic_regression",
            "params_vec": {"token_pattern": r"\S+", "min_df": 2},
            "params_clf": {"max_iter": 1000, "random_state": RANDOM_STATE},
            "param_grid": {
                "vec__ngram_range": [(1, 1), (1, 2)],
                "vec__sublinear_tf": [False, True],
                "vec__max_features": [15000, 30000],
                "clf__C": [0.5, 1.0, 3.0],
                "clf__class_weight": [None, "balanced"],
            },
        },
        {
            "nome": "BoW (1,2) + Reg. Logística",
            "vetorizador": "bow",
            "classificador": "logistic_regression",
            "params_vec": {"token_pattern": r"\S+", "ngram_range": (1, 2), "min_df": 2, "max_features": 20000, "binary": True},
            "params_clf": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "random_state": RANDOM_STATE},
        },
        {
            "nome": "TF-IDF + LinearSVC (Tuned)",
            "vetorizador": "tfidf",
            "classificador": "linear_svc",
            "params_vec": {"token_pattern": r"\S+", "min_df": 2, "sublinear_tf": True},
            "params_clf": {"max_iter": 2000, "dual": "auto", "random_state": RANDOM_STATE},
            "param_grid": {
                "vec__ngram_range": [(1, 1), (1, 2)],
                "clf__C": [0.1, 0.5, 1.0],
                "clf__class_weight": [None, "balanced"],
            },
        },
        {
            "nome": "TF-IDF + Naive Bayes (Tuned)",
            "vetorizador": "tfidf",
            "classificador": "naive_bayes",
            "params_vec": {"token_pattern": r"\S+", "ngram_range": (1, 2), "min_df": 2, "max_features": 20000},
            "param_grid": {
                "clf__alpha": [0.1, 0.5, 1.0, 2.0],
            },
        },
        {
            "nome": "TF-IDF + ComplementNB (Tuned)",
            "vetorizador": "tfidf",
            "classificador": "complement_nb",
            "params_vec": {"token_pattern": r"\S+", "ngram_range": (1, 2), "min_df": 2, "max_features": 20000},
            "param_grid": {
                "clf__alpha": [0.1, 0.5, 1.0, 2.0],
            },
        },
        {
            "nome": "BoW (1,2) + Naive Bayes",
            "vetorizador": "bow",
            "classificador": "naive_bayes",
            "params_vec": {"token_pattern": r"\S+", "ngram_range": (1, 2), "min_df": 2, "max_features": 20000},
            "params_clf": {"alpha": 1.0},
        },
        {
            "nome": "TF-IDF Char (3,5) + LinearSVC",
            "vetorizador": "tfidf",
            "classificador": "linear_svc",
            "params_vec": {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 3, "max_features": 40000, "sublinear_tf": True},
            "params_clf": {"C": 0.5, "dual": "auto", "class_weight": "balanced", "random_state": RANDOM_STATE},
        },
    ]


def main():
    configurar_console_utf8()
    inicio_total = time.perf_counter()

    print("=== 1. CARREGANDO RECURSOS NLP E CORPUS ===")
    nlp_model = carregar_spacy(
        SPACY_MODELS["SMALL"],
        emoticons_especiais=EMOTICONS,
        usar_lema=USAR_LEMA,
    )
    stopwords_pt = carregar_stopwords_pt()
    dataset_tweets = carregar_corpus()
    print(f"Corpus original carregado: {len(dataset_tweets):,} registros".replace(",", "."))

    print("\n=== 2. PRÉ-PROCESSAMENTO E LIMPEZA ===")
    dataset_limpo = limpar_dataset(
        dataframe=dataset_tweets,
        stopwords_set=stopwords_pt,
        nlp_model=nlp_model,
        limite=LIMITE_AMOSTRA,
        usar_lema=USAR_LEMA,
        remover_emoticons=REMOVER_EMOTICONS_PRINCIPAL,
        normalizar_alongamentos=NORMALIZAR_ALONGAMENTOS,
        n_process=N_PROCESS_SPACY,
        random_state=RANDOM_STATE,
        usar_cache=USAR_CACHE_PREPROCESSAMENTO,
    )

    print("\n=== 3. ANÁLISE EXPLORATÓRIA (EDA) ===")
    plotar_analise_exploratoria(dataset_limpo, n_termos=10, pasta_saida=PASTA_GRAFICOS)

    print("\n=== 4. SPLIT COMPARTILHADO E SUITE DE BENCHMARKS ===")
    df_treino, df_teste = criar_split_compartilhado(dataset_limpo, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Total único: {len(df_treino) + len(df_teste):,} | Treino: {len(df_treino):,} | Teste: {len(df_teste):,}".replace(",", "."))

    df_bench, detalhes, classes_nomes = executar_suite_benchmarks(
        df_treino=df_treino,
        df_teste=df_teste,
        experimentos=obter_grade_experimentos(),
        stopwords_set=stopwords_pt,
        nlp_model=nlp_model,
        usar_lema=USAR_LEMA,
        remover_emoticons=REMOVER_EMOTICONS_PRINCIPAL,
        normalizar_alongamentos=NORMALIZAR_ALONGAMENTOS,
        pasta_modelos=PASTA_MODELOS,
        n_amostras_runtime=100,
        incluir_leia=True,
        remover_emoticons_leia=True,
        cv_folds=CV_FOLDS,
        n_jobs=N_JOBS_SKLEARN,
        random_state=RANDOM_STATE,
    )

    # Exporta tabela consolidada em CSV para facilitar inserção no relatório
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    caminho_csv_resumo = os.path.join(PASTA_GRAFICOS, "resultados_benchmarks.csv")
    df_bench.to_csv(caminho_csv_resumo, index=False, encoding="utf-8-sig")
    print(f"\n[Tabela salva] {caminho_csv_resumo}")

    colunas_exibicao = [
        "Experimento", "Acurácia", "Precision_Macro", "Recall_Macro", "F1_Macro",
        "F1_CV_Treino", "Tempo_Treino_s", "Tempo_Teste_Lote_s",
        "Runtime_Modelo_ms", "Runtime_E2E_ms", "Tamanho_PKL_MB",
    ]
    print("\n=== TABELA COMPARATIVA FINAL ===")
    print(df_bench[colunas_exibicao].round(4).to_string(index=False))

    melhor_nome = df_bench.iloc[0]["Experimento"]
    print(f"\n=== RELATÓRIO DETALHADO DO MELHOR MODELO: {melhor_nome} ===")
    print(f"Melhores hiperparâmetros: {detalhes[melhor_nome]['melhores_params']}\n")
    print(detalhes[melhor_nome]["classification_report"])

    print("\n=== 5. GERANDO GRÁFICOS DESACOPLADOS E ANÁLISE DE ERROS ===")
    plotar_relatorio_benchmarks(df_bench, detalhes, classes_nomes, top_n_matrizes=4, pasta_saida=PASTA_GRAFICOS)

    erros_top = inspecionar_erros(df_teste, detalhes, melhor_nome, n=10, pasta_saida=PASTA_GRAFICOS)
    print(erros_top.to_string(index=False))

    print("\n=== TESTE DE INFERÊNCIA CARREGANDO O MELHOR .PKL ===")
    melhor_pkl = df_bench.dropna(subset=["Caminho_PKL"]).iloc[0]["Caminho_PKL"]
    frases_teste = [
        "Adorei demais esse filme, não esperava que fosse tão bom!!",
        "Péssimo atendimento, esperei 2 horas e ninguém resolveu nada",
        "Excelente, o sistema caiu de novo na hora da entrega...",
    ]
    for frase in frases_teste:
        print(carregar_modelo_e_prever(melhor_pkl, frase, nlp_model))

    if RODAR_EXPERIMENTO_EMOTICONS:
        executar_experimento_hipotese_emoticons(
            df_treino=df_treino,
            df_teste=df_teste,
            stopwords_set=stopwords_pt,
            nlp_sem_emo=nlp_model,
            usar_lema=USAR_LEMA,
            n_process=N_PROCESS_SPACY,
            pasta_saida=PASTA_GRAFICOS,
        )

    duracao = time.perf_counter() - inicio_total
    print(f"\nPipeline finalizado com sucesso em {duracao:.1f}s.")
    print(f"Gráficos salvos em: {os.path.abspath(PASTA_GRAFICOS)}")
    print(f"Modelos .pkl salvos em: {os.path.abspath(PASTA_MODELOS)}")


if __name__ == "__main__":
    main()