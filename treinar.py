"""
Treina um modelo XGBoost para prever o tempo (em segundos) que um ônibus
leva para percorrer o trecho entre duas paradas consecutivas.

Features usadas:
  - hora_dia         : hora do dia (0-23)
  - dia_semana       : dia da semana (0=Dom, 6=Sáb)
  - distancia_metros : comprimento do trecho em metros
  - velocidade_media : velocidade média do veículo ao chegar na parada (km/h)
  - posicao_na_rota  : onde na rota estava o veículo (0.0 a 1.0)
  - linha_cod        : código da linha (codificado como inteiro)

Target:
  - tempo_segundos   : tempo real que levou para percorrer o trecho
"""

import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error
from xgboost import XGBRegressor
from dotenv import load_dotenv
load_dotenv()

# ── Configuração ──────────────────────────────────────────────────────────────

# Edite com as suas variáveis de ambiente ou coloque direto aqui para testar
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")

# Linhas com dados suficientes para treinar
#LINHAS_TREINO = ["838", "232", "867", "864", "397"]

# Onde salvar o modelo treinado
CAMINHO_MODELO = "modelo_eta.joblib"
CAMINHO_ENCODER = "linha_encoder.joblib"

# ── Conexão com o banco ───────────────────────────────────────────────────────

def conectar():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)

# ── Carrega e prepara os dados ────────────────────────────────────────────────

def carregar_dados(engine):
    #linhas_str = ", ".join(f"'{l}'" for l in LINHAS_TREINO)

    query = f"""
        SELECT
            h."CodigoLinha"                          AS linha,
            h."HoraDia"                              AS hora_dia,
            h."DiaSemana"                            AS dia_semana,
            h."DistanciaTrechoMetros"                AS distancia_metros,
            h."VelocidadeMedia"                      AS velocidade_media,
            h."PosicaoNaRota"                        AS posicao_na_rota,
            h."TempoDesdeParadaAnteriorSegundos"      AS tempo_segundos
        FROM "HistoricoPassagens" h
        WHERE
            h."TempoDesdeParadaAnteriorSegundos" IS NOT NULL
            AND h."DistanciaTrechoMetros" IS NOT NULL
            AND h."VelocidadeMedia" IS NOT NULL
            AND h."TempoDesdeParadaAnteriorSegundos" BETWEEN 30 AND 1200
            AND h."DistanciaTrechoMetros" BETWEEN 50 AND 8000
            AND h."VelocidadeMedia" BETWEEN 0 AND 90
        ORDER BY h."TimestampGps"
    """

    print("Carregando dados do banco...")
    df = pd.read_sql(query, engine)
    print(f"  {len(df):,} registros carregados")
    return df

def preparar_features(df):
    # Codifica linha como inteiro (Label Encoding simples)
    linhas_unicas = sorted(df["linha"].unique())
    linha_map = {l: i for i, l in enumerate(linhas_unicas)}
    df = df.copy()
    df["linha_cod"] = df["linha"].map(linha_map)

    # Features e target
    features = [
        "hora_dia",
        "dia_semana",
        "distancia_metros",
        "velocidade_media",
        "posicao_na_rota",
        "linha_cod",
    ]

    X = df[features]
    y = df["tempo_segundos"]

    return X, y, linha_map

# ── Treino ────────────────────────────────────────────────────────────────────

def treinar(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTreino: {len(X_train):,} registros | Teste: {len(X_test):,} registros")

    modelo = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,          # usa todos os núcleos disponíveis
        tree_method="hist", # mais rápido para datasets médios
    )

    print("\nTreinando modelo XGBoost...")
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── Avaliação ─────────────────────────────────────────────────────────────

    y_pred = modelo.predict(X_test)

    mae    = mean_absolute_error(y_test, y_pred)
    medae  = median_absolute_error(y_test, y_pred)
    dentro_1min = (abs(y_test - y_pred) <= 60).mean() * 100
    dentro_2min = (abs(y_test - y_pred) <= 120).mean() * 100

    print("\n── Resultados ──────────────────────────────────────")
    print(f"  MAE (erro médio absoluto)  : {mae:.1f}s  ({mae/60:.2f} min)")
    print(f"  Mediana erro absoluto      : {medae:.1f}s  ({medae/60:.2f} min)")
    print(f"  Acertos dentro de 1 min    : {dentro_1min:.1f}%")
    print(f"  Acertos dentro de 2 min    : {dentro_2min:.1f}%")
    print("────────────────────────────────────────────────────\n")

    # ── Importância das features ──────────────────────────────────────────────

    importancias = pd.Series(
        modelo.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("Importância das features:")
    for feat, imp in importancias.items():
        barra = "█" * int(imp * 40)
        print(f"  {feat:<22} {barra} {imp:.3f}")

    return modelo

# ── Salva modelo e encoder ────────────────────────────────────────────────────

def salvar(modelo, linha_map):
    joblib.dump(modelo, CAMINHO_MODELO)
    joblib.dump(linha_map, CAMINHO_ENCODER)
    print(f"\nModelo salvo em: {CAMINHO_MODELO}")
    print(f"Encoder salvo em: {CAMINHO_ENCODER}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = conectar()
    df     = carregar_dados(engine)

    print(f"\nDistribuição por linha:")
    print(df.groupby("linha")["tempo_segundos"].agg(["count", "mean", "median"])
            .rename(columns={"count": "passagens", "mean": "media_s", "median": "mediana_s"})
            .round(1))

    X, y, linha_map = preparar_features(df)
    modelo          = treinar(X, y)
    salvar(modelo, linha_map)

    print("\nEncoder de linhas (para usar no servidor):")
    for linha, cod in linha_map.items():
        print(f"  '{linha}' → {cod}")