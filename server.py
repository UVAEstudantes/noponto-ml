"""
Servidor FastAPI que serve predições de ETA do modelo XGBoost treinado.

Endpoint principal:
  POST /eta
  Body: { linha, hora_dia, dia_semana, distancia_metros, velocidade_media, posicao_na_rota }
  Response: { eta_segundos, eta_minutos, confianca }

Rode com:
  uvicorn servidor:app --host 0.0.0.0 --port 5200
"""

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Carrega modelo e encoder ──────────────────────────────────────────────────

try:
    modelo      = joblib.load("modelo_eta.joblib")
    linha_map   = joblib.load("linha_encoder.joblib")
    print(f"Modelo carregado. Linhas conhecidas: {list(linha_map.keys())}")
except FileNotFoundError as e:
    print(f"ERRO: {e}")
    print("Execute treinar.py antes de iniciar o servidor.")
    raise

# ── Modelos de dados ──────────────────────────────────────────────────────────

class EtaRequest(BaseModel):
    linha:             str   = Field(..., description="Código da linha, ex: '838'")
    hora_dia:          int   = Field(..., ge=0, le=23)
    dia_semana:        int   = Field(..., ge=0, le=6, description="0=Dom, 6=Sáb")
    distancia_metros:  float = Field(..., gt=0, description="Distância até a próxima parada em metros")
    velocidade_media:  float = Field(..., default=0, ge=0, description="Velocidade média atual em km/h")
    posicao_na_rota:   float = Field(..., ge=0, le=1, description="Posição na rota (0.0 a 1.0)")

class EtaResponse(BaseModel):
    eta_segundos: float
    eta_minutos:  float
    confianca:    str   # "alta", "media", "baixa"
    linha_conhecida: bool

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NoPonto ML — ETA Service",
    description="Predição de tempo de chegada baseada em histórico real de passagens.",
    version="1.0.0",
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "linhas_disponiveis": list(linha_map.keys()),
        "total_linhas": len(linha_map),
    }

@app.post("/eta", response_model=EtaResponse)
def prever_eta(req: EtaRequest):
    linha_conhecida = req.linha.upper() in linha_map

    # Para linhas desconhecidas usa a média geral (linha_cod = -1 → usa 0 como fallback)
    # O modelo vai usar as outras features e vai dar uma estimativa razoável
    linha_cod = linha_map.get(req.linha.upper(), 0)

    # Monta o vetor de features na mesma ordem do treino
    X = np.array([[
        req.hora_dia,
        req.dia_semana,
        req.distancia_metros,
        req.velocidade_media,
        req.posicao_na_rota,
        linha_cod,
    ]])

    eta_s = float(modelo.predict(X)[0])

    # Garante que o ETA é positivo e razoável
    eta_s = max(10.0, min(eta_s, 3600.0))

    # Confiança baseada na distância — mais longe = menos certeza
    if req.distancia_metros < 300:
        confianca = "alta"
    elif req.distancia_metros < 800:
        confianca = "media"
    else:
        confianca = "baixa"

    return EtaResponse(
        eta_segundos    = round(eta_s, 1),
        eta_minutos     = round(eta_s / 60, 2),
        confianca       = confianca,
        linha_conhecida = linha_conhecida,
    )

@app.post("/eta/batch")
def prever_eta_batch(requests: list[EtaRequest]):
    if len(requests) > 500:   # aumenta limite para 500 com EnriquecerTodasLinhas
        raise HTTPException(status_code=400, detail="Máximo 500 veículos por lote.")

    if not requests:
        return []

    linhas_cod = [linha_map.get(r.linha.upper(), 0) for r in requests]

    X = np.array([
        [
            r.hora_dia,
            r.dia_semana,
            max(r.distancia_metros, 1),    # garante > 0
            max(r.velocidade_media, 0),    # garante >= 0
            min(max(r.posicao_na_rota, 0), 1),  # garante [0,1]
            lc
        ]
        for r, lc in zip(requests, linhas_cod)
    ])

    predicoes = modelo.predict(X)

    resultados = []
    for req, eta_s in zip(requests, predicoes):
        eta_s = float(max(10.0, min(eta_s, 3600.0)))

        if req.distancia_metros < 300:
            confianca = "alta"
        elif req.distancia_metros < 800:
            confianca = "media"
        else:
            confianca = "baixa"

        resultados.append({
            "eta_segundos":    round(eta_s, 1),
            "eta_minutos":     round(eta_s / 60, 2),
            "confianca":       confianca,
            "linha_conhecida": req.linha.upper() in linha_map,
        })

    return resultados