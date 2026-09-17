"""Avaliador offline causal de posicao corrigida (Etapa 4.2.1).

O modulo nao integra o runtime. A fonte PostgreSQL abre uma transacao somente
leitura e entrega lotes ordenados; o avaliador conserva em memoria apenas um
veiculo por vez e nunca usa observacoes futuras para construir previsoes.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Iterable, Iterator, Mapping, Protocol, Sequence


AVALIADOR_VERSION = "noponto-posicao-corrigida-v1"
BASELINES = (
    "B0",
    "B1",
    "B2_MEDIANA",
    "B2_RECENCIA",
    "B2_LEGACY",
    "B3_MEDIANA",
    "B3_CONSERVADOR",
    "B3_ADAPTATIVO",
)


class EstadoMovimento(StrEnum):
    MOVIMENTO = "MOVIMENTO"
    PARADO = "PARADO"
    INDETERMINADO = "INDETERMINADO"


@dataclass(frozen=True)
class ConfiguracaoAvaliador:
    horizontes_segundos: tuple[int, ...] = (10, 30, 60, 120)
    tolerancias_segundos: tuple[tuple[int, float], ...] = (
        (10, 7.5), (30, 7.5), (60, 10.0), (120, 15.0)
    )
    velocidade_maxima_kmh: float = 90.0
    janela_causal_segundos: float = 180.0
    velocidade_parado_kmh: float = 3.0
    deslocamento_parado_metros: float = 10.0
    confirmacoes_parado: int = 2
    limite_runtime_segundos: float = 30.0
    fracao_terminal: float = 0.95
    minimo_amostras_linha: int = 30
    limite_b1_adaptativo_segundos: float = 15.0
    tolerancia_comprimento_relativa: float = 1e-6
    tolerancia_comprimento_absoluta: float = 0.01
    amostra_percentis_maxima: int = 50_000

    def tolerancia(self, horizonte: int) -> float:
        valores = dict(self.tolerancias_segundos)
        if horizonte not in valores:
            raise ValueError(f"Tolerancia ausente para horizonte {horizonte}.")
        return valores[horizonte]


@dataclass(frozen=True)
class Telemetria:
    id: str
    observacao_id: str
    modal: str
    provedor: str
    ordem_veiculo: str
    codigo_linha: str
    origem_posicao: str
    latitude_recebida: float
    longitude_recebida: float
    velocidade_instantanea: float
    bearing: float | None
    timestamp_gps: datetime
    timestamp_envio_fonte: datetime | None
    timestamp_servidor_fonte: datetime | None
    recebido_em_utc: datetime
    itinerario_id: str | None
    sentido_id: str | None
    viagem_id: str | None
    posicao_na_rota: float | None
    comprimento_rota_metros: float | None
    velocidade_media_causal: float | None


@dataclass(frozen=True)
class EstadoCausal:
    velocidades: tuple[float, ...]
    velocidade_mediana: float | None
    velocidade_recencia: float | None
    estado_movimento: EstadoMovimento


@dataclass(frozen=True)
class Avaliacao:
    origem_id: str
    modal: str
    provedor: str
    ordem_veiculo: str
    codigo_linha: str
    itinerario_id: str
    viagem_id: str | None
    horizonte_nominal_segundos: int
    faixa_runtime: str
    timestamp_origem: datetime
    timestamp_alvo: datetime
    timestamp_ground_truth: datetime
    diferenca_alvo_segundos: float
    delta_real_segundos: float
    posicao_origem: float
    posicao_ground_truth: float
    comprimento_rota_metros: float
    idade_gps_segundos: float
    idade_correcao_segundos: float
    estado_movimento: str
    categoria_terminal: str
    com_viagem: str
    transicao_movimento: str
    velocidade_instantanea_kmh: float
    velocidade_causal_mediana_kmh: float | None
    velocidade_causal_recencia_kmh: float | None
    previsoes: Mapping[str, float]
    erros_metros: Mapping[str, float]
    fallbacks: Mapping[str, str]
    clamps: Mapping[str, bool]
    erros_geograficos_metros: Mapping[str, float]

    def como_dict(self) -> dict:
        linha = asdict(self)
        for campo in ("timestamp_origem", "timestamp_alvo", "timestamp_ground_truth"):
            linha[campo] = linha[campo].isoformat()
        linha["previsoes"] = dict(self.previsoes)
        linha["erros_metros"] = dict(self.erros_metros)
        linha["fallbacks"] = dict(self.fallbacks)
        return linha


def _finito(valor: float | None) -> bool:
    return valor is not None and math.isfinite(valor)


def _posicao_valida(valor: float | None) -> bool:
    return _finito(valor) and 0 <= valor <= 1


def _velocidade_valida(valor: float | None, maximo: float) -> bool:
    return _finito(valor) and 0 <= valor <= maximo


def _comprimento_compativel(a: float | None, b: float | None, cfg: ConfiguracaoAvaliador) -> bool:
    return (
        _finito(a) and _finito(b) and a > 0 and b > 0
        and math.isclose(a, b, rel_tol=cfg.tolerancia_comprimento_relativa,
                         abs_tol=cfg.tolerancia_comprimento_absoluta)
    )


def contexto_compativel(a: Telemetria, b: Telemetria, cfg: ConfiguracaoAvaliador) -> tuple[bool, str | None]:
    if (a.modal, a.provedor, a.ordem_veiculo) != (b.modal, b.provedor, b.ordem_veiculo):
        return False, "identidade_diferente"
    if a.codigo_linha != b.codigo_linha:
        return False, "linha_diferente"
    if not a.itinerario_id or a.itinerario_id != b.itinerario_id:
        return False, "itinerario_diferente"
    if a.sentido_id is not None or b.sentido_id is not None:
        if a.sentido_id != b.sentido_id:
            return False, "sentido_diferente"
    if a.viagem_id is not None or b.viagem_id is not None:
        if a.viagem_id != b.viagem_id:
            return False, "viagem_diferente"
    if not _comprimento_compativel(a.comprimento_rota_metros, b.comprimento_rota_metros, cfg):
        return False, "comprimento_incompativel"
    return True, None


def telemetria_elegivel(t: Telemetria) -> tuple[bool, str | None]:
    if t.origem_posicao.upper() != "REAL":
        return False, "origem_nao_real"
    if not t.itinerario_id:
        return False, "matching_ausente"
    if not _posicao_valida(t.posicao_na_rota):
        return False, "posicao_invalida"
    if not _finito(t.comprimento_rota_metros) or t.comprimento_rota_metros <= 0:
        return False, "comprimento_invalido"
    if t.timestamp_gps.tzinfo is None or t.recebido_em_utc.tzinfo is None:
        return False, "timestamp_sem_timezone"
    return True, None


def _velocidade_par(a: Telemetria, b: Telemetria, cfg: ConfiguracaoAvaliador) -> tuple[float | None, str | None]:
    compativel, motivo = contexto_compativel(a, b, cfg)
    if not compativel:
        return None, motivo
    dt = (b.timestamp_gps - a.timestamp_gps).total_seconds()
    if dt <= 0:
        return None, "tempo_nao_positivo"
    delta = (b.posicao_na_rota - a.posicao_na_rota) * a.comprimento_rota_metros
    if not math.isfinite(delta) or delta < 0:
        return None, "regressao_posicao"
    velocidade = delta / dt * 3.6
    if not _velocidade_valida(velocidade, cfg.velocidade_maxima_kmh):
        return None, "velocidade_causal_invalida"
    return velocidade, None


def construir_estados_causais(
    observacoes: Sequence[Telemetria], cfg: ConfiguracaoAvaliador
) -> list[EstadoCausal]:
    """Calcula estados em uma unica passagem, usando somente indices <= T."""
    estados: list[EstadoCausal] = []
    janela: deque[tuple[datetime, float]] = deque()
    sinais_parado: deque[bool] = deque(maxlen=cfg.confirmacoes_parado)
    anterior: Telemetria | None = None

    for atual in observacoes:
        elegivel, _ = telemetria_elegivel(atual)
        contexto_continua = False
        deslocamento: float | None = None
        if anterior is not None and elegivel:
            contexto_continua, _ = contexto_compativel(anterior, atual, cfg)
        if not contexto_continua:
            janela.clear()
            sinais_parado.clear()
        elif anterior is not None:
            dt = (atual.timestamp_gps - anterior.timestamp_gps).total_seconds()
            if dt > cfg.janela_causal_segundos:
                janela.clear()
                sinais_parado.clear()
                contexto_continua = False
            if dt > 0:
                deslocamento = (atual.posicao_na_rota - anterior.posicao_na_rota) * atual.comprimento_rota_metros
            if contexto_continua:
                velocidade, _ = _velocidade_par(anterior, atual, cfg)
                if velocidade is not None:
                    janela.append((atual.timestamp_gps, velocidade))

        limite = atual.timestamp_gps - timedelta(seconds=cfg.janela_causal_segundos)
        while janela and janela[0][0] < limite:
            janela.popleft()

        sinal_baixo = (
            _velocidade_valida(atual.velocidade_instantanea, cfg.velocidade_maxima_kmh)
            and atual.velocidade_instantanea < cfg.velocidade_parado_kmh
            and deslocamento is not None
            and 0 <= deslocamento <= cfg.deslocamento_parado_metros
        )
        sinais_parado.append(sinal_baixo)
        if len(sinais_parado) == cfg.confirmacoes_parado and all(sinais_parado):
            movimento = EstadoMovimento.PARADO
        elif (
            _velocidade_valida(atual.velocidade_instantanea, cfg.velocidade_maxima_kmh)
            and atual.velocidade_instantanea >= cfg.velocidade_parado_kmh
        ) or (deslocamento is not None and deslocamento > cfg.deslocamento_parado_metros):
            movimento = EstadoMovimento.MOVIMENTO
        else:
            movimento = EstadoMovimento.INDETERMINADO

        valores = tuple(v for _, v in janela)
        med = median(valores) if valores else None
        if valores:
            pesos = tuple(range(1, len(valores) + 1))
            recencia = sum(v * p for v, p in zip(valores, pesos)) / sum(pesos)
        else:
            recencia = None
        estados.append(EstadoCausal(valores, med, recencia, movimento))
        anterior = atual
    return estados


def projetar(posicao: float, comprimento: float, velocidade_kmh: float, delta_s: float) -> float:
    metros = velocidade_kmh / 3.6 * delta_s
    return min(1.0, max(posicao, posicao + metros / comprimento))


def calcular_previsoes(
    origem: Telemetria, estado: EstadoCausal, delta_s: float, cfg: ConfiguracaoAvaliador
) -> tuple[dict[str, float], dict[str, str]]:
    previsoes = {"B0": origem.posicao_na_rota}
    fallbacks: dict[str, str] = {}
    velocidades: dict[str, float | None] = {
        "B1": origem.velocidade_instantanea,
        "B2_MEDIANA": estado.velocidade_mediana,
        "B2_RECENCIA": estado.velocidade_recencia,
        "B2_LEGACY": origem.velocidade_media_causal,
    }
    validas_hibrido = [
        v for v in (origem.velocidade_instantanea, estado.velocidade_mediana)
        if _velocidade_valida(v, cfg.velocidade_maxima_kmh)
    ]
    velocidades["B3_MEDIANA"] = median(validas_hibrido) if validas_hibrido else None
    velocidades["B3_CONSERVADOR"] = min(validas_hibrido) if validas_hibrido else None
    velocidades["B3_ADAPTATIVO"] = (
        origem.velocidade_instantanea
        if delta_s <= cfg.limite_b1_adaptativo_segundos
        else (median(validas_hibrido) if validas_hibrido else None)
    )

    for nome, velocidade in velocidades.items():
        if nome.startswith("B3_") and estado.estado_movimento == EstadoMovimento.PARADO:
            velocidade = 0.0
        if not _velocidade_valida(velocidade, cfg.velocidade_maxima_kmh):
            fallbacks[nome] = "velocidade_indisponivel_ou_invalida"
            continue
        previsoes[nome] = projetar(
            origem.posicao_na_rota, origem.comprimento_rota_metros, velocidade, delta_s
        )
    return previsoes, fallbacks


def previsao_foi_clampada(origem: Telemetria, previsao: float, velocidade: float | None,
                          delta_s: float, cfg: ConfiguracaoAvaliador) -> bool:
    if not _velocidade_valida(velocidade, cfg.velocidade_maxima_kmh):
        return False
    sem_clamp = origem.posicao_na_rota + (velocidade / 3.6 * delta_s) / origem.comprimento_rota_metros
    return sem_clamp > 1.0 and previsao == 1.0


def _velocidades_baselines(origem: Telemetria, estado: EstadoCausal, delta_s: float,
                           cfg: ConfiguracaoAvaliador) -> dict[str, float | None]:
    validas = [v for v in (origem.velocidade_instantanea, estado.velocidade_mediana)
               if _velocidade_valida(v, cfg.velocidade_maxima_kmh)]
    valores = {
        "B0": 0.0,
        "B1": origem.velocidade_instantanea,
        "B2_MEDIANA": estado.velocidade_mediana,
        "B2_RECENCIA": estado.velocidade_recencia,
        "B2_LEGACY": origem.velocidade_media_causal,
        "B3_MEDIANA": median(validas) if validas else None,
        "B3_CONSERVADOR": min(validas) if validas else None,
        "B3_ADAPTATIVO": origem.velocidade_instantanea if delta_s <= cfg.limite_b1_adaptativo_segundos
        else (median(validas) if validas else None),
    }
    if estado.estado_movimento == EstadoMovimento.PARADO:
        for nome in ("B3_MEDIANA", "B3_CONSERVADOR", "B3_ADAPTATIVO"):
            valores[nome] = 0.0
    return valores


def _linhas_geojson(geojson: Mapping) -> list[list[tuple[float, float]]]:
    tipo = geojson.get("type")
    coordenadas = geojson.get("coordinates", [])
    if tipo == "LineString":
        return [[(float(p[0]), float(p[1])) for p in coordenadas]]
    if tipo == "MultiLineString":
        return [[(float(p[0]), float(p[1])) for p in linha] for linha in coordenadas]
    raise ValueError(f"Geometria não linear não suportada: {tipo}")


def _haversine_metros(a: tuple[float, float], b: tuple[float, float]) -> float:
    raio = 6_371_008.8
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * raio * math.asin(min(1.0, math.sqrt(h)))


@dataclass(frozen=True)
class GeometriaPreparada:
    segmentos: tuple[tuple[tuple[float, float], tuple[float, float], float], ...]
    finais_acumulados: tuple[float, ...]
    comprimento_metros: float


def preparar_geometria(geojson: Mapping) -> GeometriaPreparada:
    segmentos: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    total = 0.0
    for linha in _linhas_geojson(geojson):
        for a, b in zip(linha, linha[1:]):
            # ST_LineInterpolatePoint sobre geometry usa o comprimento 2D no
            # sistema de coordenadas da geometria. Haversine entra somente na
            # medição final entre os pontos já interpolados.
            tamanho = math.hypot(b[0] - a[0], b[1] - a[1])
            if tamanho > 0:
                segmentos.append((a, b, tamanho))
                total += tamanho
    if not segmentos or total <= 0:
        raise ValueError("Geometria sem segmentos válidos.")
    acumulado = 0.0
    finais = []
    for _, _, tamanho in segmentos:
        acumulado += tamanho
        finais.append(acumulado)
    return GeometriaPreparada(tuple(segmentos), tuple(finais), total)


def interpolar_geometria(geojson: Mapping | GeometriaPreparada, fracao: float) -> tuple[float, float]:
    preparada = geojson if isinstance(geojson, GeometriaPreparada) else preparar_geometria(geojson)
    alvo = min(1.0, max(0.0, fracao)) * preparada.comprimento_metros
    indice = min(bisect_left(preparada.finais_acumulados, alvo), len(preparada.segmentos) - 1)
    a, b, tamanho = preparada.segmentos[indice]
    anterior = preparada.finais_acumulados[indice - 1] if indice else 0.0
    proporcao = (alvo - anterior) / tamanho
    return a[0] + (b[0] - a[0]) * proporcao, a[1] + (b[1] - a[1]) * proporcao


def erro_geografico_metros(geojson: Mapping | GeometriaPreparada, prevista: float, real: float) -> float:
    return _haversine_metros(interpolar_geometria(geojson, prevista), interpolar_geometria(geojson, real))


def _motivo_sem_ground_truth(
    origem: Telemetria, candidatos: Sequence[Telemetria], cfg: ConfiguracaoAvaliador
) -> str:
    if not candidatos:
        return "ground_truth_ausente_na_tolerancia"
    for futuro in candidatos:
        elegivel, motivo = telemetria_elegivel(futuro)
        if not elegivel:
            return f"ground_truth_{motivo}"
        compativel, motivo = contexto_compativel(origem, futuro, cfg)
        if not compativel:
            return f"ground_truth_{motivo}"
        if futuro.posicao_na_rota < origem.posicao_na_rota:
            return "wrap_ou_regressao"
    return "ground_truth_invalido"


def selecionar_ground_truth(
    observacoes: Sequence[Telemetria], tempos: Sequence[datetime], indice: int,
    horizonte: int, cfg: ConfiguracaoAvaliador,
) -> tuple[Telemetria | None, str | None]:
    origem = observacoes[indice]
    alvo = origem.timestamp_gps + timedelta(seconds=horizonte)
    tolerancia = timedelta(seconds=cfg.tolerancia(horizonte))
    inicio = max(indice + 1, bisect_left(tempos, alvo - tolerancia, lo=indice + 1))
    fim = bisect_right(tempos, alvo + tolerancia, lo=inicio)
    candidatos = list(observacoes[inicio:fim])
    validos: list[Telemetria] = []
    for futuro in candidatos:
        elegivel, _ = telemetria_elegivel(futuro)
        compativel, _ = contexto_compativel(origem, futuro, cfg) if elegivel else (False, None)
        if compativel and futuro.posicao_na_rota >= origem.posicao_na_rota:
            validos.append(futuro)
    if not validos:
        return None, _motivo_sem_ground_truth(origem, candidatos, cfg)
    return min(validos, key=lambda t: (
        abs((t.timestamp_gps - alvo).total_seconds()), t.timestamp_gps, t.id
    )), None


class AmostraQuantil:
    """Reservatorio deterministico, exato ate o limite configurado."""

    def __init__(self, limite: int, semente: int):
        self.limite = limite
        self.total = 0
        self.valores: list[float] = []
        self._rng = random.Random(semente)

    def adicionar(self, valor: float) -> None:
        self.total += 1
        if len(self.valores) < self.limite:
            self.valores.append(valor)
            return
        indice = self._rng.randrange(self.total)
        if indice < self.limite:
            self.valores[indice] = valor

    def percentil(self, p: float) -> float | None:
        if not self.valores:
            return None
        ordenados = sorted(self.valores)
        pos = (len(ordenados) - 1) * p
        baixo, alto = math.floor(pos), math.ceil(pos)
        if baixo == alto:
            return ordenados[baixo]
        return ordenados[baixo] + (ordenados[alto] - ordenados[baixo]) * (pos - baixo)


class AcumuladorMetrica:
    def __init__(self, limite: int, chave: str):
        semente = int.from_bytes(hashlib.sha256(chave.encode()).digest()[:8], "big")
        self.n = 0
        self.soma = 0.0
        self.amostra = AmostraQuantil(limite, semente)

    def adicionar(self, erro: float) -> None:
        self.n += 1
        self.soma += erro
        self.amostra.adicionar(erro)

    def como_dict(self, origens_elegiveis: int) -> dict:
        return {
            "n": self.n,
            "cobertura": self.n / origens_elegiveis if origens_elegiveis else 0.0,
            "mae_m": self.soma / self.n if self.n else None,
            "mediana_m": self.amostra.percentil(0.5),
            "p90_m": self.amostra.percentil(0.9),
            "p95_m": self.amostra.percentil(0.95),
            "percentis_aproximados": self.amostra.total > self.amostra.limite,
        }


class EstatisticasAvaliador:
    def __init__(self, cfg: ConfiguracaoAvaliador):
        self.cfg = cfg
        self.total_lido = 0
        self.origens_elegiveis = 0
        self.avaliacoes = 0
        self.ground_truth_por_horizonte: Counter[int] = Counter()
        self.exclusoes: Counter[str] = Counter()
        self.fallbacks: Counter[str] = Counter()
        self.metricas: dict[tuple, AcumuladorMetrica] = {}
        self.metricas_geograficas: dict[tuple, AcumuladorMetrica] = {}
        self.metricas_clamp: dict[tuple, AcumuladorMetrica] = {}
        self.clamps: Counter[tuple] = Counter()
        self.denominadores: Counter[tuple] = Counter()
        self.desvios_ground_truth: dict[int, AcumuladorMetrica] = {}
        self.falsos_avancos_aparentes: Counter[tuple] = Counter()
        self.veiculos: set[tuple[str, str, str]] = set()
        self.linhas: set[str] = set()
        self.modalidades: set[str] = set()
        self.periodo_inicio: datetime | None = None
        self.periodo_fim: datetime | None = None

    def _adicionar(self, dimensao: str, valor: str, avaliacao: Avaliacao, baseline: str, erro: float) -> None:
        chave = (dimensao, valor, avaliacao.horizonte_nominal_segundos, baseline)
        if chave not in self.metricas:
            self.metricas[chave] = AcumuladorMetrica(self.cfg.amostra_percentis_maxima, repr(chave))
        self.metricas[chave].adicionar(erro)

    def adicionar_avaliacao(self, a: Avaliacao) -> None:
        self.avaliacoes += 1
        self.ground_truth_por_horizonte[a.horizonte_nominal_segundos] += 1
        if a.horizonte_nominal_segundos not in self.desvios_ground_truth:
            self.desvios_ground_truth[a.horizonte_nominal_segundos] = AcumuladorMetrica(
                self.cfg.amostra_percentis_maxima, f"desvio:{a.horizonte_nominal_segundos}"
            )
        self.desvios_ground_truth[a.horizonte_nominal_segundos].adicionar(abs(a.diferenca_alvo_segundos))
        faixa_velocidade = _faixa_velocidade(a.velocidade_instantanea_kmh)
        faixa_idade = _faixa_idade(a.idade_gps_segundos)
        dimensoes = {
            "geral": "TODOS",
            "modal": a.modal,
            "provedor": a.provedor,
            "linha": a.codigo_linha,
            "faixa_velocidade": faixa_velocidade,
            "faixa_idade_gps": faixa_idade,
            "faixa_idade_correcao": _faixa_idade_correcao(a.idade_correcao_segundos),
            "movimento": a.estado_movimento,
            "terminal": a.categoria_terminal,
            "viagem": a.com_viagem,
            "retomada": a.transicao_movimento,
            "faixa_runtime": a.faixa_runtime,
        }
        for dimensao, valor in dimensoes.items():
            self.denominadores[(dimensao, valor, a.horizonte_nominal_segundos)] += 1
        for baseline, erro in a.erros_metros.items():
            for dimensao, valor in dimensoes.items():
                self._adicionar(dimensao, valor, a, baseline, erro)
        for baseline, erro in a.erros_geograficos_metros.items():
            chave = (a.modal, a.horizonte_nominal_segundos, baseline)
            if chave not in self.metricas_geograficas:
                self.metricas_geograficas[chave] = AcumuladorMetrica(
                    self.cfg.amostra_percentis_maxima, f"geo:{chave}"
                )
            self.metricas_geograficas[chave].adicionar(erro)
        for baseline, clamp in a.clamps.items():
            if not clamp:
                continue
            chave = (a.modal, a.horizonte_nominal_segundos, baseline, a.categoria_terminal)
            self.clamps[chave] += 1
            if chave not in self.metricas_clamp:
                self.metricas_clamp[chave] = AcumuladorMetrica(
                    self.cfg.amostra_percentis_maxima, f"clamp:{chave}"
                )
            self.metricas_clamp[chave].adicionar(a.erros_metros[baseline])
        if a.estado_movimento == EstadoMovimento.PARADO.value:
            deslocamento_real = abs(a.posicao_ground_truth - a.posicao_origem) * a.comprimento_rota_metros
            if deslocamento_real <= self.cfg.deslocamento_parado_metros:
                for baseline, prevista in a.previsoes.items():
                    if prevista > a.posicao_origem:
                        self.falsos_avancos_aparentes[(a.modal, a.horizonte_nominal_segundos, baseline)] += 1
        for baseline, motivo in a.fallbacks.items():
            self.fallbacks[f"{baseline}:{motivo}"] += 1

    def como_dict(self) -> dict:
        linhas = []
        for (dimensao, valor, horizonte, baseline), metrica in sorted(self.metricas.items()):
            denominador = self.denominadores[(dimensao, valor, horizonte)]
            if dimensao == "linha" and denominador < self.cfg.minimo_amostras_linha:
                continue
            linhas.append({
                "dimensao": dimensao,
                "valor": valor,
                "horizonte_segundos": horizonte,
                "baseline": baseline,
                **metrica.como_dict(denominador),
            })
        return {
            "total_lido": self.total_lido,
            "veiculos_distintos": len(self.veiculos),
            "linhas_distintas": len(self.linhas),
            "modalidades": sorted(self.modalidades),
            "origens_elegiveis": self.origens_elegiveis,
            "avaliacoes_com_ground_truth": self.avaliacoes,
            "ground_truth_por_horizonte": {
                str(h): {
                    "n": self.ground_truth_por_horizonte[h],
                    "cobertura_sobre_origens_elegiveis": (
                        self.ground_truth_por_horizonte[h] / self.origens_elegiveis
                        if self.origens_elegiveis else 0.0
                    ),
                    "desvio_alvo_mediana_segundos": self.desvios_ground_truth[h].amostra.percentil(.5)
                    if h in self.desvios_ground_truth else None,
                    "desvio_alvo_p90_segundos": self.desvios_ground_truth[h].amostra.percentil(.9)
                    if h in self.desvios_ground_truth else None,
                }
                for h in self.cfg.horizontes_segundos
            },
            "periodo": {"inicio": _iso(self.periodo_inicio), "fim": _iso(self.periodo_fim)},
            "exclusoes": dict(sorted(self.exclusoes.items())),
            "fallbacks": dict(sorted(self.fallbacks.items())),
            "metricas": linhas,
            "metricas_geograficas": [
                {"modal": modal, "horizonte_segundos": horizonte, "baseline": baseline,
                 **metrica.como_dict(self.ground_truth_por_horizonte[horizonte])}
                for (modal, horizonte, baseline), metrica in sorted(self.metricas_geograficas.items())
            ],
            "clamps": [
                {"modal": modal, "horizonte_segundos": horizonte, "baseline": baseline,
                 "categoria_terminal": terminal, "n": self.clamps[chave],
                 "erro": self.metricas_clamp[chave].como_dict(self.clamps[chave])}
                for chave in sorted(self.clamps)
                for modal, horizonte, baseline, terminal in (chave,)
            ],
            "falsos_avancos_aparentes": [
                {"modal": modal, "horizonte_segundos": horizonte,
                 "baseline": baseline, "n": n,
                 "definicao": "classificado PARADO, deslocamento real dentro do limiar, mas previsão avançou"}
                for (modal, horizonte, baseline), n in sorted(self.falsos_avancos_aparentes.items())
            ],
        }


def _faixa_velocidade(v: float) -> str:
    if not math.isfinite(v) or v < 0:
        return "INVALIDA"
    if v < 3:
        return "0_3"
    if v < 20:
        return "3_20"
    if v < 40:
        return "20_40"
    if v <= 90:
        return "40_90"
    return "ACIMA_90"


def _faixa_idade(idade: float) -> str:
    if idade < 0:
        return "NEGATIVA"
    if idade <= 10:
        return "0_10"
    if idade <= 30:
        return "10_30"
    if idade <= 60:
        return "30_60"
    return "ACIMA_60"


def _faixa_idade_correcao(idade: float) -> str:
    if idade < 0:
        return "NEGATIVA"
    if idade <= 5:
        return "0_5"
    if idade <= 10:
        return "5_10"
    if idade <= 15:
        return "10_15"
    if idade <= 20:
        return "15_20"
    if idade <= 30:
        return "20_30"
    return "ACIMA_30_EXPERIMENTAL"


def _categoria_terminal(posicao: float) -> str:
    restante = 1.0 - posicao
    if restante <= .02:
        return "RESTANTE_ATE_2"
    if restante <= .05:
        return "RESTANTE_2_5"
    if restante <= .10:
        return "RESTANTE_5_10"
    return "RESTANTE_MAIOR_10"


class EscritorDetalhesCsv:
    def __init__(self, caminho: Path | None):
        self.caminho = caminho
        self._arquivo = None
        self._writer = None

    def escrever(self, avaliacao: Avaliacao) -> None:
        if self.caminho is None:
            return
        linha = avaliacao.como_dict()
        linha["previsoes"] = json.dumps(linha["previsoes"], sort_keys=True)
        linha["erros_metros"] = json.dumps(linha["erros_metros"], sort_keys=True)
        linha["fallbacks"] = json.dumps(linha["fallbacks"], sort_keys=True)
        if self._arquivo is None:
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
            self._arquivo = self.caminho.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._arquivo, fieldnames=list(linha))
            self._writer.writeheader()
        self._writer.writerow(linha)

    def fechar(self) -> None:
        if self._arquivo:
            self._arquivo.close()


class FonteTelemetria(Protocol):
    def lotes(self, tamanho_lote: int) -> Iterator[Sequence[Telemetria]]: ...


SQL_TELEMETRIA = r'''
SELECT
    "Id"::text AS id, "ObservacaoId" AS observacao_id, "Modal" AS modal,
    "Provedor" AS provedor, "OrdemVeiculo" AS ordem_veiculo,
    "CodigoLinha" AS codigo_linha, "OrigemPosicao" AS origem_posicao,
    "LatitudeRecebida" AS latitude_recebida,
    "LongitudeRecebida" AS longitude_recebida,
    "VelocidadeInstantanea" AS velocidade_instantanea, "Bearing" AS bearing,
    "TimestampGps" AS timestamp_gps, "TimestampEnvioFonte" AS timestamp_envio_fonte,
    "TimestampServidorFonte" AS timestamp_servidor_fonte,
    "RecebidoEmUtc" AS recebido_em_utc, "ItinerarioId"::text AS itinerario_id,
    "SentidoId"::text AS sentido_id, "ViagemId"::text AS viagem_id,
    "PosicaoNaRota" AS posicao_na_rota,
    "ComprimentoRotaMetros" AS comprimento_rota_metros,
    "VelocidadeMediaCausal" AS velocidade_media_causal
FROM "TelemetriasVeiculoMl"
WHERE (%(inicio)s IS NULL OR "TimestampGps" >= %(inicio)s)
  AND (%(fim)s IS NULL OR "TimestampGps" < %(fim)s)
  AND (%(modal)s IS NULL OR "Modal" = %(modal)s)
  AND (%(linha)s IS NULL OR "CodigoLinha" = %(linha)s)
ORDER BY "Modal", "Provedor", "OrdemVeiculo", "TimestampGps", "Id"
'''


class FontePostgresTelemetria:
    def __init__(self, dsn: str, *, inicio: datetime | None = None, fim: datetime | None = None,
                 modal: str | None = None, linha: str | None = None, limite: int | None = None):
        self.dsn = dsn
        self.parametros = {"inicio": inicio, "fim": fim, "modal": modal, "linha": linha}
        self.limite = limite

    def lotes(self, tamanho_lote: int) -> Iterator[Sequence[Telemetria]]:
        if tamanho_lote <= 0:
            raise ValueError("tamanho_lote deve ser positivo.")
        if self.limite is not None and self.limite <= 0:
            raise ValueError("limite deve ser positivo.")
        import psycopg2
        from psycopg2.extras import RealDictCursor

        consulta = SQL_TELEMETRIA + ("\nLIMIT %(limite)s" if self.limite is not None else "")
        parametros = {**self.parametros, "limite": self.limite}
        with psycopg2.connect(self.dsn) as conexao:
            conexao.set_session(readonly=True, autocommit=False)
            with conexao.cursor(name="avaliador_posicao_v1", cursor_factory=RealDictCursor) as cursor:
                cursor.itersize = tamanho_lote
                cursor.execute(consulta, parametros)
                while linhas := cursor.fetchmany(tamanho_lote):
                    yield [Telemetria(**dict(linha)) for linha in linhas]


SQL_GEOMETRIAS = r'''
SELECT i."Id"::text AS itinerario_id, ST_AsGeoJSON(i."Geometria") AS geojson
FROM "Itinerarios" i
JOIN (
    SELECT DISTINCT "ItinerarioId"
    FROM "TelemetriasVeiculoMl"
    WHERE "ItinerarioId" IS NOT NULL
      AND (%(inicio)s IS NULL OR "TimestampGps" >= %(inicio)s)
      AND (%(fim)s IS NULL OR "TimestampGps" < %(fim)s)
      AND (%(modal)s IS NULL OR "Modal" = %(modal)s)
      AND (%(linha)s IS NULL OR "CodigoLinha" = %(linha)s)
) t ON t."ItinerarioId" = i."Id"
ORDER BY i."Id"
'''


class FontePostgresGeometrias:
    """Carrega cada geometria necessária uma vez, nunca por observação."""

    def __init__(self, dsn: str):
        self.dsn = dsn

    def carregar(self, *, inicio: datetime | None = None, fim: datetime | None = None,
                 modal: str | None = None, linha: str | None = None) -> dict[str, Mapping]:
        import psycopg2

        parametros = {"inicio": inicio, "fim": fim, "modal": modal, "linha": linha}
        resultado: dict[str, Mapping] = {}
        with psycopg2.connect(self.dsn) as conexao:
            conexao.set_session(readonly=True, autocommit=False)
            with conexao.cursor() as cursor:
                cursor.execute(SQL_GEOMETRIAS, parametros)
                for itinerario_id, geojson in cursor:
                    if geojson:
                        resultado[itinerario_id] = json.loads(geojson)
        return resultado


class AvaliadorPosicao:
    def __init__(self, cfg: ConfiguracaoAvaliador, detalhes: EscritorDetalhesCsv | None = None,
                 geometrias: Mapping[str, Mapping] | None = None):
        self.cfg = cfg
        self.stats = EstatisticasAvaliador(cfg)
        self.detalhes = detalhes or EscritorDetalhesCsv(None)
        self.geometrias: dict[str, GeometriaPreparada] = {}
        for itinerario_id, geometria in (geometrias or {}).items():
            try:
                self.geometrias[itinerario_id] = preparar_geometria(geometria)
            except ValueError:
                continue
        self._chave_atual: tuple[str, str, str] | None = None
        self._grupo: list[Telemetria] = []

    def consumir_lote(self, lote: Iterable[Telemetria]) -> None:
        for t in lote:
            self.stats.total_lido += 1
            self.stats.veiculos.add((t.modal, t.provedor, t.ordem_veiculo))
            self.stats.linhas.add(t.codigo_linha)
            self.stats.modalidades.add(t.modal)
            self.stats.periodo_inicio = min(filter(None, (self.stats.periodo_inicio, t.timestamp_gps)))
            self.stats.periodo_fim = max(filter(None, (self.stats.periodo_fim, t.timestamp_gps)))
            chave = (t.modal, t.provedor, t.ordem_veiculo)
            if self._chave_atual is not None and chave != self._chave_atual:
                self._processar_grupo()
            self._chave_atual = chave
            self._grupo.append(t)

    def finalizar(self) -> EstatisticasAvaliador:
        self._processar_grupo()
        self.detalhes.fechar()
        return self.stats

    def _processar_grupo(self) -> None:
        observacoes = self._grupo
        self._grupo = []
        self._chave_atual = None
        if not observacoes:
            return
        observacoes.sort(key=lambda t: (t.timestamp_gps, t.id))
        estados = construir_estados_causais(observacoes, self.cfg)
        tempos = [t.timestamp_gps for t in observacoes]
        for indice, (origem, estado) in enumerate(zip(observacoes, estados)):
            elegivel, motivo = telemetria_elegivel(origem)
            if not elegivel:
                self.stats.exclusoes[f"origem:{motivo}"] += len(self.cfg.horizontes_segundos)
                continue
            self.stats.origens_elegiveis += 1
            for horizonte in self.cfg.horizontes_segundos:
                futuro, motivo = selecionar_ground_truth(observacoes, tempos, indice, horizonte, self.cfg)
                if futuro is None:
                    self.stats.exclusoes[motivo] += 1
                    continue
                delta = (futuro.timestamp_gps - origem.timestamp_gps).total_seconds()
                if delta <= 0:
                    self.stats.exclusoes["intervalo_temporal_invalido"] += 1
                    continue
                previsoes, fallbacks = calcular_previsoes(origem, estado, delta, self.cfg)
                velocidades = _velocidades_baselines(origem, estado, delta, self.cfg)
                erros = {
                    nome: abs(predicao - futuro.posicao_na_rota) * origem.comprimento_rota_metros
                    for nome, predicao in previsoes.items()
                }
                clamps = {
                    nome: previsao_foi_clampada(origem, predicao, velocidades.get(nome), delta, self.cfg)
                    for nome, predicao in previsoes.items()
                }
                geometria = self.geometrias.get(origem.itinerario_id)
                erros_geo: dict[str, float] = {}
                if geometria is not None:
                    for nome, predicao in previsoes.items():
                        try:
                            erros_geo[nome] = erro_geografico_metros(
                                geometria, predicao, futuro.posicao_na_rota
                            )
                        except ValueError:
                            break
                alvo = origem.timestamp_gps + timedelta(seconds=horizonte)
                idade = (origem.recebido_em_utc - origem.timestamp_gps).total_seconds()
                avaliacao = Avaliacao(
                    origem_id=origem.id, modal=origem.modal, provedor=origem.provedor,
                    ordem_veiculo=origem.ordem_veiculo, codigo_linha=origem.codigo_linha,
                    itinerario_id=origem.itinerario_id, viagem_id=origem.viagem_id,
                    horizonte_nominal_segundos=horizonte,
                    faixa_runtime="RUNTIME" if delta <= self.cfg.limite_runtime_segundos else "EXPERIMENTAL",
                    timestamp_origem=origem.timestamp_gps, timestamp_alvo=alvo,
                    timestamp_ground_truth=futuro.timestamp_gps,
                    diferenca_alvo_segundos=(futuro.timestamp_gps - alvo).total_seconds(),
                    delta_real_segundos=delta, posicao_origem=origem.posicao_na_rota,
                    posicao_ground_truth=futuro.posicao_na_rota,
                    comprimento_rota_metros=origem.comprimento_rota_metros,
                    idade_gps_segundos=idade, idade_correcao_segundos=delta,
                    estado_movimento=estado.estado_movimento.value,
                    categoria_terminal=_categoria_terminal(origem.posicao_na_rota),
                    com_viagem="COM_VIAGEM" if origem.viagem_id else "SEM_VIAGEM",
                    transicao_movimento=(
                        "RETOMADA_APOS_PARADO"
                        if indice > 0
                        and estados[indice - 1].estado_movimento == EstadoMovimento.PARADO
                        and estado.estado_movimento == EstadoMovimento.MOVIMENTO
                        else "SEM_RETOMADA"
                    ),
                    velocidade_instantanea_kmh=origem.velocidade_instantanea,
                    velocidade_causal_mediana_kmh=estado.velocidade_mediana,
                    velocidade_causal_recencia_kmh=estado.velocidade_recencia,
                    previsoes=previsoes, erros_metros=erros, fallbacks=fallbacks,
                    clamps=clamps, erros_geograficos_metros=erros_geo,
                )
                self.stats.adicionar_avaliacao(avaliacao)
                self.detalhes.escrever(avaliacao)


def avaliar_fonte(fonte: FonteTelemetria, cfg: ConfiguracaoAvaliador, tamanho_lote: int,
                  detalhes: EscritorDetalhesCsv | None = None,
                  geometrias: Mapping[str, Mapping] | None = None) -> EstatisticasAvaliador:
    avaliador = AvaliadorPosicao(cfg, detalhes, geometrias)
    for lote in fonte.lotes(tamanho_lote):
        avaliador.consumir_lote(lote)
    return avaliador.finalizar()


def criar_relatorio(stats: EstatisticasAvaliador, cfg: ConfiguracaoAvaliador,
                    parametros_fonte: Mapping[str, object], gerado_em: datetime | None = None) -> dict:
    return {
        "avaliador_version": AVALIADOR_VERSION,
        "gerado_em_utc": (gerado_em or datetime.now(timezone.utc)).isoformat(),
        "parametros": {
            **dict(parametros_fonte),
            **asdict(cfg),
            "definicoes": {
                "B0": "posicao congelada",
                "B1": "velocidade instantanea da origem",
                "B2_MEDIANA": "mediana das velocidades de progresso causais contextuais",
                "B2_RECENCIA": "media causal contextual ponderada por recencia",
                "B2_LEGACY": "VelocidadeMediaCausal persistida pelo runtime",
                "B3_MEDIANA": "mediana entre velocidade instantanea e B2_MEDIANA; zero se PARADO",
                "B3_CONSERVADOR": "menor entre velocidade instantanea e B2_MEDIANA; zero se PARADO",
            },
        },
        "resultados": stats.como_dict(),
        "metrica_geografica": {
            "status": "CALCULADA" if stats.metricas_geograficas else "NAO_SOLICITADA_OU_INDISPONIVEL",
            "metodo": "interpolacao causal da fracao na geometria em cache e Haversine",
            "consultas_por_observacao": 0,
        },
    }


def escrever_relatorio(relatorio: Mapping, output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "relatorio.json"
    md_path = output / "resumo.md"
    json_path.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    resultados = relatorio["resultados"]
    linhas = [
        "# Avaliador offline de posição corrigida",
        "",
        f"- Versão: `{relatorio['avaliador_version']}`",
        f"- Registros lidos: {resultados['total_lido']}",
        f"- Origens elegíveis: {resultados['origens_elegiveis']}",
        f"- Avaliações com ground truth: {resultados['avaliacoes_com_ground_truth']}",
        "", "## Resultado geral", "",
        "| Horizonte | Baseline | N | Cobertura | MAE m | Mediana m | P90 m | P95 m |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    gerais = [m for m in resultados["metricas"] if m["dimensao"] == "geral"]
    for m in gerais:
        linhas.append(
            f"| {m['horizonte_segundos']} | {m['baseline']} | {m['n']} | "
            f"{m['cobertura']:.4f} | {_fmt(m['mae_m'])} | {_fmt(m['mediana_m'])} | "
            f"{_fmt(m['p90_m'])} | {_fmt(m['p95_m'])} |"
        )
    linhas.extend(["", "## Exclusões", ""])
    linhas.extend(f"- `{k}`: {v}" for k, v in resultados["exclusoes"].items())
    md_path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return json_path, md_path


def _fmt(valor: float | None) -> str:
    return "-" if valor is None else f"{valor:.2f}"


def _iso(valor: datetime | None) -> str | None:
    return valor.isoformat() if valor else None
