"""Fundacao reproduzivel do dataset preditivo NoPonto (schema v1).

Este modulo e independente do treinamento legado. Ele le somente passagens
estruturadas, deriva segmentos dentro de uma viagem e grava Parquet em partes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Protocol, Sequence


DATASET_VERSION = "noponto-segmentos-v1"
FEATURE_SCHEMA_VERSION = "1"
ORIGEM_ESTRUTURADA = "ESTRUTURADO"


class Split(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class Modal(StrEnum):
    ONIBUS = "ONIBUS"
    BRT = "BRT"


# Somente valores conhecidos na origem da previsao podem ser features.
FEATURE_COLUMNS = frozenset(
    {
        "modal",
        "ordem_veiculo",
        "codigo_linha",
        "itinerario_id",
        "sentido_id",
        "parada_origem_id",
        "parada_destino_id",
        "parada_itinerario_origem_id",
        "parada_itinerario_destino_id",
        "ordem_origem",
        "ordem_destino",
        "timestamp_origem",
        "hora_dia",
        "dia_semana",
        "posicao_origem",
        "posicao_destino",
        "distancia_segmento_metros",
        "velocidade_instantanea_origem",
        "velocidade_media_origem",
    }
)

# Estes campos dependem da passagem futura e nunca podem entrar nas features.
TARGET_COLUMNS = frozenset(
    {
        "timestamp_destino",
        "tempo_segmento_segundos",
    }
)


def validar_schema_sem_leakage(
    features: Iterable[str] = FEATURE_COLUMNS,
    targets: Iterable[str] = TARGET_COLUMNS,
) -> None:
    vazamento = set(features).intersection(targets)
    if vazamento:
        raise ValueError(f"Targets presentes nas features: {sorted(vazamento)}")


@dataclass(frozen=True)
class SplitConfig:
    train_ate: datetime
    validation_ate: datetime

    def __post_init__(self) -> None:
        if self.train_ate.tzinfo is None or self.validation_ate.tzinfo is None:
            raise ValueError("Fronteiras de split devem possuir timezone.")
        if self.train_ate >= self.validation_ate:
            raise ValueError("train_ate deve ser anterior a validation_ate.")

    def atribuir(self, ancora_viagem: datetime) -> Split:
        """Atribui a viagem inteira pelo timestamp de sua ultima passagem."""
        if ancora_viagem <= self.train_ate:
            return Split.TRAIN
        if ancora_viagem <= self.validation_ate:
            return Split.VALIDATION
        return Split.TEST


@dataclass(frozen=True)
class Passagem:
    id: str
    viagem_id: str | None
    ordem_veiculo: str | None
    codigo_linha: str | None
    modal: str | None
    itinerario_id: str | None
    sentido_id: str | None
    parada_id: str | None
    parada_itinerario_id: str | None
    ordem_parada: int | None
    timestamp_passagem: datetime | None
    timestamp_gps: datetime | None
    posicao_na_rota: float | None
    distancia_itinerario_metros: float | None
    velocidade_instantanea: float | None = None
    velocidade_media: float | None = None
    origem_dataset: str = ORIGEM_ESTRUTURADA


@dataclass(frozen=True)
class Segmento:
    viagem_id: str
    ordem_veiculo: str
    codigo_linha: str
    modal: str
    itinerario_id: str
    sentido_id: str
    parada_origem_id: str
    parada_destino_id: str
    parada_itinerario_origem_id: str
    parada_itinerario_destino_id: str
    ordem_origem: int
    ordem_destino: int
    timestamp_origem: datetime
    timestamp_destino: datetime
    hora_dia: int
    dia_semana: int
    posicao_origem: float
    posicao_destino: float
    distancia_segmento_metros: float
    tempo_segmento_segundos: float
    velocidade_instantanea_origem: float | None
    velocidade_media_origem: float | None
    origem_dataset: str
    split: str

    def como_dict(self) -> dict:
        return asdict(self)


@dataclass
class EstatisticasDataset:
    total_passagens_lidas: int = 0
    passagens_estruturadas_elegiveis: int = 0
    viagens_distintas: int = 0
    segmentos_validos: int = 0
    segmentos_rejeitados: int = 0
    passagens_legadas_ignoradas: int = 0

    def __post_init__(self) -> None:
        self.motivos_rejeicao: Counter[str] = Counter()
        self.cobertura_linha: Counter[str] = Counter()
        self.cobertura_itinerario: Counter[str] = Counter()
        self.cobertura_sentido: Counter[str] = Counter()
        self.segmentos_por_split: Counter[str] = Counter()
        self.viagens_por_split: Counter[str] = Counter()
        self.periodo_inicio: datetime | None = None
        self.periodo_fim: datetime | None = None

    def rejeitar(self, motivo: str) -> None:
        self.segmentos_rejeitados += 1
        self.motivos_rejeicao[motivo] += 1

    def como_dict(self) -> dict:
        return {
            "total_passagens_lidas": self.total_passagens_lidas,
            "passagens_estruturadas_elegiveis": self.passagens_estruturadas_elegiveis,
            "passagens_legadas_ignoradas": self.passagens_legadas_ignoradas,
            "viagens_distintas": self.viagens_distintas,
            "segmentos_validos": self.segmentos_validos,
            "segmentos_rejeitados": self.segmentos_rejeitados,
            "motivos_rejeicao": dict(sorted(self.motivos_rejeicao.items())),
            "cobertura_por_linha": dict(sorted(self.cobertura_linha.items())),
            "cobertura_por_itinerario": dict(sorted(self.cobertura_itinerario.items())),
            "cobertura_por_sentido": dict(sorted(self.cobertura_sentido.items())),
            "periodo_temporal": {
                "inicio": _iso(self.periodo_inicio),
                "fim": _iso(self.periodo_fim),
            },
            "segmentos_por_split": dict(sorted(self.segmentos_por_split.items())),
            "viagens_por_split": dict(sorted(self.viagens_por_split.items())),
        }


def normalizar_modal(valor: str | None) -> Modal | None:
    if not valor:
        return None
    texto = valor.strip().upper()
    if "BRT" in texto:
        return Modal.BRT
    if texto in {"ONIBUS", "ONIBUS CONVENCIONAL", "ÔNIBUS", "BUS"}:
        return Modal.ONIBUS
    return None


def _finito_entre_zero_e_um(valor: float | None) -> bool:
    return valor is not None and math.isfinite(valor) and 0 <= valor <= 1


def _validar_passagem(p: Passagem) -> str | None:
    if p.origem_dataset != ORIGEM_ESTRUTURADA or not p.viagem_id:
        return "legado"
    obrigatorios = (
        p.ordem_veiculo,
        p.codigo_linha,
        p.itinerario_id,
        p.sentido_id,
        p.parada_id,
        p.parada_itinerario_id,
        p.ordem_parada,
        p.timestamp_passagem,
        p.timestamp_gps,
    )
    if any(v is None or v == "" for v in obrigatorios):
        return "campo_estruturado_ausente"
    if p.timestamp_passagem.tzinfo is None or p.timestamp_gps.tzinfo is None:
        return "timestamp_sem_timezone"
    if p.timestamp_passagem > p.timestamp_gps:
        return "timestamp_passagem_apos_gps"
    if not _finito_entre_zero_e_um(p.posicao_na_rota):
        return "posicao_invalida"
    if p.distancia_itinerario_metros is None or not math.isfinite(p.distancia_itinerario_metros):
        return "distancia_itinerario_invalida"
    if p.distancia_itinerario_metros <= 0:
        return "distancia_itinerario_nao_positiva"
    if normalizar_modal(p.modal) is None:
        return "modal_nao_suportado"
    return None


class ConstrutorSegmentos:
    """Processa viagens completas e mantem memoria proporcional a uma viagem."""

    def __init__(self, split_config: SplitConfig, estatisticas: EstatisticasDataset | None = None):
        validar_schema_sem_leakage()
        self.split_config = split_config
        self.stats = estatisticas or EstatisticasDataset()
        self._viagem_atual: str | None = None
        self._passagens_viagem: list[Passagem] = []

    def consumir_lote(self, passagens: Iterable[Passagem]) -> list[Segmento]:
        saida: list[Segmento] = []
        for passagem in passagens:
            self.stats.total_passagens_lidas += 1
            if self._viagem_atual is not None and passagem.viagem_id != self._viagem_atual:
                saida.extend(self._finalizar_viagem())
            if self._viagem_atual is None:
                self._viagem_atual = passagem.viagem_id
            self._passagens_viagem.append(passagem)
        return saida

    def finalizar(self) -> list[Segmento]:
        return self._finalizar_viagem()

    def _finalizar_viagem(self) -> list[Segmento]:
        passagens = self._passagens_viagem
        self._passagens_viagem = []
        self._viagem_atual = None
        if not passagens:
            return []

        # A consulta ja entrega esta ordem; ordenar novamente torna a regra
        # explicita e protege fontes sinteticas/testes contra ordem incidental.
        passagens.sort(
            key=lambda p: (
                p.ordem_parada if p.ordem_parada is not None else 2**31,
                p.timestamp_passagem or datetime.max.replace(tzinfo=timezone.utc),
                p.id,
            )
        )
        validas: list[Passagem] = []
        ids_validos: set[str] = set()
        for p in passagens:
            motivo = _validar_passagem(p)
            if motivo == "legado":
                self.stats.passagens_legadas_ignoradas += 1
                continue
            if motivo:
                self.stats.rejeitar(motivo)
                continue
            self.stats.passagens_estruturadas_elegiveis += 1
            validas.append(p)
            ids_validos.add(p.id)

        if not validas:
            return []
        self.stats.viagens_distintas += 1
        ancora = max(p.timestamp_passagem for p in validas if p.timestamp_passagem is not None)
        split = self.split_config.atribuir(ancora)
        self.stats.viagens_por_split[split.value] += 1
        self.stats.periodo_inicio = min(
            [p.timestamp_passagem for p in validas if p.timestamp_passagem is not None]
            + ([self.stats.periodo_inicio] if self.stats.periodo_inicio else [])
        )
        self.stats.periodo_fim = max(
            [p.timestamp_passagem for p in validas if p.timestamp_passagem is not None]
            + ([self.stats.periodo_fim] if self.stats.periodo_fim else [])
        )

        segmentos: list[Segmento] = []
        # Pares sao formados sobre a sequencia original. Uma passagem invalida
        # interrompe a continuidade e nunca e "pulada" para ligar suas vizinhas.
        for origem, destino in zip(passagens, passagens[1:]):
            if origem.id not in ids_validos or destino.id not in ids_validos:
                continue
            segmento, motivo = self._derivar(origem, destino, split)
            if motivo:
                self.stats.rejeitar(motivo)
                continue
            assert segmento is not None
            segmentos.append(segmento)
            self.stats.segmentos_validos += 1
            self.stats.segmentos_por_split[split.value] += 1
            self.stats.cobertura_linha[segmento.codigo_linha] += 1
            self.stats.cobertura_itinerario[segmento.itinerario_id] += 1
            self.stats.cobertura_sentido[segmento.sentido_id] += 1
        return segmentos

    @staticmethod
    def _derivar(origem: Passagem, destino: Passagem, split: Split) -> tuple[Segmento | None, str | None]:
        if origem.viagem_id != destino.viagem_id:
            return None, "viagem_diferente"
        if origem.itinerario_id != destino.itinerario_id:
            return None, "itinerario_diferente"
        if origem.sentido_id != destino.sentido_id:
            return None, "sentido_diferente"
        if origem.codigo_linha != destino.codigo_linha or origem.ordem_veiculo != destino.ordem_veiculo:
            return None, "identidade_operacional_diferente"
        if destino.ordem_parada <= origem.ordem_parada:
            return None, "ordem_nao_progressiva"
        tempo = (destino.timestamp_passagem - origem.timestamp_passagem).total_seconds()
        if tempo <= 0:
            return None, "tempo_nao_positivo"
        delta_posicao = destino.posicao_na_rota - origem.posicao_na_rota
        distancia = delta_posicao * origem.distancia_itinerario_metros
        if not math.isfinite(distancia) or distancia <= 0:
            return None, "distancia_nao_positiva"
        if not math.isclose(
            origem.distancia_itinerario_metros,
            destino.distancia_itinerario_metros,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            return None, "distancia_itinerario_inconsistente"

        modal = normalizar_modal(origem.modal)
        assert modal is not None
        ts = origem.timestamp_passagem.astimezone(timezone.utc)
        return Segmento(
            viagem_id=origem.viagem_id,
            ordem_veiculo=origem.ordem_veiculo,
            codigo_linha=origem.codigo_linha,
            modal=modal.value,
            itinerario_id=origem.itinerario_id,
            sentido_id=origem.sentido_id,
            parada_origem_id=origem.parada_id,
            parada_destino_id=destino.parada_id,
            parada_itinerario_origem_id=origem.parada_itinerario_id,
            parada_itinerario_destino_id=destino.parada_itinerario_id,
            ordem_origem=origem.ordem_parada,
            ordem_destino=destino.ordem_parada,
            timestamp_origem=ts,
            timestamp_destino=destino.timestamp_passagem.astimezone(timezone.utc),
            hora_dia=ts.hour,
            dia_semana=(ts.weekday() + 1) % 7,
            posicao_origem=origem.posicao_na_rota,
            posicao_destino=destino.posicao_na_rota,
            distancia_segmento_metros=distancia,
            tempo_segmento_segundos=tempo,
            velocidade_instantanea_origem=origem.velocidade_instantanea,
            velocidade_media_origem=origem.velocidade_media,
            origem_dataset=ORIGEM_ESTRUTURADA,
            split=split.value,
        ), None


class FontePassagens(Protocol):
    def lotes(self, tamanho_lote: int) -> Iterator[Sequence[Passagem]]: ...


SQL_PASSAGENS_ESTRUTURADAS = r'''
SELECT
    h."Id"::text AS id,
    h."ViagemId"::text AS viagem_id,
    h."Ordem" AS ordem_veiculo,
    h."CodigoLinha" AS codigo_linha,
    m."Nome" AS modal,
    h."ItinerarioId"::text AS itinerario_id,
    h."SentidoId"::text AS sentido_id,
    h."ParadaId"::text AS parada_id,
    h."ParadaItinerarioId"::text AS parada_itinerario_id,
    pi."Ordem" AS ordem_parada,
    h."TimestampPassagem" AS timestamp_passagem,
    h."TimestampGps" AS timestamp_gps,
    h."PosicaoNaRota" AS posicao_na_rota,
    i."DistanciaMetros" AS distancia_itinerario_metros,
    h."VelocidadeInstantanea" AS velocidade_instantanea,
    h."VelocidadeMedia" AS velocidade_media,
    'ESTRUTURADO' AS origem_dataset
FROM "HistoricoPassagens" h
LEFT JOIN "ParadasItinerario" pi ON pi."Id" = h."ParadaItinerarioId"
LEFT JOIN "Itinerarios" i ON i."Id" = h."ItinerarioId"
LEFT JOIN "Sentidos" s ON s."Id" = h."SentidoId"
LEFT JOIN "Linhas" l ON l."Id" = s."LinhaId"
LEFT JOIN "Modais" m ON m."Id" = l."ModalId"
WHERE h."ViagemId" IS NOT NULL
ORDER BY h."ViagemId", pi."Ordem" NULLS LAST,
         h."TimestampPassagem" NULLS LAST, h."Id"
'''


class FontePostgres:
    """Leitura read-only com cursor de servidor; sem OFFSET e sem DataFrame global."""

    def __init__(self, dsn: str):
        self.dsn = dsn

    def lotes(self, tamanho_lote: int) -> Iterator[Sequence[Passagem]]:
        if tamanho_lote <= 0:
            raise ValueError("tamanho_lote deve ser positivo.")
        import psycopg2
        from psycopg2.extras import RealDictCursor

        with psycopg2.connect(self.dsn) as conexao:
            conexao.set_session(readonly=True, autocommit=False)
            with conexao.cursor(name="dataset_v1", cursor_factory=RealDictCursor) as cursor:
                cursor.itersize = tamanho_lote
                cursor.execute(SQL_PASSAGENS_ESTRUTURADAS)
                while True:
                    linhas = cursor.fetchmany(tamanho_lote)
                    if not linhas:
                        break
                    yield [Passagem(**dict(linha)) for linha in linhas]


class EscritorParquet:
    def __init__(self, diretorio: Path, tamanho_parte: int):
        if tamanho_parte <= 0:
            raise ValueError("tamanho_parte deve ser positivo.")
        if diretorio.exists() and any(diretorio.glob("split=*/part-*.parquet")):
            raise FileExistsError(
                f"A saida {diretorio} ja contem partes Parquet; use um diretorio novo."
            )
        self.diretorio = diretorio
        self.tamanho_parte = tamanho_parte
        self._buffers: dict[str, list[dict]] = {s.value: [] for s in Split}
        self._partes: Counter[str] = Counter()

    def escrever(self, segmentos: Iterable[Segmento]) -> None:
        for segmento in segmentos:
            buffer = self._buffers[segmento.split]
            linha = segmento.como_dict()
            # O split e representado pela particao Hive (split=TRAIN etc.).
            # Nao duplicar a coluna evita conflito de schema ao ler o dataset.
            linha.pop("split")
            buffer.append(linha)
            if len(buffer) >= self.tamanho_parte:
                self._descarregar(segmento.split)

    def finalizar(self) -> None:
        for split in Split:
            self._descarregar(split.value)

    def _descarregar(self, split: str) -> None:
        linhas = self._buffers[split]
        if not linhas:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        pasta = self.diretorio / f"split={split}"
        pasta.mkdir(parents=True, exist_ok=True)
        numero = self._partes[split]
        caminho = pasta / f"part-{numero:06d}.parquet"
        pq.write_table(pa.Table.from_pylist(linhas), caminho, compression="zstd")
        self._partes[split] += 1
        linhas.clear()


def gerar_dataset(
    fonte: FontePassagens,
    escritor: EscritorParquet,
    split_config: SplitConfig,
    tamanho_lote: int,
) -> EstatisticasDataset:
    construtor = ConstrutorSegmentos(split_config)
    for lote in fonte.lotes(tamanho_lote):
        escritor.escrever(construtor.consumir_lote(lote))
    escritor.escrever(construtor.finalizar())
    escritor.finalizar()
    return construtor.stats


def escrever_manifesto(
    caminho: Path,
    stats: EstatisticasDataset,
    split_config: SplitConfig,
    tamanho_lote: int,
) -> None:
    manifesto = {
        "dataset_version": DATASET_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "origem_dataset": ORIGEM_ESTRUTURADA,
        "modais_iniciais": [Modal.ONIBUS.value, Modal.BRT.value],
        "criterios_elegibilidade": [
            "viagem estruturada",
            "identidades de itinerario/sentido/parada presentes",
            "TimestampPassagem e TimestampGps validos",
            "ordem, tempo, posicao e distancia progressivos",
        ],
        "split": {
            "regra": "viagem inteira pelo maior TimestampPassagem",
            "train_ate": split_config.train_ate.isoformat(),
            "validation_ate": split_config.validation_ate.isoformat(),
        },
        "tamanho_lote_leitura": tamanho_lote,
        "feature_columns": sorted(FEATURE_COLUMNS),
        "target_columns": sorted(TARGET_COLUMNS),
        "estatisticas": stats.como_dict(),
    }
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")


def _iso(valor: datetime | None) -> str | None:
    return valor.isoformat() if valor else None
