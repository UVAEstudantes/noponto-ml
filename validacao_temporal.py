"""Orquestracao temporal e de sensibilidade do avaliador de posicao."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Callable, Mapping

from posicao_corrigida import (
    AVALIADOR_VERSION,
    ConfiguracaoAvaliador,
    FonteTelemetria,
    EstatisticasAvaliador,
    avaliar_fonte,
)


VALIDACAO_VERSION = "noponto-validacao-temporal-v1"


@dataclass(frozen=True)
class JanelaTemporal:
    nome: str
    inicio: datetime
    fim: datetime

    def __post_init__(self) -> None:
        if self.inicio.tzinfo is None or self.fim.tzinfo is None:
            raise ValueError("Janelas precisam de timezone.")
        if self.inicio >= self.fim:
            raise ValueError("Início deve ser anterior ao fim.")


def dividir_janelas(inicio: datetime, fim: datetime, duracao: timedelta) -> list[JanelaTemporal]:
    if duracao.total_seconds() <= 0:
        raise ValueError("Duração da janela deve ser positiva.")
    janelas: list[JanelaTemporal] = []
    cursor = inicio
    numero = 1
    while cursor < fim:
        proximo = min(fim, cursor + duracao)
        janelas.append(JanelaTemporal(f"JANELA_{numero:02d}", cursor, proximo))
        cursor = proximo
        numero += 1
    return janelas


def _metricas_indice(stats: Mapping) -> dict[tuple, Mapping]:
    return {
        (m["dimensao"], m["valor"], m["horizonte_segundos"], m["baseline"]): m
        for m in stats["metricas"]
    }


def melhoria_percentual(candidato: float | None, referencia: float | None) -> float | None:
    if candidato is None or referencia is None or referencia == 0:
        return None
    return (referencia - candidato) / referencia * 100.0


def comparacoes_relativas(stats: Mapping) -> list[dict]:
    indice = _metricas_indice(stats)
    saida: list[dict] = []
    dimensoes = sorted({k[:3] for k in indice})
    for dimensao, valor, horizonte in dimensoes:
        b0 = indice.get((dimensao, valor, horizonte, "B0"))
        b1 = indice.get((dimensao, valor, horizonte, "B1"))
        if not b0 or not b1:
            continue
        for baseline in ("B2_MEDIANA", "B2_RECENCIA", "B2_LEGACY",
                         "B3_MEDIANA", "B3_CONSERVADOR", "B3_ADAPTATIVO"):
            atual = indice.get((dimensao, valor, horizonte, baseline))
            if not atual:
                continue
            saida.append({
                "dimensao": dimensao, "valor": valor, "horizonte_segundos": horizonte,
                "baseline": baseline, "n": atual["n"],
                "mae_referencia_b0_m": b0["mae_m"],
                "mae_referencia_b1_m": b1["mae_m"],
                "p95_referencia_b0_m": b0["p95_m"],
                "p95_referencia_b1_m": b1["p95_m"],
                "melhoria_mae_vs_b0_pct": melhoria_percentual(atual["mae_m"], b0["mae_m"]),
                "melhoria_p95_vs_b0_pct": melhoria_percentual(atual["p95_m"], b0["p95_m"]),
                "melhoria_mae_vs_b1_pct": melhoria_percentual(atual["mae_m"], b1["mae_m"]),
                "melhoria_p95_vs_b1_pct": melhoria_percentual(atual["p95_m"], b1["p95_m"]),
            })
    return saida


def resultado_janela(janela: JanelaTemporal, stats: EstatisticasAvaliador,
                     cfg: ConfiguracaoAvaliador) -> dict:
    dados = stats.como_dict()
    return {
        "nome": janela.nome, "inicio": janela.inicio.isoformat(), "fim": janela.fim.isoformat(),
        "configuracao": asdict(cfg), "resultados": dados,
        "comparacoes_relativas": comparacoes_relativas(dados),
    }


def executar_janelas(
    janelas: list[JanelaTemporal],
    fonte_factory: Callable[[JanelaTemporal], FonteTelemetria],
    cfg: ConfiguracaoAvaliador,
    tamanho_lote: int,
    geometrias_factory: Callable[[JanelaTemporal], Mapping[str, Mapping]] | None = None,
) -> list[dict]:
    resultados = []
    for janela in janelas:
        geometrias = geometrias_factory(janela) if geometrias_factory else None
        stats = avaliar_fonte(fonte_factory(janela), cfg, tamanho_lote, geometrias=geometrias)
        resultados.append(resultado_janela(janela, stats, cfg))
    return resultados


def cenarios_sensibilidade(base: ConfiguracaoAvaliador) -> dict[str, ConfiguracaoAvaliador]:
    cenarios = {
        f"CAUSAL_{s}s": replace(base, janela_causal_segundos=float(s))
        for s in (30, 60, 120, 180)
    }
    cenarios.update({
        "PARADA_BASE": base,
        "PARADA_V2": replace(base, velocidade_parado_kmh=2),
        "PARADA_V5": replace(base, velocidade_parado_kmh=5),
        "PARADA_D5": replace(base, deslocamento_parado_metros=5),
        "PARADA_D15": replace(base, deslocamento_parado_metros=15),
        "PARADA_C3": replace(base, confirmacoes_parado=3),
    })
    for tolerancia in (5.0, 10.0, 15.0):
        cenarios[f"GT_MAIS_MENOS_{int(tolerancia)}s"] = replace(
            base,
            tolerancias_segundos=tuple((h, tolerancia) for h in base.horizontes_segundos),
        )
    return cenarios


def executar_sensibilidade(
    janela: JanelaTemporal,
    fonte_factory: Callable[[JanelaTemporal], FonteTelemetria],
    cenarios: Mapping[str, ConfiguracaoAvaliador],
    tamanho_lote: int,
) -> list[dict]:
    saida = []
    for nome, cfg in cenarios.items():
        stats = avaliar_fonte(fonte_factory(janela), cfg, tamanho_lote)
        dados = stats.como_dict()
        saida.append({
            "cenario": nome, "janela": janela.nome, "configuracao": asdict(cfg),
            "resultados": dados, "comparacoes_relativas": comparacoes_relativas(dados),
        })
    return saida


def analisar_estabilidade(janelas: list[dict]) -> list[dict]:
    grupos: dict[tuple, list[float]] = {}
    for janela in janelas:
        for m in janela["resultados"]["metricas"]:
            if m["dimensao"] not in {"geral", "modal"} or m["mae_m"] is None:
                continue
            chave = (m["dimensao"], m["valor"], m["horizonte_segundos"], m["baseline"])
            grupos.setdefault(chave, []).append(m["mae_m"])
    return [
        {"dimensao": k[0], "valor": k[1], "horizonte_segundos": k[2], "baseline": k[3],
         "janelas": len(v), "mae_min_m": min(v), "mae_max_m": max(v),
         "amplitude_mae_m": max(v) - min(v),
         "amplitude_relativa_pct": ((max(v) - min(v)) / min(v) * 100) if min(v) else None}
        for k, v in sorted(grupos.items())
    ]


def selecionar_holdout(janelas: list[JanelaTemporal]) -> tuple[list[JanelaTemporal], JanelaTemporal | None, str]:
    dias = {j.inicio.astimezone(timezone.utc).date() for j in janelas}
    if len(janelas) >= 3 and len(dias) >= 2:
        ordenadas = sorted(janelas, key=lambda j: j.inicio)
        return ordenadas[:-1], ordenadas[-1], "janela mais recente de dia posterior preservada"
    return janelas, None, "período não cobre ao menos dois dias; holdout independente não criado"


def criar_relatorio_consolidado(janelas: list[dict], sensibilidades: list[dict],
                                holdout: Mapping | None, gerado_em: datetime | None = None) -> dict:
    return {
        "validacao_version": VALIDACAO_VERSION,
        "avaliador_version": AVALIADOR_VERSION,
        "gerado_em_utc": (gerado_em or datetime.now(timezone.utc)).isoformat(),
        "janelas": janelas,
        "estabilidade": analisar_estabilidade(janelas),
        "sensibilidades": sensibilidades,
        "holdout": holdout,
        "garantias": [
            "PostgreSQL read-only", "previsões usam somente observações <= T",
            "ground truth usa delta temporal real", "janelas reportadas separadamente",
        ],
    }


def escrever_relatorio_consolidado(relatorio: Mapping, output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "validacao-temporal.json"
    md_path = output / "validacao-temporal.md"
    json_path.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    linhas = ["# Validação temporal da posição corrigida", "", "## Por janela", ""]
    for janela in relatorio["janelas"]:
        r = janela["resultados"]
        linhas.extend([
            f"### {janela['nome']}", "",
            f"Período: `{janela['inicio']}` a `{janela['fim']}`.", "",
            f"Leituras: {r['total_lido']}; veículos: {r['veiculos_distintos']}; "
            f"linhas: {r['linhas_distintas']}; avaliações: {r['avaliacoes_com_ground_truth']}.", "",
            "| Horizonte | Baseline | N | MAE m | P95 m |", "|---:|---|---:|---:|---:|",
        ])
        for m in r["metricas"]:
            if m["dimensao"] == "geral":
                linhas.append(f"| {m['horizonte_segundos']} | {m['baseline']} | {m['n']} | "
                              f"{_fmt(m['mae_m'])} | {_fmt(m['p95_m'])} |")
        linhas.append("")
    linhas.extend([
        "## Estabilidade", "",
        "| Horizonte | Baseline | Janelas | MAE mín. | MAE máx. | Amplitude |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for e in relatorio["estabilidade"]:
        if e["dimensao"] == "geral" and e["baseline"] in {"B1", "B3_MEDIANA", "B3_ADAPTATIVO"}:
            linhas.append(f"| {e['horizonte_segundos']} | {e['baseline']} | {e['janelas']} | "
                          f"{_fmt(e['mae_min_m'])} | {_fmt(e['mae_max_m'])} | "
                          f"{_fmt(e['amplitude_mae_m'])} |")
    linhas.extend(["", "## Sensibilidade", "",
                   f"Cenários executados: {len(relatorio['sensibilidades'])}.", "",
                   "| Cenário | Horizonte | B2 mediana MAE | B2 recência MAE | B3 MAE | B3 P95 |",
                   "|---|---:|---:|---:|---:|---:|"])
    for s in relatorio["sensibilidades"]:
        gerais = {(m["horizonte_segundos"], m["baseline"]): m for m in s["resultados"]["metricas"]
                  if m["dimensao"] == "geral"}
        for h in (10, 30, 60, 120):
            b2m, b2r, b3 = (gerais.get((h, nome)) for nome in
                            ("B2_MEDIANA", "B2_RECENCIA", "B3_MEDIANA"))
            if b3:
                linhas.append(f"| {s['cenario']} | {h} | {_fmt(b2m['mae_m'] if b2m else None)} | "
                              f"{_fmt(b2r['mae_m'] if b2r else None)} | {_fmt(b3['mae_m'])} | "
                              f"{_fmt(b3['p95_m'])} |")
    linhas.extend(["", "## Modal", "",
                   "| Janela | Modal | Horizonte | B1 MAE | B3 MAE | B1 P95 | B3 P95 |",
                   "|---|---|---:|---:|---:|---:|---:|"])
    for j in relatorio["janelas"]:
        idx = {(m["valor"], m["horizonte_segundos"], m["baseline"]): m
               for m in j["resultados"]["metricas"] if m["dimensao"] == "modal"}
        for modal in ("ONIBUS", "BRT"):
            for h in (10, 30):
                b1, b3 = idx.get((modal, h, "B1")), idx.get((modal, h, "B3_MEDIANA"))
                if b1 and b3:
                    linhas.append(f"| {j['nome']} | {modal} | {h} | {_fmt(b1['mae_m'])} | "
                                  f"{_fmt(b3['mae_m'])} | {_fmt(b1['p95_m'])} | {_fmt(b3['p95_m'])} |")
    linhas.extend(["", "## Estado", "",
                   "PARADO/MOVIMENTO/INDETERMINADO e retomadas estão nas estratificações com N explícito.", "",
                   "## Terminal", "",
                   "As faixas restantes (>10%, 5–10%, 2–5%, <=2%) e erros dos clamps estão no JSON.", "",
                   "## Ground truth", "",
                   "| Janela | Horizonte | N | Cobertura | Desvio mediano | Desvio p90 |",
                   "|---|---:|---:|---:|---:|---:|"])
    for j in relatorio["janelas"]:
        for h, gt in j["resultados"]["ground_truth_por_horizonte"].items():
            linhas.append(f"| {j['nome']} | {h} | {gt['n']} | "
                          f"{gt['cobertura_sobre_origens_elegiveis']:.4f} | "
                          f"{_fmt(gt['desvio_alvo_mediana_segundos'])} | "
                          f"{_fmt(gt['desvio_alvo_p90_segundos'])} |")
    linhas.extend(["", "## Geografia", "",
                   "Métricas Haversine estão em `metricas_geograficas` quando a carga em lote é solicitada.", ""])
    if relatorio["holdout"] is None:
        linhas.extend(["## Holdout", "", "Não disponível com independência temporal suficiente.", ""])
    md_path.write_text("\n".join(linhas), encoding="utf-8")
    return json_path, md_path


def _fmt(valor: float | None) -> str:
    return "-" if valor is None else f"{valor:.2f}"
