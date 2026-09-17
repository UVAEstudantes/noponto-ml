"""CLI read-only do avaliador offline de posição corrigida NoPonto."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from urllib.parse import quote_plus

from posicao_corrigida import (
    ConfiguracaoAvaliador,
    EscritorDetalhesCsv,
    FontePostgresGeometrias,
    FontePostgresTelemetria,
    avaliar_fonte,
    criar_relatorio,
    escrever_relatorio,
)


def parse_data(valor: str) -> datetime:
    data = datetime.fromisoformat(valor.replace("Z", "+00:00"))
    if data.tzinfo is None:
        raise argparse.ArgumentTypeError("Use timestamp ISO-8601 com timezone.")
    return data


def dsn_do_ambiente() -> str:
    dsn = os.getenv("DATASET_DATABASE_URL")
    if dsn:
        return dsn
    nomes = ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    ausentes = [nome for nome in nomes if not os.getenv(nome)]
    if ausentes:
        raise RuntimeError(f"Configuração PostgreSQL ausente: {', '.join(ausentes)}")
    return (
        f"postgresql://{quote_plus(os.environ['POSTGRES_USER'])}:"
        f"{quote_plus(os.environ['POSTGRES_PASSWORD'])}@{os.environ['POSTGRES_HOST']}:"
        f"{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inicio", type=parse_data)
    parser.add_argument("--fim", type=parse_data)
    parser.add_argument("--modal")
    parser.add_argument("--linha")
    parser.add_argument("--limite", type=int)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detalhes", action="store_true", help="Grava avaliações individuais em CSV.")
    parser.add_argument("--geografia", action="store_true", help="Carrega geometrias em lote e calcula Haversine.")
    args = parser.parse_args()
    if args.inicio and args.fim and args.inicio >= args.fim:
        parser.error("--inicio deve ser anterior a --fim")

    cfg = ConfiguracaoAvaliador()
    dsn = dsn_do_ambiente()
    fonte = FontePostgresTelemetria(
        dsn, inicio=args.inicio, fim=args.fim, modal=args.modal,
        linha=args.linha, limite=args.limite,
    )
    geometrias = FontePostgresGeometrias(dsn).carregar(
        inicio=args.inicio, fim=args.fim, modal=args.modal, linha=args.linha
    ) if args.geografia else None
    detalhes = EscritorDetalhesCsv(args.output / "avaliacoes.csv" if args.detalhes else None)
    stats = avaliar_fonte(fonte, cfg, args.batch_size, detalhes, geometrias)
    parametros = {
        "inicio": args.inicio.isoformat() if args.inicio else None,
        "fim": args.fim.isoformat() if args.fim else None,
        "modal": args.modal, "linha": args.linha, "limite": args.limite,
        "batch_size": args.batch_size,
        "geografia": args.geografia,
    }
    caminhos = escrever_relatorio(criar_relatorio(stats, cfg, parametros), args.output)
    print(*caminhos, sep="\n")


if __name__ == "__main__":
    main()
