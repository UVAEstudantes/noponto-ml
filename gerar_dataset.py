"""CLI read-only para gerar o dataset estruturado v1 em Parquet particionado."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from urllib.parse import quote_plus

from dataset_v1 import (
    EscritorParquet,
    FontePostgres,
    SplitConfig,
    escrever_manifesto,
    gerar_dataset,
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
    obrigatorias = ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    ausentes = [nome for nome in obrigatorias if not os.getenv(nome)]
    if ausentes:
        raise RuntimeError(f"Configuracao PostgreSQL ausente: {', '.join(ausentes)}")
    senha = quote_plus(os.environ["POSTGRES_PASSWORD"])
    usuario = quote_plus(os.environ["POSTGRES_USER"])
    return (
        f"postgresql://{usuario}:{senha}@{os.environ['POSTGRES_HOST']}:"
        f"{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--train-ate", type=parse_data, required=True)
    parser.add_argument("--validation-ate", type=parse_data, required=True)
    parser.add_argument("--tamanho-lote", type=int, default=int(os.getenv("DATASET_BATCH_SIZE", "10000")))
    parser.add_argument("--tamanho-parte", type=int, default=int(os.getenv("DATASET_PART_SIZE", "100000")))
    args = parser.parse_args()

    config = SplitConfig(args.train_ate, args.validation_ate)
    escritor = EscritorParquet(args.saida, args.tamanho_parte)
    stats = gerar_dataset(FontePostgres(dsn_do_ambiente()), escritor, config, args.tamanho_lote)
    escrever_manifesto(args.saida / "manifest.json", stats, config, args.tamanho_lote)
    print(args.saida / "manifest.json")


if __name__ == "__main__":
    main()

