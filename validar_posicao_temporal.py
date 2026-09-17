"""CLI da validação temporal e sensibilidade da posição corrigida."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

from avaliar_posicao_corrigida import dsn_do_ambiente, parse_data
from posicao_corrigida import ConfiguracaoAvaliador, FontePostgresGeometrias, FontePostgresTelemetria
from validacao_temporal import (
    cenarios_sensibilidade,
    criar_relatorio_consolidado,
    dividir_janelas,
    escrever_relatorio_consolidado,
    executar_janelas,
    executar_sensibilidade,
    selecionar_holdout,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inicio", type=parse_data, required=True)
    parser.add_argument("--fim", type=parse_data, required=True)
    parser.add_argument("--janela-minutos", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--modal")
    parser.add_argument("--linha")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geografia", action="store_true")
    parser.add_argument("--sensibilidade", action="store_true")
    args = parser.parse_args()
    if args.inicio >= args.fim:
        parser.error("--inicio deve ser anterior a --fim")

    dsn = dsn_do_ambiente()
    cfg = ConfiguracaoAvaliador()
    janelas = dividir_janelas(args.inicio, args.fim, timedelta(minutes=args.janela_minutos))

    def fonte(janela):
        return FontePostgresTelemetria(
            dsn, inicio=janela.inicio, fim=janela.fim, modal=args.modal, linha=args.linha
        )

    geometria_fonte = FontePostgresGeometrias(dsn)
    def geometrias(janela):
        return geometria_fonte.carregar(
            inicio=janela.inicio, fim=janela.fim, modal=args.modal, linha=args.linha
        )

    resultados = executar_janelas(
        janelas, fonte, cfg, args.batch_size, geometrias if args.geografia else None
    )
    exploracao, holdout_janela, motivo_holdout = selecionar_holdout(janelas)
    sensibilidades = []
    if args.sensibilidade and exploracao:
        sensibilidades = executar_sensibilidade(
            exploracao[0], fonte, cenarios_sensibilidade(cfg), args.batch_size
        )
    holdout = ({"janela": holdout_janela.nome, "regra": motivo_holdout}
               if holdout_janela else None)
    relatorio = criar_relatorio_consolidado(resultados, sensibilidades, holdout)
    print(*escrever_relatorio_consolidado(relatorio, args.output), sep="\n")


if __name__ == "__main__":
    main()
