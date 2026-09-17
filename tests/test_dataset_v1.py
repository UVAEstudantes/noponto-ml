from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from dataset_v1 import (
    ConstrutorSegmentos,
    EscritorParquet,
    FEATURE_COLUMNS,
    ORIGEM_ESTRUTURADA,
    Passagem,
    Split,
    SplitConfig,
    TARGET_COLUMNS,
    escrever_manifesto,
    gerar_dataset,
    validar_schema_sem_leakage,
)


UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def passagem(
    n: int,
    *,
    viagem: str | None = "v1",
    itinerario: str = "i1",
    sentido: str = "s1",
    ordem: int | None = None,
    segundos: int | None = None,
    posicao: float | None = None,
    origem: str = ORIGEM_ESTRUTURADA,
) -> Passagem:
    ts = None if segundos is None else BASE + timedelta(seconds=segundos)
    return Passagem(
        id=f"id-{n}",
        viagem_id=viagem,
        ordem_veiculo="A001",
        codigo_linha="10",
        modal="ONIBUS",
        itinerario_id=itinerario,
        sentido_id=sentido,
        parada_id=f"p{n}",
        parada_itinerario_id=f"pi{n}",
        ordem_parada=n if ordem is None else ordem,
        timestamp_passagem=ts,
        timestamp_gps=None if ts is None else ts + timedelta(seconds=2),
        posicao_na_rota=n / 10 if posicao is None else posicao,
        distancia_itinerario_metros=10_000,
        velocidade_instantanea=20,
        velocidade_media=18,
        origem_dataset=origem,
    )


def config() -> SplitConfig:
    return SplitConfig(BASE + timedelta(days=1), BASE + timedelta(days=2))


def construir(*lotes):
    c = ConstrutorSegmentos(config())
    saida = []
    for lote in lotes:
        saida.extend(c.consumir_lote(lote))
    saida.extend(c.finalizar())
    return saida, c.stats


class DatasetV1Tests(unittest.TestCase):
    def test_duas_passagens_geram_um_segmento(self):
        segmentos, _ = construir([passagem(1, segundos=0), passagem(2, segundos=60)])
        self.assertEqual(1, len(segmentos))
        self.assertEqual(60, segmentos[0].tempo_segmento_segundos)

    def test_tres_passagens_geram_dois_segmentos(self):
        segmentos, _ = construir([passagem(1, segundos=0), passagem(2, segundos=60), passagem(3, segundos=150)])
        self.assertEqual(2, len(segmentos))

    def test_viagens_diferentes_nao_conectam(self):
        segmentos, _ = construir([passagem(1, viagem="v1", segundos=0), passagem(2, viagem="v2", segundos=60)])
        self.assertEqual([], segmentos)

    def test_itinerarios_diferentes_nao_conectam(self):
        segmentos, stats = construir([passagem(1, segundos=0), passagem(2, itinerario="i2", segundos=60)])
        self.assertEqual([], segmentos)
        self.assertEqual(1, stats.motivos_rejeicao["itinerario_diferente"])

    def test_sentidos_diferentes_nao_conectam(self):
        segmentos, stats = construir([passagem(1, segundos=0), passagem(2, sentido="s2", segundos=60)])
        self.assertEqual([], segmentos)
        self.assertEqual(1, stats.motivos_rejeicao["sentido_diferente"])

    def test_tempo_nao_positivo_e_rejeitado(self):
        segmentos, stats = construir([passagem(1, segundos=60), passagem(2, segundos=0)])
        self.assertEqual([], segmentos)
        self.assertEqual(1, stats.motivos_rejeicao["tempo_nao_positivo"])

    def test_timestamp_ausente_e_rejeitado(self):
        segmentos, stats = construir([passagem(1, segundos=None), passagem(2, segundos=60)])
        self.assertEqual([], segmentos)
        self.assertEqual(1, stats.motivos_rejeicao["campo_estruturado_ausente"])

    def test_posicao_invalida_e_rejeitada(self):
        segmentos, stats = construir([passagem(1, segundos=0, posicao=-0.1), passagem(2, segundos=60)])
        self.assertEqual([], segmentos)
        self.assertEqual(1, stats.motivos_rejeicao["posicao_invalida"])

    def test_passagem_invalida_interrompe_continuidade(self):
        segmentos, _ = construir([
            passagem(1, segundos=0),
            passagem(2, segundos=60, posicao=-1),
            passagem(3, segundos=120),
        ])
        self.assertEqual([], segmentos)

    def test_ordenacao_deterministica(self):
        segmentos, _ = construir([passagem(3, segundos=120), passagem(1, segundos=0), passagem(2, segundos=60)])
        self.assertEqual([(1, 2), (2, 3)], [(s.ordem_origem, s.ordem_destino) for s in segmentos])

    def test_viagem_inteira_permanece_no_split_da_ultima_passagem(self):
        c = ConstrutorSegmentos(config())
        p1 = passagem(1, segundos=0)
        p2 = passagem(2, segundos=2 * 86400)
        segmentos = c.consumir_lote([p1, p2]) + c.finalizar()
        self.assertEqual({Split.VALIDATION.value}, {s.split for s in segmentos})

    def test_splits_sao_temporalmente_ordenados(self):
        cfg = config()
        self.assertEqual(Split.TRAIN, cfg.atribuir(BASE))
        self.assertEqual(Split.VALIDATION, cfg.atribuir(BASE + timedelta(days=1, seconds=1)))
        self.assertEqual(Split.TEST, cfg.atribuir(BASE + timedelta(days=2, seconds=1)))

    def test_target_nao_aparece_nas_features(self):
        validar_schema_sem_leakage()
        self.assertTrue(FEATURE_COLUMNS.isdisjoint(TARGET_COLUMNS))
        with self.assertRaises(ValueError):
            validar_schema_sem_leakage({"tempo_segmento_segundos"}, TARGET_COLUMNS)

    def test_lotes_nao_duplicam_e_preservam_fronteira(self):
        segmentos, _ = construir(
            [passagem(1, segundos=0), passagem(2, segundos=60)],
            [passagem(3, segundos=120)],
        )
        self.assertEqual([(1, 2), (2, 3)], [(s.ordem_origem, s.ordem_destino) for s in segmentos])

    def test_legado_nao_entra_no_dataset_estruturado(self):
        segmentos, stats = construir([
            passagem(1, viagem=None, segundos=0, origem="LEGADO"),
            passagem(2, viagem="v1", segundos=60),
        ])
        self.assertEqual([], segmentos)
        self.assertEqual(1, stats.passagens_legadas_ignoradas)

    def test_estatisticas_contabilizam_rejeicoes_e_cobertura(self):
        segmentos, stats = construir([
            passagem(1, segundos=0),
            passagem(2, segundos=60),
            passagem(3, segundos=120, posicao=-1),
        ])
        self.assertEqual(1, len(segmentos))
        self.assertEqual(1, stats.segmentos_validos)
        self.assertEqual(1, stats.motivos_rejeicao["posicao_invalida"])
        self.assertEqual(1, stats.cobertura_linha["10"])

    def test_gera_parquet_particionado_e_manifesto_com_fonte_em_lotes(self):
        class FonteSintetica:
            def lotes(self, tamanho_lote):
                self.tamanho_recebido = tamanho_lote
                yield [passagem(1, segundos=0), passagem(2, segundos=60)]
                yield [passagem(3, segundos=120)]

        import pyarrow.parquet as pq

        fonte = FonteSintetica()
        with tempfile.TemporaryDirectory() as temporario:
            saida = Path(temporario)
            stats = gerar_dataset(fonte, EscritorParquet(saida, tamanho_parte=1), config(), tamanho_lote=2)
            escrever_manifesto(saida / "manifest.json", stats, config(), 2)

            partes = sorted((saida / "split=TRAIN").glob("*.parquet"))
            self.assertEqual(2, len(partes))
            self.assertEqual(2, sum(pq.read_table(p).num_rows for p in partes))
            self.assertEqual(2, fonte.tamanho_recebido)
            manifesto = json.loads((saida / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("noponto-segmentos-v1", manifesto["dataset_version"])
            self.assertEqual(2, manifesto["estatisticas"]["segmentos_validos"])

    def test_nao_mistura_nova_geracao_com_partes_existentes(self):
        with tempfile.TemporaryDirectory() as temporario:
            saida = Path(temporario)
            pasta = saida / "split=TRAIN"
            pasta.mkdir()
            (pasta / "part-000000.parquet").touch()
            with self.assertRaises(FileExistsError):
                EscritorParquet(saida, tamanho_parte=10)


if __name__ == "__main__":
    unittest.main()
