from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from posicao_corrigida import (
    AcumuladorMetrica,
    AvaliadorPosicao,
    ConfiguracaoAvaliador,
    EstadoMovimento,
    EscritorDetalhesCsv,
    FontePostgresTelemetria,
    SQL_TELEMETRIA,
    Telemetria,
    avaliar_fonte,
    calcular_previsoes,
    construir_estados_causais,
    criar_relatorio,
    escrever_relatorio,
    projetar,
    selecionar_ground_truth,
)


UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def obs(n: int, *, segundos: float | None = None, posicao: float | None = None,
        velocidade: float = 36, itinerario: str | None = "i1", viagem: str | None = "v1",
        linha: str = "10", modal: str = "ONIBUS", provedor: str = "SPPO_ZIRIX",
        ordem: str = "A1", comprimento: float = 10_000, legado: float | None = 30,
        origem: str = "REAL", sentido: str | None = None) -> Telemetria:
    s = float(n * 10 if segundos is None else segundos)
    p = n / 100 if posicao is None else posicao
    ts = BASE + timedelta(seconds=s)
    return Telemetria(
        id=f"{n:04d}", observacao_id=f"o{n}", modal=modal, provedor=provedor,
        ordem_veiculo=ordem, codigo_linha=linha, origem_posicao=origem,
        latitude_recebida=-22.9, longitude_recebida=-43.2,
        velocidade_instantanea=velocidade, bearing=90, timestamp_gps=ts,
        timestamp_envio_fonte=None, timestamp_servidor_fonte=None,
        recebido_em_utc=ts + timedelta(seconds=15), itinerario_id=itinerario,
        sentido_id=sentido, viagem_id=viagem, posicao_na_rota=p,
        comprimento_rota_metros=comprimento, velocidade_media_causal=legado,
    )


def cfg(**kwargs) -> ConfiguracaoAvaliador:
    base = dict(
        horizontes_segundos=(10,), tolerancias_segundos=((10, 2.0),),
        janela_causal_segundos=180, amostra_percentis_maxima=100,
    )
    base.update(kwargs)
    return ConfiguracaoAvaliador(**base)


class FonteLista:
    def __init__(self, lotes):
        self._lotes = lotes
        self.tamanho = None

    def lotes(self, tamanho_lote):
        self.tamanho = tamanho_lote
        yield from self._lotes


class PosicaoCorrigidaTests(unittest.TestCase):
    def test_b0_congela_posicao(self):
        estados = construir_estados_causais([obs(0, posicao=.2)], cfg())
        previsoes, _ = calcular_previsoes(obs(0, posicao=.2), estados[0], 10, cfg())
        self.assertEqual(.2, previsoes["B0"])

    def test_b1_usa_velocidade_instantanea_e_delta_real(self):
        origem = obs(0, posicao=.2, velocidade=36, comprimento=1000)
        estado = construir_estados_causais([origem], cfg())[0]
        previsoes, _ = calcular_previsoes(origem, estado, 12, cfg())
        self.assertAlmostEqual(.32, previsoes["B1"])

    def test_b2_usa_somente_passado(self):
        dados = [obs(0, segundos=0, posicao=.10), obs(1, segundos=10, posicao=.11)]
        estados = construir_estados_causais(dados, cfg())
        self.assertIsNone(estados[0].velocidade_mediana)
        self.assertAlmostEqual(36, estados[1].velocidade_mediana)

    def test_b2_reinicia_em_troca_itinerario(self):
        dados = [obs(0, posicao=.1), obs(1, posicao=.11), obs(2, posicao=.12, itinerario="i2")]
        self.assertIsNone(construir_estados_causais(dados, cfg())[2].velocidade_mediana)

    def test_b2_reinicia_em_troca_viagem(self):
        dados = [obs(0, posicao=.1), obs(1, posicao=.11), obs(2, posicao=.12, viagem="v2")]
        self.assertIsNone(construir_estados_causais(dados, cfg())[2].velocidade_mediana)

    def test_b2_reinicia_em_troca_linha(self):
        dados = [obs(0, posicao=.1), obs(1, posicao=.11, linha="20")]
        self.assertIsNone(construir_estados_causais(dados, cfg())[1].velocidade_mediana)

    def test_b2_reinicia_apos_intervalo_maior_que_janela(self):
        dados = [obs(0, segundos=0, posicao=.1), obs(1, segundos=10, posicao=.11),
                 obs(2, segundos=400, posicao=.2)]
        self.assertIsNone(construir_estados_causais(dados, cfg())[2].velocidade_mediana)

    def test_b2_rejeita_regressao(self):
        estados = construir_estados_causais([obs(0, posicao=.2), obs(1, posicao=.1)], cfg())
        self.assertIsNone(estados[1].velocidade_mediana)

    def test_b2_rejeita_velocidade_acima_90(self):
        estados = construir_estados_causais(
            [obs(0, segundos=0, posicao=.1), obs(1, segundos=1, posicao=.2)], cfg()
        )
        self.assertIsNone(estados[1].velocidade_mediana)

    def test_media_ponderada_da_mais_peso_ao_recente(self):
        dados = [obs(0, segundos=0, posicao=0), obs(1, segundos=10, posicao=.005),
                 obs(2, segundos=20, posicao=.025)]
        estado = construir_estados_causais(dados, cfg())[2]
        self.assertGreater(estado.velocidade_recencia, estado.velocidade_mediana)

    def test_parada_usa_somente_passado(self):
        dados = [obs(0, posicao=.2, velocidade=0), obs(1, posicao=.2, velocidade=0),
                 obs(2, posicao=.2, velocidade=0), obs(3, posicao=.8, velocidade=80)]
        estados = construir_estados_causais(dados, cfg())
        self.assertEqual(EstadoMovimento.PARADO, estados[2].estado_movimento)

    def test_retomada_movimento(self):
        dados = [obs(0, posicao=.2, velocidade=0), obs(1, posicao=.2, velocidade=0),
                 obs(2, posicao=.21, velocidade=36)]
        self.assertEqual(EstadoMovimento.MOVIMENTO, construir_estados_causais(dados, cfg())[2].estado_movimento)

    def test_b3_parado_projeta_zero(self):
        dados = [obs(0, posicao=.2, velocidade=0), obs(1, posicao=.2, velocidade=0),
                 obs(2, posicao=.2, velocidade=0)]
        estado = construir_estados_causais(dados, cfg())[2]
        previsoes, _ = calcular_previsoes(dados[2], estado, 30, cfg())
        self.assertEqual(.2, previsoes["B3_MEDIANA"])
        self.assertEqual(.2, previsoes["B3_CONSERVADOR"])

    def test_b3_sem_leakage_quando_futuro_muda(self):
        passado = [obs(0, posicao=.1), obs(1, posicao=.11)]
        a = construir_estados_causais(passado + [obs(2, posicao=.12)], cfg())[1]
        b = construir_estados_causais(passado + [obs(2, posicao=.9)], cfg())[1]
        self.assertEqual(a, b)
        self.assertEqual(
            calcular_previsoes(passado[1], a, 10, cfg()),
            calcular_previsoes(passado[1], b, 10, cfg()),
        )

    def test_inserir_futuro_nao_muda_previsao_historica(self):
        passado = [obs(0, posicao=.1), obs(1, posicao=.11)]
        estado1 = construir_estados_causais(passado, cfg())[1]
        estado2 = construir_estados_causais(passado + [obs(2, posicao=.12)], cfg())[1]
        self.assertEqual(estado1, estado2)

    def test_ground_truth_mesmo_veiculo_por_grupo(self):
        fonte = FonteLista([[obs(0, ordem="A"), obs(1, ordem="A"), obs(0, ordem="B"), obs(1, ordem="B")]])
        stats = avaliar_fonte(fonte, cfg(), 2)
        self.assertEqual(2, stats.avaliacoes)

    def test_ground_truth_rejeita_itinerario_diferente(self):
        dados = [obs(0, itinerario="i1"), obs(1, itinerario="i2")]
        futuro, motivo = selecionar_ground_truth(dados, [x.timestamp_gps for x in dados], 0, 10, cfg())
        self.assertIsNone(futuro)
        self.assertEqual("ground_truth_itinerario_diferente", motivo)

    def test_ground_truth_rejeita_viagem_diferente(self):
        dados = [obs(0, viagem="v1"), obs(1, viagem="v2")]
        futuro, motivo = selecionar_ground_truth(dados, [x.timestamp_gps for x in dados], 0, 10, cfg())
        self.assertIsNone(futuro)
        self.assertEqual("ground_truth_viagem_diferente", motivo)

    def test_ground_truth_respeita_tolerancia(self):
        dados = [obs(0, segundos=0), obs(1, segundos=13)]
        futuro, motivo = selecionar_ground_truth(dados, [x.timestamp_gps for x in dados], 0, 10, cfg())
        self.assertIsNone(futuro)
        self.assertEqual("ground_truth_ausente_na_tolerancia", motivo)

    def test_delta_real_e_registrado_e_usado(self):
        fonte = FonteLista([[obs(0, segundos=0, posicao=.1, velocidade=36, comprimento=1000),
                             obs(1, segundos=11, posicao=.21, comprimento=1000)]])
        with tempfile.TemporaryDirectory() as tmp:
            detalhes = EscritorDetalhesCsv(Path(tmp) / "d.csv")
            avaliar_fonte(fonte, cfg(tolerancias_segundos=((10, 2),)), 1, detalhes)
            import csv
            with (Path(tmp) / "d.csv").open(encoding="utf-8") as f:
                linha = next(csv.DictReader(f))
            self.assertEqual(11, float(linha["delta_real_segundos"]))
            self.assertAlmostEqual(.21, json.loads(linha["previsoes"])["B1"])

    def test_clamp_terminal(self):
        self.assertEqual(1, projetar(.99, 1000, 90, 30))

    def test_sem_wrap_circular(self):
        dados = [obs(0, posicao=.99), obs(1, posicao=.01)]
        futuro, motivo = selecionar_ground_truth(dados, [x.timestamp_gps for x in dados], 0, 10, cfg())
        self.assertIsNone(futuro)
        self.assertEqual("wrap_ou_regressao", motivo)

    def test_timestamp_duplicado_tem_desempate_por_id(self):
        dados = [obs(0, segundos=0), obs(2, segundos=10, posicao=.12), obs(1, segundos=10, posicao=.11)]
        avaliador = AvaliadorPosicao(cfg())
        avaliador.consumir_lote(dados)
        avaliador.finalizar()
        self.assertGreaterEqual(avaliador.stats.avaliacoes, 1)

    def test_fronteira_de_lote_nao_perde_historico(self):
        fonte = FonteLista([[obs(0, posicao=.1)], [obs(1, posicao=.11), obs(2, posicao=.12)]])
        stats = avaliar_fonte(fonte, cfg(), 1)
        self.assertEqual(2, stats.avaliacoes)

    def test_metricas_mae_e_percentis(self):
        m = AcumuladorMetrica(100, "teste")
        for valor in [1, 2, 3, 4]:
            m.adicionar(valor)
        d = m.como_dict(4)
        self.assertEqual(2.5, d["mae_m"])
        self.assertEqual(2.5, d["mediana_m"])
        self.assertAlmostEqual(3.7, d["p90_m"])
        self.assertAlmostEqual(3.85, d["p95_m"])

    def test_exclusoes_sao_contabilizadas(self):
        stats = avaliar_fonte(FonteLista([[obs(0), obs(1, itinerario=None)]]), cfg(), 2)
        self.assertGreater(stats.exclusoes["ground_truth_matching_ausente"], 0)

    def test_processamento_paginado_recebe_batch_size(self):
        fonte = FonteLista([[obs(0)], [obs(1)]])
        avaliar_fonte(fonte, cfg(), 37)
        self.assertEqual(37, fonte.tamanho)

    def test_sql_tem_ordem_deterministica_e_so_select(self):
        texto = SQL_TELEMETRIA.upper()
        self.assertIn('ORDER BY "MODAL", "PROVEDOR", "ORDEMVEICULO", "TIMESTAMPGPS", "ID"', texto)
        for proibido in ("INSERT ", "UPDATE ", "DELETE ", "CREATE ", "OFFSET "):
            self.assertNotIn(proibido, texto)

    def test_fonte_postgres_configura_transacao_read_only(self):
        chamadas = []

        class Cursor:
            def __enter__(self): return self
            def __exit__(self, *_): return None
            def execute(self, sql, params): chamadas.append((sql, params))
            def fetchmany(self, _): return []

        class Conexao:
            def __enter__(self): return self
            def __exit__(self, *_): return None
            def set_session(self, **kwargs): chamadas.append(kwargs)
            def cursor(self, **_): return Cursor()

        fake_extras = types.ModuleType("psycopg2.extras")
        fake_extras.RealDictCursor = object
        fake = types.ModuleType("psycopg2")
        fake.connect = lambda _: Conexao()
        fake.extras = fake_extras
        with patch.dict(sys.modules, {"psycopg2": fake, "psycopg2.extras": fake_extras}):
            list(FontePostgresTelemetria("dsn").lotes(10))
        self.assertIn({"readonly": True, "autocommit": False}, chamadas)

    def test_relatorio_reproduzivel_com_instante_fixo(self):
        stats1 = avaliar_fonte(FonteLista([[obs(0), obs(1)]]), cfg(), 1)
        stats2 = avaliar_fonte(FonteLista([[obs(0), obs(1)]]), cfg(), 2)
        instante = BASE + timedelta(days=1)
        r1 = criar_relatorio(stats1, cfg(), {"inicio": None}, instante)
        r2 = criar_relatorio(stats2, cfg(), {"inicio": None}, instante)
        self.assertEqual(r1, r2)
        with tempfile.TemporaryDirectory() as tmp:
            caminhos = escrever_relatorio(r1, Path(tmp))
            self.assertTrue(all(p.exists() for p in caminhos))

    def test_b2_legacy_e_diagnostico_separado(self):
        origem = obs(0, posicao=.1, legado=18)
        estado = construir_estados_causais([origem], cfg())[0]
        previsoes, _ = calcular_previsoes(origem, estado, 10, cfg())
        self.assertIn("B2_LEGACY", previsoes)
        self.assertNotIn("B2_MEDIANA", previsoes)


if __name__ == "__main__":
    unittest.main()
