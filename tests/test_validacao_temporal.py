from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from posicao_corrigida import (
    ConfiguracaoAvaliador,
    FontePostgresGeometrias,
    Telemetria,
    avaliar_fonte,
    construir_estados_causais,
    calcular_previsoes,
    erro_geografico_metros,
    interpolar_geometria,
)
from validacao_temporal import (
    JanelaTemporal,
    analisar_estabilidade,
    cenarios_sensibilidade,
    comparacoes_relativas,
    criar_relatorio_consolidado,
    dividir_janelas,
    escrever_relatorio_consolidado,
    executar_janelas,
    melhoria_percentual,
    selecionar_holdout,
)


UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def obs(n: int, *, segundos=None, posicao=None, velocidade=36, ordem="A", itinerario="i1"):
    s = n * 10 if segundos is None else segundos
    p = n / 100 if posicao is None else posicao
    ts = BASE + timedelta(seconds=s)
    return Telemetria(
        id=f"{n:04d}", observacao_id=f"o{n}", modal="ONIBUS", provedor="SPPO_ZIRIX",
        ordem_veiculo=ordem, codigo_linha="10", origem_posicao="REAL",
        latitude_recebida=-22, longitude_recebida=-43, velocidade_instantanea=velocidade,
        bearing=90, timestamp_gps=ts, timestamp_envio_fonte=None,
        timestamp_servidor_fonte=None, recebido_em_utc=ts + timedelta(seconds=5),
        itinerario_id=itinerario, sentido_id=None, viagem_id="v1",
        posicao_na_rota=p, comprimento_rota_metros=1000, velocidade_media_causal=30,
    )


class Fonte:
    def __init__(self, dados): self.dados = dados
    def lotes(self, tamanho):
        for i in range(0, len(self.dados), tamanho):
            yield self.dados[i:i + tamanho]


def cfg(**kwargs):
    valores = dict(horizontes_segundos=(10,), tolerancias_segundos=((10, 2),),
                   amostra_percentis_maxima=100)
    valores.update(kwargs)
    return ConfiguracaoAvaliador(**valores)


class ValidacaoTemporalTests(unittest.TestCase):
    def test_divide_multiplas_janelas_sem_sobreposicao(self):
        j = dividir_janelas(BASE, BASE + timedelta(minutes=65), timedelta(minutes=30))
        self.assertEqual(3, len(j))
        self.assertEqual(j[0].fim, j[1].inicio)
        self.assertEqual(BASE + timedelta(minutes=65), j[-1].fim)

    def test_resultados_de_janelas_sao_independentes(self):
        janelas = [JanelaTemporal("A", BASE, BASE + timedelta(minutes=1)),
                   JanelaTemporal("B", BASE + timedelta(minutes=1), BASE + timedelta(minutes=2))]
        dados = {"A": [obs(0), obs(1)], "B": [obs(0), obs(1, posicao=.5)]}
        resultados = executar_janelas(janelas, lambda j: Fonte(dados[j.nome]), cfg(), 1)
        self.assertNotEqual(
            resultados[0]["resultados"]["metricas"][0]["mae_m"],
            resultados[1]["resultados"]["metricas"][0]["mae_m"],
        )

    def test_faixas_de_idade_real_sao_reportadas(self):
        stats = avaliar_fonte(Fonte([obs(0, segundos=0), obs(1, segundos=11)]), cfg(), 1)
        faixas = {m["valor"] for m in stats.como_dict()["metricas"]
                  if m["dimensao"] == "faixa_idade_correcao"}
        self.assertEqual({"10_15"}, faixas)

    def test_cenarios_de_sensibilidade_causal(self):
        cenarios = cenarios_sensibilidade(ConfiguracaoAvaliador())
        self.assertEqual({30, 60, 120, 180},
                         {int(c.janela_causal_segundos) for n, c in cenarios.items() if n.startswith("CAUSAL_")})

    def test_sensibilidade_parada_e_explicita(self):
        cenarios = cenarios_sensibilidade(ConfiguracaoAvaliador())
        self.assertIn("PARADA_V2", cenarios)
        self.assertIn("PARADA_D15", cenarios)
        self.assertIn("PARADA_C3", cenarios)

    def test_sensibilidade_tolerancia_ground_truth(self):
        cenarios = cenarios_sensibilidade(ConfiguracaoAvaliador())
        self.assertEqual((10, 5.0), cenarios["GT_MAIS_MENOS_5s"].tolerancias_segundos[0])

    def test_comparacao_relativa_tem_denominador_explicito(self):
        stats = avaliar_fonte(Fonte([obs(0), obs(1)]), cfg(), 2).como_dict()
        comparacoes = comparacoes_relativas(stats)
        b3 = next(x for x in comparacoes if x["baseline"] == "B3_MEDIANA")
        self.assertIn("mae_referencia_b0_m", b3)
        self.assertIn("mae_referencia_b1_m", b3)
        self.assertEqual(50, melhoria_percentual(5, 10))

    def test_terminal_e_clamp_sao_reportados(self):
        dados = [obs(0, posicao=.99, velocidade=90), obs(1, posicao=1, velocidade=0)]
        resultado = avaliar_fonte(Fonte(dados), cfg(), 1).como_dict()
        self.assertTrue(any(m["valor"] == "RESTANTE_ATE_2" for m in resultado["metricas"]))
        self.assertTrue(any(c["baseline"] == "B1" for c in resultado["clamps"]))

    def test_falso_avanco_parado_e_retomada_sao_reportados(self):
        dados = [obs(0, posicao=.2, velocidade=0), obs(1, posicao=.2, velocidade=0),
                 obs(2, posicao=.2, velocidade=0), obs(3, posicao=.2, velocidade=36),
                 obs(4, posicao=.21, velocidade=36)]
        resultado = avaliar_fonte(Fonte(dados), cfg(), 2).como_dict()
        retomadas = [m for m in resultado["metricas"]
                     if m["dimensao"] == "retomada" and m["valor"] == "RETOMADA_APOS_PARADO"]
        self.assertTrue(retomadas)
        # B3 usa zero quando PARADO, portanto não pode constar como falso avanço.
        self.assertFalse(any(x["baseline"] == "B3_MEDIANA"
                             for x in resultado["falsos_avancos_aparentes"]))

    def test_b2_legacy_permanece_separado(self):
        resultado = avaliar_fonte(Fonte([obs(0), obs(1), obs(2)]), cfg(), 2).como_dict()
        nomes = {m["baseline"] for m in resultado["metricas"]}
        self.assertIn("B2_LEGACY", nomes)
        self.assertIn("B2_MEDIANA", nomes)

    def test_b3_adaptativo_usa_b1_em_delta_curto(self):
        origem = obs(0, posicao=.1, velocidade=36)
        estado = construir_estados_causais([origem], cfg())[0]
        previsoes, _ = calcular_previsoes(origem, estado, 10, cfg())
        self.assertEqual(previsoes["B1"], previsoes["B3_ADAPTATIVO"])

    def test_b3_adaptativo_usa_hibrido_em_delta_maior(self):
        dados = [obs(0, segundos=0, posicao=.1, velocidade=18),
                 obs(1, segundos=10, posicao=.12, velocidade=54)]
        estado = construir_estados_causais(dados, cfg())[1]
        previsoes, _ = calcular_previsoes(dados[1], estado, 30, cfg())
        self.assertNotEqual(previsoes["B1"], previsoes["B3_ADAPTATIVO"])

    def test_interpolacao_e_erro_geografico(self):
        geo = {"type": "LineString", "coordinates": [[-43, -22], [-43, -21.99]]}
        meio = interpolar_geometria(geo, .5)
        self.assertAlmostEqual(-21.995, meio[1], places=6)
        self.assertGreater(erro_geografico_metros(geo, .25, .75), 500)

    def test_geografia_e_calculada_do_cache(self):
        geo = {"i1": {"type": "LineString", "coordinates": [[-43, -22], [-43, -21.99]]}}
        resultado = avaliar_fonte(Fonte([obs(0), obs(1)]), cfg(), 1, geometrias=geo).como_dict()
        self.assertTrue(resultado["metricas_geograficas"])

    def test_fonte_geometria_declara_uma_carga_em_lote(self):
        self.assertTrue(hasattr(FontePostgresGeometrias("dsn"), "carregar"))
        from posicao_corrigida import SQL_GEOMETRIAS
        self.assertIn("SELECT DISTINCT", SQL_GEOMETRIAS)
        self.assertNotIn("LIMIT 1", SQL_GEOMETRIAS)

    def test_holdout_so_existe_em_dia_posterior(self):
        mesmo_dia = dividir_janelas(BASE, BASE + timedelta(hours=3), timedelta(hours=1))
        _, holdout, _ = selecionar_holdout(mesmo_dia)
        self.assertIsNone(holdout)
        janelas = mesmo_dia + [JanelaTemporal("DIA2", BASE + timedelta(days=1),
                                              BASE + timedelta(days=1, hours=1))]
        exploracao, holdout, _ = selecionar_holdout(janelas)
        self.assertEqual("DIA2", holdout.nome)
        self.assertEqual(3, len(exploracao))

    def test_estabilidade_compara_janelas(self):
        janelas = [JanelaTemporal("A", BASE, BASE + timedelta(minutes=1)),
                   JanelaTemporal("B", BASE + timedelta(minutes=1), BASE + timedelta(minutes=2))]
        dados = {"A": [obs(0), obs(1)], "B": [obs(0), obs(1, posicao=.5)]}
        resultados = executar_janelas(janelas, lambda j: Fonte(dados[j.nome]), cfg(), 2)
        self.assertTrue(analisar_estabilidade(resultados))

    def test_relatorio_consolidado_reproduzivel(self):
        j = JanelaTemporal("A", BASE, BASE + timedelta(minutes=1))
        resultados = executar_janelas([j], lambda _: Fonte([obs(0), obs(1)]), cfg(), 2)
        r1 = criar_relatorio_consolidado(resultados, [], None, BASE)
        r2 = criar_relatorio_consolidado(resultados, [], None, BASE)
        self.assertEqual(r1, r2)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(all(p.exists() for p in escrever_relatorio_consolidado(r1, Path(tmp))))


if __name__ == "__main__":
    unittest.main()
