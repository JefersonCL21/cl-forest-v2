# cl_allometry

Ajuste, comparação e guarda de modelos alométricos (altura, volume, biomassa, DAP pela copa) com os indicadores
estatísticos, para que os coeficientes sejam sempre os ajustados e possam ser usados em qualquer lugar. Independe do
Streamlit: funciona nas páginas, em notebooks e em outros apps.

```python
from cl_allometry import fit, cross_validate, forms_for

f = fit("height.logistic", df, seed=0)      # df com dap_cm e ht_m
f.coefficients, f.ci95, f.p_value          # coeficientes, IC 95% e teste t
f.metrics                                  # r2, r2_adj, rmse(%), syx(%), bias(%), mae, aic, bic, furnival
f.checks                                   # Shapiro-Wilk, Breusch-Pagan, distância de Cook
f.predict(dap_cm=25)
cross_validate("height.logistic", df, k=5)  # métricas fora da amostra
[x.key for x in forms_for("ht_m")]           # as formas de altura do catálogo
```

- **Formas** (`forms.py`): altura (Curtis, Curtis em log, Logístico, Gompertz, Parabólico, Richards,
  Chapman-Richards, Henricksen, Stoffels), volume (Schumacher-Hall, Spurr, Husch, fator de forma), biomassa (potência
  do DAP, D²H, ρD²H de Chave) e DAP pela copa do drone. Cada fórmula é escrita **uma vez, como texto** (`expr.py`), e
  o mesmo texto serve para ajustar, aplicar e, depois, gerar SQL.
- **Ajuste** (`fit.py`): mínimos quadrados não lineares a partir de vários pontos de partida, ponderado se pedido;
  as formas log-lineares levam a correção do viés da volta dos logs. Um ajuste que não converge volta sem
  coeficientes, nunca com valores velhos.
- **Estatísticas** (`metrics.py`): definidas num lugar só; o viés é observado − estimado (positivo: o modelo
  subestima).

## Dados versionados, registro e campeã

```python
from cl_allometry import Filter, Registry, Scope, open_store
from cl_allometry import datasets

store = open_store("gs://<bucket>")              # ou uma pasta, para desenvolver
raw = datasets.read_upload("censo.xlsx", conteudo)
r = datasets.import_dataset(store, raw, family="inventario", mapping=datasets.suggest_mapping(raw.columns),
                            author="ana@courageousland.com", source_name="censo.xlsx")
df, m = datasets.load(store, "inventario@v2")      # confere o sha256 ao ler

reg = Registry(store)
rec = reg.run_fit(df, m, "height.logistic", Filter(especie=("Khaya ivorensis",)), author="ana@courageousland.com")
reg.promote(rec.fit_id, Scope.species("Khaya ivorensis"), author="ana@courageousland.com",
            reason="menor RMSE% na validação cruzada")
reg.resolve("ht_m", "Khaya ivorensis").predict(dap_cm=25)   # espécie → gênero → genérico
```

- **Armazenamento** (`store.py`): só cria; nada é sobrescrito nem apagado. Gravar os mesmos bytes de novo não faz
  nada, e bytes diferentes num caminho que existe são recusados (é isso que ordena duas pessoas gravando ao mesmo
  tempo). Uma pasta para desenvolver e testar; um bucket do Cloud Storage em produção.
- **Dados** (`datasets.py`): cada importação vira uma versão imutável de uma família (`inventario@v1`, `@v2`…),
  com as colunas em nomes com unidade (`dap_cm`, `ht_m`, `cap_cm`…), o DAP pelo CAP quando falta, e o **sha256 dos
  valores**, que não muda com a ordem das linhas e das colunas nem com o formato do arquivo. As linhas com problema
  são **marcadas** (`flag_*`), não descartadas; o ajuste escolhe o que deixar de fora e isso fica registrado.
- **Registro** (`registry.py`): cada ajuste é um registro imutável com os coeficientes, as estatísticas, a validação
  cruzada, o sha256 dos dados, o filtro, o autor, o commit e a data; o mesmo ajuste pedido de novo é encontrado, não
  refeito. A **campeã** de cada alvo e escopo (espécie, gênero ou genérico) é a última promoção, com justificativa;
  o rollback é promover de novo um ajuste antigo; o histórico nunca é reescrito.

Estrutura do armazenamento:
```
datasets/<sha256>/data.parquet | manifest.json        os dados e o que eles contêm
families/<família>/v000001.json                        a versão: sha256, anterior, autor, origem, mapeamento
fits/<alvo>/<sha12 dos dados>/<spec16>.json             um ajuste
promotions/<alvo>/<escopo>/000001.json                  as promoções e aposentadorias, só acrescentadas
```

## Escolher, aplicar e exportar

```python
from cl_allometry import selection, export
from cl_allometry.apply import apply, predict_one

ranked = selection.rank(reg.list_fits("ht_m", m.sha256))   # mesma versão e mesmas linhas
rec = selection.recommend(ranked, incumbent=campea_atual)  # rec.fit_id, rec.reasons, rec.warnings

out = apply(df, "ht_m", reg)    # ht_m_est, ht_m_final, ht_m_source, ht_m_model_id, ht_m_out_of_range
predict_one(reg, "ht_m", "Khaya ivorensis", dap_cm=25)

export.model_card(record, champion)        # JSON para outros sistemas
export.to_sql("height.logistic", coefs)    # a mesma fórmula em BigQuery ou Postgres
```

- **Seleção** (`selection.py`): cada estatística dá um posto no seu sentido e os postos são somados com pesos
  (padrão: RMSE% da validação cruzada ×2, |viés%| da validação cruzada, R² ajustado e AIC). Só se comparam
  candidatos do mesmo alvo, dos mesmos dados e das mesmas linhas. Num empate (até 0,5 ponto de RMSE%), ganha o de
  menos parâmetros, depois a campeã em uso. A recomendação diz o porquê e os avisos; quem decide é uma pessoa.
- **Aplicação** (`apply.py`): a altura medida tem prioridade; só se estima o que falta. Cada valor diz de onde
  veio (`measured`, `species`, `genus`, `generic` ou `none`) e qual ajuste o fez; fora da faixa do ajuste é
  marcado, nunca cortado. Encadeia: o volume pode usar a `ht_m_final` da altura.
- **Exportação** (`export.py`): o model card em JSON tem a fórmula, os coeficientes, as unidades, a faixa, as
  estatísticas e a origem dos dados; o SQL sai da mesma fórmula do ajuste e dá o mesmo valor do Python (testado
  em todas as formas).

## Os modelos de antes (`legacy.py`) e a versão 1
Os 9 modelos de altura da página `3_📈Modelos_alometricos.py` foram migrados como estavam, lidos do código-fonte
(sem redigitar) para `data/legacy_height_models.json`: os coeficientes das funções (`code`) e os da tabela exportada
(`table`), que divergem. `tests/test_legacy.py` registra o que os dados mostram:
- o `HT_Est` do `GERAL_V2.csv` veio da logística da **tabela** para Khaya grandifoliola, e é igual nas duas para as
  outras 8 espécies;
- reajustada nas 448 alturas medidas do Khaya grandifoliola, a logística dá os coeficientes **da tabela** (o código
  guarda um ajuste anterior);
- o `b2` parabólico da tabela está errado em todas as espécies (o do código é o certo);
- o Curtis antigo foi ajustado em log (`height.curtis_log` devolve os seus coeficientes).

**A versão 1** (decisão de 16/10/2026) é o reajuste: `scripts/bootstrap_height_v1.py` importa o `GERAL_V2.csv` como
`inventario@v1`, ajusta todas as formas de altura nas alturas medidas de cada espécie e promove a logística, que
reproduz os coeficientes que geraram o `HT_Est`. Nada é promovido se alguma se afastar mais de 1e-3 da tabela.
```
python scripts/bootstrap_height_v1.py --dry-run --author dev@local          # numa pasta temporária
python scripts/bootstrap_height_v1.py --store .allometry-local --author dev@local
```

## Testes
```
pip install -r requirements-dev.txt
python -m pytest -q
```
O teste central: dados feitos com coeficientes conhecidos devolvem esses coeficientes no ajuste. Os testes do
armazenamento rodam numa pasta, num bucket simulado e, com `ALLOMETRY_TEST_GCS=gs://bucket/prefixo`, num bucket real.
