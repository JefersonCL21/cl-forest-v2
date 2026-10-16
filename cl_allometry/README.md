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

- **Formas** (`forms.py`): altura (Curtis, Logístico, Gompertz, Parabólico, Richards, Chapman-Richards, Henricksen,
  Stoffels), volume (Schumacher-Hall, Spurr, Husch, fator de forma), biomassa (potência do DAP, D²H, ρD²H de Chave)
  e DAP pela copa do drone. Cada fórmula é escrita **uma vez, como texto** (`expr.py`), e o mesmo texto serve para
  ajustar, aplicar e, depois, gerar SQL.
- **Ajuste** (`fit.py`): mínimos quadrados não lineares a partir de vários pontos de partida, ponderado se pedido;
  as formas log-lineares levam a correção do viés da volta dos logs. Um ajuste que não converge volta sem
  coeficientes, nunca com valores velhos.
- **Estatísticas** (`metrics.py`): definidas num lugar só; o viés é observado − estimado (positivo: o modelo
  subestima).

## Os modelos de antes (`legacy.py`)
Os 9 modelos de altura da página `3_📈Modelos_alometricos.py` foram migrados como estavam, lidos do código-fonte
(sem redigitar) para `data/legacy_height_models.json`: os coeficientes das funções (`code`) e os da tabela exportada
(`table`), que divergem. `tests/test_legacy.py` registra o que os dados mostram:
- o `HT_Est` do `GERAL_V2.csv` veio da logística da **tabela** para Khaya grandifoliola, e é igual nas duas para as
  outras 8 espécies;
- reajustada nas 448 alturas medidas do Khaya grandifoliola, a logística dá os coeficientes **da tabela** (o código
  guarda um ajuste anterior);
- o `b2` parabólico da tabela está errado em todas as espécies (o do código é o certo).

## Testes
```
pip install -r requirements-dev.txt
python -m pytest -q
```
O teste central: dados feitos com coeficientes conhecidos devolvem esses coeficientes no ajuste.
