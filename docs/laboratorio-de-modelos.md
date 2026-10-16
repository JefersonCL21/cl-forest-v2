# Laboratório de Modelos

A página **🧪 Laboratório de Modelos** guarda e usa os modelos alométricos. Ela registra quem fez cada coisa, quando
e com quais dados. Por trás dela está o pacote [`cl_allometry`](../cl_allometry/README.md).

| Aba | O que faz | Quem pode |
|---|---|---|
| Dados | Importa uma planilha como nova versão, com hash, alertas de qualidade e diferença para a versão anterior | consultar: todos; importar: login |
| Ajustar | Ajusta várias espécies e formas de uma vez, com validação cruzada | login |
| Comparar | Mostra o ranking, a recomendação, as curvas, os resíduos e os coeficientes, e permite promover o modelo | consultar: todos; promover: promotores |
| Modelos | Mostra o modelo em uso por alvo e espécie e o histórico; permite voltar a um modelo anterior e baixar o model card e o SQL | consultar: todos; voltar e aposentar: promotores |
| Aplicar | Estima uma árvore ou uma planilha inteira, dizendo de onde veio cada valor | todos |

## Por enquanto, sem login
O login Google está desligado. Com `ALLOMETRY_WRITE=open`, quem vai importar, ajustar ou promover escreve o nome ou
o e-mail na barra lateral, e esse nome fica registrado como autor.

- **Quem pode gravar:** qualquer pessoa com o link.
- **O que continua protegido:** nada é sobrescrito nem apagado, só acrescentado. Uma promoção errada se desfaz com
  "Voltar para este", na aba Modelos.

**Por que o login está desligado:** com o Authlib 1.9.1, o login do Streamlit 1.44 falha. Ao clicar em entrar, aparece
`400: 'NoneType' object does not support item assignment`. A biblioteca grava o estado do login de um jeito que a
integração do Streamlit 1.44 não suporta.

**Para religar:**
1. Fixe um Authlib da mesma época do Streamlit 1.44 (por exemplo, 1.5.x) e confirme o login na revisão `lab`.
2. No `cloudbuild.yaml`, volte a montar o secret (o comentário no passo de deploy diz como) e tire o `open`.

## Rodar no computador (modo local)
Sem login e sem bucket, a página grava na pasta `.allometry-local/`, que fica fora do git, como `dev@local`.

```
python scripts/bootstrap_height_v1.py --store .allometry-local --author dev@local   # a versão 1 (uns 30 s)
streamlit run 1_🏠Inicio.py
```

Para testar o login localmente, copie `.streamlit/secrets.toml.example` para `.streamlit/secrets.toml` e preencha os
dados do cliente OAuth de desenvolvimento.

## Infraestrutura (uma vez, com o OK do Jeferson)
`<PROJETO>` é o projeto do Cloud Run `cl-biometrics`. Quem roda precisa ser dono do projeto.

```bash
gcloud config set project <PROJETO>
gcloud services enable secretmanager.googleapis.com

# 1. O bucket do registro: sem acesso público, com versionamento como rede de segurança
gcloud storage buckets create gs://<PROJETO>-allometry --location=us-central1 \
  --uniform-bucket-level-access --public-access-prevention
gcloud storage buckets update gs://<PROJETO>-allometry --versioning

# 2. Uma conta de serviço só para o app. Ela lê e cria objetos, mas não apaga nem sobrescreve: nem um bug
#    consegue mudar o histórico
gcloud iam service-accounts create cl-biometrics-run --display-name="cl-biometrics (Cloud Run)"
SA=cl-biometrics-run@<PROJETO>.iam.gserviceaccount.com
gcloud storage buckets add-iam-policy-binding gs://<PROJETO>-allometry --member=serviceAccount:$SA \
  --role=roles/storage.objectViewer
gcloud storage buckets add-iam-policy-binding gs://<PROJETO>-allometry --member=serviceAccount:$SA \
  --role=roles/storage.objectCreator
# o Cloud Build precisa poder fazer o deploy com essa conta
gcloud iam service-accounts add-iam-policy-binding $SA --role=roles/iam.serviceAccountUser \
  --member=serviceAccount:<CONTA_DO_CLOUD_BUILD>
```

3. **O cliente OAuth** (Console → APIs e serviços):
   - **Tela de consentimento:** tipo **Interno**. Assim, só contas da organização entram.
   - **Credenciais:** um cliente "Aplicativo da Web", com o URI de redirecionamento
     `https://<URL-PRINCIPAL-DO-APP>/oauth2callback`.
   - **Um segundo cliente** para `http://localhost:8501/oauth2callback`, para testar localmente.

4. **O secret com o login:** use o conteúdo de `.streamlit/secrets.toml.example` com:
   - o `redirect_uri` da URL principal;
   - um `cookie_secret` novo;
   - o cliente OAuth de produção;
   - `store = "gs://<PROJETO>-allometry"` e a lista de `promoters`.

   ```bash
   gcloud secrets create cl-biometrics-streamlit-secrets --data-file=secrets-producao.toml
   gcloud secrets add-iam-policy-binding cl-biometrics-streamlit-secrets --member=serviceAccount:$SA \
     --role=roles/secretmanager.secretAccessor
   ```

   Apague o `secrets-producao.toml` local depois. Para mudar os promotores, crie uma versão nova do secret e faça um
   novo deploy.

5. **A versão 1 no bucket**, com as suas credenciais (`gcloud auth application-default login`):
   ```
   python scripts/bootstrap_height_v1.py --store gs://<PROJETO>-allometry --author <seu e-mail>
   ```

## Deploy
O `cloudbuild.yaml` faz o deploy de uma revisão de teste (tag `test`) sem tráfego, já com a conta de serviço, o
secret, o bucket, a afinidade de sessão e o timeout de 1 hora.

- **Na URL de teste a página fica somente leitura.** O login do Streamlit só funciona no endereço do `redirect_uri`.
- **Depois de conferir, passe o tráfego** como de costume. A partir daí, gravar funciona na URL principal.

O que conferir na revisão de teste, antes de passar o tráfego:
- **As outras páginas abrem.**
- **O upload de uma planilha de ~10 MB funciona na página de Modelos alométricos.** Com o login configurado, o
  Streamlit liga a proteção XSRF, que antes estava desligada por causa do "erro 400". O `cookie_secret` fixo deve
  evitar o erro.
- **Depois de passar o tráfego, na URL principal:**
  - o login Google funciona;
  - uma conta de fora do domínio é recusada;
  - quem não é promotor não vê o "Promover";
  - um ajuste aparece em `gs://<PROJETO>-allometry/fits/`.

## Como os dados ficam guardados
```
datasets/<sha256>/data.parquet | manifest.json        os dados de cada versão e o que contêm
families/<família>/v000001.json                        a versão: sha256, anterior, autor, origem, mapeamento
fits/<alvo>/<sha12 dos dados>/<spec16>.json             cada ajuste, com coeficientes e estatísticas
promotions/<alvo>/<escopo>/000001.json                  promoções e aposentadorias, só acrescentadas
```
Nada é sobrescrito nem apagado. Um modelo usado fora do app sai pelo model card (JSON) ou pelo SQL da aba Modelos,
ou direto pelo pacote:

```python
from cl_allometry import champion
champion("ht_m", "Khaya grandifoliola", store="gs://<PROJETO>-allometry").predict(dap_cm=25)
```
