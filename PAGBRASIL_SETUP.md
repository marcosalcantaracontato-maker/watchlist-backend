# Configuração da PagBrasil — Pix Automático (assinatura recorrente)

> ⚠️ **Status: adaptador pronto, pendente de onboarding + verificação.**
> Diferente do Stripe, a PagBrasil **não é self-service**. O código segue a
> estrutura documentada por eles, mas os **nomes exatos de alguns campos** só
> aparecem na documentação do dashboard após a aprovação da sua conta. Os
> pontos a confirmar estão marcados com `# CONFIRMAR` no `server.py`.

## Por que PagBrasil (e não Pix no Stripe)
O Stripe só aceita Pix em **pagamento único** — não serve para assinatura
recorrente. A PagBrasil tem o **Pix Automático**, o padrão do Banco Central
para cobranças recorrentes via Pix. Por isso os dois provedores convivem:
- **Stripe** → cartão recorrente (funciona já)
- **PagBrasil** → Pix Automático recorrente

## Passo a passo

### 1. Abrir conta de comerciante
- Acesse https://www.pagbrasil.com e solicite uma conta de comerciante.
- Assine o "Payment Service Agreement". A aprovação não é instantânea.
- Suporte: `support@pagbrasil.com`.

### 2. Pegar as credenciais
- No Dashboard PagBrasil → **Account** → defina a **"Secret Phrase"**.
- Essa frase autentica as chamadas à API.
- Comece no **sandbox** (`https://sandbox.pagbrasil.com/api`). A URL de
  produção é fornecida quando você pede para a conta ir ao ar.

### 3. Confirmar os campos da API (importante)
Acesse a documentação no dashboard (seções "Requesting a payment", "Security"
e "Automatic Pix") e confirme/ajuste no `server.py` os pontos marcados
`# CONFIRMAR`:
- Nome do parâmetro de autenticação (`secret_token`?)
- Seletor do método Pix Automático e a flag de recorrência/consentimento
- Campo do `pix_rec_id` na resposta
- Formato e header da **assinatura do webhook** (HMAC? qual header?)
- Valores de `status` enviados no webhook (authorized/paid/canceled...)

### 4. Configurar o webhook
- No Dashboard PagBrasil, aponte as notificações para:
  `https://web-production-99f91.up.railway.app/api/billing/pagbrasil/webhook`

### 5. Variáveis de ambiente no Railway
| Variável | Valor |
|---|---|
| `PAGBRASIL_SECRET` | sua Secret Phrase |
| `PAGBRASIL_API_URL` | `https://sandbox.pagbrasil.com/api` (depois a de produção) |
| `PAGBRASIL_PRICE` | `19.00` |
| `BACKEND_PUBLIC_URL` | `https://web-production-99f91.up.railway.app` |
| `PAYMENT_PROVIDER` | `stripe` ou `pagbrasil` (padrão para novas assinaturas) |

Enquanto `PAGBRASIL_SECRET` estiver vazio, o Pix Automático fica **desativado**
e o app continua normal (só o cartão/Stripe aparece, se configurado).

### 6. Testar no sandbox
- App → Configurações → **Fazer upgrade** → **Pix Automático**.
- O backend chama `/api/order/add` e devolve o QR Code.
- Autorize no ambiente de teste; o webhook libera o Premium.

## Como o app decide o plano
O **webhook é a única fonte de verdade**. O Premium só é concedido quando a
PagBrasil confirma a autorização da recorrência (casada pelo `pix_rec_id`
salvo no usuário). O retorno no navegador (`?checkout=success`) apenas atualiza
a tela — nunca concede acesso sozinho.

## Endpoints implementados
- `POST /api/billing/pagbrasil/checkout` — cria o pedido com consentimento e
  devolve `pixImage`, `pixCode`, `expiration`, `pixRecId`.
- `POST /api/billing/pagbrasil/webhook` — recebe o status e atualiza o plano.
- `GET /api/billing/config` — informa ao frontend os métodos ativos.
