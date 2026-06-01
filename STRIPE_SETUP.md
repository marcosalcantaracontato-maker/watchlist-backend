# Configuração do Stripe (assinatura Premium)

O código de pagamento já está pronto. Ele fica **desativado** até você
preencher as variáveis de ambiente abaixo no Railway. Sem elas, o app
funciona normal e o botão de upgrade retorna um erro claro (não quebra nada).

## Passo a passo

### 1. Criar conta Stripe
- Acesse https://dashboard.stripe.com e crie a conta (Brasil).
- Faça a verificação da conta para poder receber pagamentos de verdade.
- Comece no **modo de teste** (toggle no topo do dashboard) para testar sem dinheiro real.

### 2. Criar o produto e o preço
- Dashboard → **Produtos** → **Adicionar produto**.
- Nome: `WatchList Premium`.
- Preço: **R$ 19,00**, **Recorrente**, **Mensal**.
- Salve. Copie o **ID do preço** (começa com `price_...`).

### 3. Pegar a chave secreta
- Dashboard → **Desenvolvedores** → **Chaves de API**.
- Copie a **Chave secreta** (`sk_test_...` em teste, `sk_live_...` em produção).

### 4. Criar o webhook
- Dashboard → **Desenvolvedores** → **Webhooks** → **Adicionar endpoint**.
- URL do endpoint:
  `https://web-production-99f91.up.railway.app/api/billing/webhook`
- Eventos a escutar (selecione estes):
  - `checkout.session.completed`
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
  - `invoice.payment_failed`
- Salve e copie o **Signing secret** (`whsec_...`).

### 5. Configurar as variáveis no Railway
No projeto do backend no Railway → aba **Variables**, adicione:

| Variável | Valor |
|---|---|
| `STRIPE_SECRET_KEY` | `sk_test_...` (ou `sk_live_...` em produção) |
| `STRIPE_WEBHOOK_SECRET` | `whsec_...` |
| `STRIPE_PRICE_ID` | `price_...` (do passo 2) |
| `FRONTEND_URL` | `https://watchlist-frontend-tawny.vercel.app` |

O Railway reinicia o serviço sozinho após salvar.

### 6. Testar (modo de teste)
- No app: Configurações → **Fazer upgrade**.
- Use um cartão de teste do Stripe: `4242 4242 4242 4242`, validade futura, CVC qualquer.
- Após pagar, o webhook ativa o Premium e o app mostra "🎉 Bem-vindo ao Premium".
- Para cancelar/gerenciar: o botão vira **Gerenciar assinatura** (portal do Stripe).

## Sobre o Pix

O Stripe **não permite Pix em assinatura recorrente** — só em pagamento único.
A assinatura mensal automática usa **cartão**. Se quiser oferecer Pix no futuro,
o caminho é um pagamento único de Pix que concede 30 dias de Premium (precisa de
endpoint adicional). Posso implementar isso depois se fizer sentido.

## Importante
- O **webhook é a única fonte de verdade** do plano. O Premium só é concedido
  quando o Stripe confirma o pagamento — nunca pelo redirect do navegador.
- Em produção, troque as chaves de teste pelas de produção (`sk_live_`,
  `whsec_` do endpoint de produção) e refaça o webhook no modo Live.
