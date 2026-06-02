# Configuração da Efí (EfiPay) — Pix Automático recorrente

> O código está pronto e segue a documentação pública da Efí
> (dev.efipay.com.br). Fica **desativado** até as variáveis abaixo serem
> preenchidas. Sem elas, o app funciona normal e só o cartão (Stripe) aparece.

## Por que Efí
- **Self-service**: você cria a conta e testa no sandbox sem contrato.
- **Pix Automático** dedicado (recorrência via Pix, padrão do Banco Central).
- Documentação técnica e SDKs muito boas.
- **Requer conta Efí Empresas (PJ)** para o Pix Automático.

## Passo a passo

### 1. Criar conta e aplicação
- Crie a conta em https://sejaefi.com.br (Efí Empresas / PJ).
- No painel → **API** → crie uma **Aplicação**.
- Habilite a **API Pix** e marque os escopos:
  `rec.write`, `rec.read`, `cobr.write`, `cobr.read`, `webhook.write`, `webhook.read`.
- Anote o **Client_Id** e o **Client_Secret** (existe um par para Sandbox e outro para Produção).

### 2. Gerar o certificado (mTLS)
- Painel → **Meus Certificados** → gere e **baixe** o certificado (`.p12`).
  ⚠️ O download só acontece uma vez.
- Converta o `.p12` para `.pem` (cert + chave juntos):
  ```bash
  openssl pkcs12 -in certificado.p12 -out efi.pem -nodes
  # (senha em branco, a menos que você tenha definido uma)
  ```
- Gere o base64 do `.pem` (uma linha só) para colar no Railway:
  ```bash
  base64 -w0 efi.pem    # Linux
  base64 -i efi.pem     # macOS
  ```

### 3. Cadastrar a chave Pix do recebedor
- No painel da Efí, cadastre/escolha a **chave Pix** que receberá os valores.
  Use essa chave em `EFI_PIX_KEY`.

### 4. Configurar o webhook
- Painel Pix → Webhooks → aponte para:
  `https://web-production-99f91.up.railway.app/api/billing/efi/webhook`
- (A Efí entrega o webhook com mTLS; o app casa o usuário pelo `idRec`.)

### 5. Variáveis no Railway
| Variável | Valor |
|---|---|
| `EFI_CLIENT_ID` | Client_Id da aplicação |
| `EFI_CLIENT_SECRET` | Client_Secret |
| `EFI_PIX_KEY` | sua chave Pix de recebimento |
| `EFI_CERT_BASE64` | o base64 do `efi.pem` (passo 2) |
| `EFI_ENV` | `sandbox` (depois `production`) |
| `EFI_PRICE` | `19.00` |
| `BACKEND_PUBLIC_URL` | `https://web-production-99f91.up.railway.app` |

Após salvar, o Railway reinicia. No log deve aparecer `✅ Certificado Efí carregado`.

### 6. Testar (sandbox)
- App → Configurações → **Fazer upgrade** → **Pix Automático**.
- Informe o **CPF** (o Pix Automático exige o CPF do pagador para o mandato).
- O backend chama `POST /v2/rec` e cria a recorrência.
- Autorize o mandato no fluxo de teste; o webhook libera o Premium.

## Pontos a confirmar no onboarding
A estrutura segue as docs, mas confirme no sandbox:
- O formato exato do corpo de `POST /v2/rec` (campos `vinculo/calendario/valor`).
- Como a **jornada de autorização** é entregue (QR/location) para mostrar ao usuário.
- Os valores de `status` enviados no webhook de recorrência (mapeados em
  `efi_webhook`: APROVADA/ATIVA liberam, CANCELADA/REJEITADA encerram).

## Como o plano é definido
O **webhook é a única fonte de verdade**. O Premium só é concedido quando a Efí
confirma a autorização da recorrência (casada pelo `idRec` salvo no usuário).
O retorno no navegador nunca concede acesso sozinho.
