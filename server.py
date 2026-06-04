"""
WatchList — Backend API
FastAPI + MongoDB (Motor) + JWT + Google OAuth

Deploy: Railway, Render, ou qualquer VPS
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from typing import List, Any
from datetime import datetime, timedelta
from bson import ObjectId
import os, httpx
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MONGODB_URL            = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
JWT_SECRET             = os.getenv("JWT_SECRET", "TROQUE_ISSO_POR_UM_SECRET_FORTE")
GOOGLE_CLIENT_ID       = os.getenv("GOOGLE_CLIENT_ID", "")
ALGORITHM              = "HS256"
TOKEN_EXPIRE_DAYS      = 30
# E-mails com acesso ao painel admin (separados por vírgula). Ajuste no Railway.
ADMIN_EMAILS           = {e.strip().lower() for e in os.getenv(
    "ADMIN_EMAILS", "marcosalcantara.contato@gmail.com").split(",") if e.strip()}
FRONTEND_URL           = os.getenv("FRONTEND_URL", "")  # ex: https://watchlist.vercel.app
# URL pública do app para redirecionar após checkout (cai no Vercel se não setada)
APP_PUBLIC_URL         = FRONTEND_URL or "https://watchlist-frontend-tawny.vercel.app"

# ─── STRIPE (assinatura Premium) ──────────────────────────────────────────────
STRIPE_SECRET_KEY      = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET  = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID        = os.getenv("STRIPE_PRICE_ID", "")  # price_... da assinatura R$19/mês
YOUTUBE_API_KEY        = os.getenv("YOUTUBE_API_KEY", "")  # opcional: habilita captura de DURAÇÃO ao salvar
try:
    import stripe
    if STRIPE_SECRET_KEY:
        stripe.api_key = STRIPE_SECRET_KEY
except ImportError:
    stripe = None  # biblioteca não instalada — endpoints de billing retornam erro claro

def stripe_ready() -> bool:
    return bool(stripe and STRIPE_SECRET_KEY and STRIPE_PRICE_ID)

# ─── EFÍ / EfiPay (Pix Automático — assinatura recorrente via Pix) ─────────────
# A API Pix da Efí usa OAuth2 + certificado mTLS (.p12/.pem) OBRIGATÓRIO em toda
# requisição. Requer conta Efí Empresas (PJ). Veja EFI_SETUP.md.
EFI_CLIENT_ID      = os.getenv("EFI_CLIENT_ID", "")
EFI_CLIENT_SECRET  = os.getenv("EFI_CLIENT_SECRET", "")
EFI_PIX_KEY        = os.getenv("EFI_PIX_KEY", "")        # chave Pix do recebedor (você)
EFI_PRICE          = os.getenv("EFI_PRICE", "19.00")     # valor mensal em BRL
EFI_ENV            = os.getenv("EFI_ENV", "sandbox")     # "sandbox" | "production"
# Certificado mTLS em base64 (PEM). No Railway: cole o conteúdo do .pem em base64.
EFI_CERT_BASE64    = os.getenv("EFI_CERT_BASE64", "")
EFI_BASE_URL       = ("https://pix.api.efipay.com.br" if EFI_ENV == "production"
                      else "https://pix-h.api.efipay.com.br")
_EFI_CERT_PATH     = None  # preenchido no startup quando o cert é gravado em disco

# URL pública DESTE backend (para os provedores enviarem webhook)
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "https://web-production-99f91.up.railway.app")

def efi_ready() -> bool:
    return bool(EFI_CLIENT_ID and EFI_CLIENT_SECRET and EFI_PIX_KEY and _EFI_CERT_PATH)

# Provedor padrão para novas assinaturas: "stripe" (cartão) ou "efi" (Pix Automático)
PAYMENT_PROVIDER   = os.getenv("PAYMENT_PROVIDER", "stripe")

# ─── RATE LIMITER ────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ─── APP ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="WatchList API", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: allow_credentials=True é incompatível com allow_origins=["*"].
# O browser bloqueia quando o header retorna "*" + credentials mode "include".
# Listamos as origens explicitamente e usamos regex para cobrir previews do Vercel.
_extra = [FRONTEND_URL] if FRONTEND_URL else []
ALLOWED_ORIGINS = _extra + [
    "https://watchlist-frontend-tawny.vercel.app",
    "https://watchlist-frontend-p4uqtoo5g.vercel.app",
    "https://watchlist-frontend-kupe3dnwt.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://watchlist-frontend.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MONGODB ──────────────────────────────────────────────────────────────────
client = AsyncIOMotorClient(MONGODB_URL)
db     = client.watchlist

# ─── HELPERS ──────────────────────────────────────────────────────────────────
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

def serialize(doc) -> dict:
    """Convert MongoDB document to JSON-serializable dict."""
    if doc is None:
        return None
    doc["id"] = str(doc.pop("_id"))
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            doc[k] = str(v)
        elif isinstance(v, datetime):
            doc[k] = v.isoformat()
    return doc

def create_jwt(user_id: str) -> str:
    exp = datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": user_id, "exp": exp}, JWT_SECRET, algorithm=ALGORITHM)

# Campos que serialize() converteu de datetime → ISO string; reconvertidos no restore.
_DATETIME_FIELDS = {"createdAt", "watchedAt", "updatedAt", "deletedAt", "dueDate", "lastLogin", "migratedAt"}

def deserialize_doc(doc: dict) -> dict:
    """Inverso de serialize(): prepara um doc do snapshot para reinserção no Mongo.
    Preserva o _id original (mantém intactas as referências parentId/categoryId/
    folderId/linkedItemId) e reconverte datas ISO em datetime."""
    out = dict(doc)
    raw_id = out.pop("id", None) or out.pop("_id", None)
    if raw_id:
        try:
            out["_id"] = ObjectId(raw_id)
        except Exception:
            pass  # id não-ObjectId (ex.: dado legado) — deixa o Mongo gerar um novo
    for k in _DATETIME_FIELDS:
        v = out.get(k)
        if isinstance(v, str) and v:
            try:
                out[k] = datetime.fromisoformat(v)
            except ValueError:
                pass
    return out

# ─── AUTH MIDDLEWARE ──────────────────────────────────────────────────────────
# auto_error=False: não lança 403 quando não há Authorization header
# (permite que o cookie seja usado como fallback)
security = HTTPBearer(auto_error=False)

COOKIE_NAME = "wl_auth"

async def get_current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    # Prioridade: 1) Authorization header  2) cookie httpOnly
    token = None
    if creds:
        token = creds.credentials
    if not token:
        token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Não autenticado")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        user    = await db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=401, detail="Usuário não encontrado")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido ou expirado")

def is_admin(user) -> bool:
    return bool(user and (user.get("email") or "").lower() in ADMIN_EMAILS)

async def get_admin_user(user=Depends(get_current_user)):
    """Garante que o usuário autenticado é admin (e-mail na lista ADMIN_EMAILS)."""
    if not is_admin(user):
        raise HTTPException(status_code=403, detail="Acesso restrito ao administrador.")
    return user

# ─── MODELS ──────────────────────────────────────────────────────────────────
class GoogleLoginRequest(BaseModel):
    google_token: str   # ID token do Google Identity Services (accounts.google.com/gsi/client)

class CategoryCreate(BaseModel):
    name:     str
    parentId: Optional[str] = None
    order:    int = 0

class CategoryUpdate(BaseModel):
    name:     Optional[str] = None
    parentId: Optional[str] = None
    order:    Optional[int] = None

class LinkCreate(BaseModel):
    url:        str
    title:      str
    thumbnail:  Optional[str] = ""
    rawThumb:   Optional[str] = ""
    platform:   str
    videoId:    Optional[str] = ""
    categoryId: Optional[str] = None
    watched:    bool = False
    notes:      Optional[str] = ""
    tags:       List[str] = []
    order:      int = 0
    # Sinais comportamentais (fundação das Coleções Inteligentes)
    durationSeconds: Optional[int] = None
    watchedSeconds:  Optional[int] = None
    lastWatchedAt:   Optional[str] = None
    watchCount:      Optional[int] = None
    isFavorite:      Optional[bool] = None

class LinkUpdate(BaseModel):
    title:      Optional[str] = None
    thumbnail:  Optional[str] = None
    rawThumb:   Optional[str] = None
    categoryId: Optional[str] = None
    watched:    Optional[bool] = None
    notes:      Optional[str] = None
    tags:       Optional[List[str]] = None
    order:      Optional[int] = None
    # Sinais comportamentais (progresso/duração) — persistem o tracking
    durationSeconds: Optional[int] = None
    watchedSeconds:  Optional[int] = None
    lastWatchedAt:   Optional[str] = None
    watchCount:      Optional[int] = None
    isFavorite:      Optional[bool] = None

class MigrateRequest(BaseModel):
    categories: List[dict]
    links:      List[dict]

class BackupCreate(BaseModel):
    label: Optional[str] = None   # rótulo opcional para backup manual

# ─── NOTES MODELS ────────────────────────────────────────────────────────────
class NoteFolderCreate(BaseModel):
    name:     str
    parentId: Optional[str] = None
    color:    Optional[str] = None   # hex color para barra lateral
    order:    int = 0

class NoteFolderUpdate(BaseModel):
    name:     Optional[str] = None
    parentId: Optional[str] = None
    color:    Optional[str] = None
    order:    Optional[int] = None

class NoteCreate(BaseModel):
    title:        Optional[str] = ""
    body:         Optional[str] = ""       # markdown leve / texto puro
    folderId:     Optional[str] = None     # null = Caixa de entrada
    linkedItemId: Optional[str] = None     # vínculo com vídeo salvo
    priority:     int = 4                  # 1=alta, 4=sem prioridade
    dueDate:      Optional[str] = None     # ISO string
    tags:         List[str] = []
    isCompleted:  bool = False
    position:     Optional[float] = None   # rank manual; maior = topo da lista

class NoteUpdate(BaseModel):
    title:        Optional[str] = None
    body:         Optional[str] = None
    folderId:     Optional[str] = None     # passe "__inbox__" para mover para Caixa
    linkedItemId: Optional[str] = None
    priority:     Optional[int] = None
    dueDate:      Optional[str] = None
    tags:         Optional[List[str]] = None
    isCompleted:  Optional[bool] = None
    position:     Optional[float] = None

# ─── AUTH ROUTES ──────────────────────────────────────────────────────────────
@app.get("/api/")
async def root():
    return {"status": "ok", "service": "WatchList API v2.0"}

@app.get("/api/health")
async def health():
    """Health check leve que também faz um ping no Mongo — mantém o cluster
    (Atlas) e o container (Railway) quentes quando chamado por um cron externo
    a cada ~5 min, evitando o cold start que faz o 1º login falhar."""
    try:
        await db.command("ping")
        return {"status": "ok", "db": "up"}
    except Exception as e:
        return {"status": "degraded", "db": "down", "error": str(e)}

@app.post("/api/auth/google")
@limiter.limit("5/minute")
async def login_with_google(request: Request, body: GoogleLoginRequest, response: Response):
    """
    Verifica Google ID token via Google Identity Services (sem Firebase).
    O frontend usa: https://accounts.google.com/gsi/client
    e envia response.credential diretamente para este endpoint.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client_http:
            # Verificação direta com Google — sem dependência de Firebase
            r = await client_http.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={body.google_token}"
            )
            if r.status_code != 200:
                raise HTTPException(status_code=401, detail="Token Google inválido")

            token_data = r.json()

            # Verifica se o token é para o seu app
            # Se GOOGLE_CLIENT_ID estiver vazio, aceita qualquer token Google (desenvolvimento)
            aud = token_data.get("aud", "")
            if GOOGLE_CLIENT_ID and aud not in [GOOGLE_CLIENT_ID, f"{GOOGLE_CLIENT_ID}.apps.googleusercontent.com"]:
                raise HTTPException(status_code=401, detail="Token não pertence a este app")

            # Verifica expiração
            import time
            if int(token_data.get("exp", 0)) < int(time.time()):
                raise HTTPException(status_code=401, detail="Token expirado")

            email  = token_data.get("email", "")
            name   = token_data.get("name", email.split("@")[0])
            avatar = token_data.get("picture", "")

            if not email:
                raise HTTPException(status_code=400, detail="Email não encontrado no token")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Falha na verificação Google: {str(e)}")
    
    # Upsert user
    existing = await db.users.find_one({"email": email})
    if existing:
        user_id = str(existing["_id"])
        await db.users.update_one(
            {"_id": existing["_id"]},
            {"$set": {"name": name, "avatar": avatar, "lastLogin": datetime.utcnow()}}
        )
        is_new = False
    else:
        result = await db.users.insert_one({
            "email": email, "name": name, "avatar": avatar,
            "plan": "free", "createdAt": datetime.utcnow(), "lastLogin": datetime.utcnow()
        })
        user_id = str(result.inserted_id)
        is_new  = True
    
    token = create_jwt(user_id)
    user  = await db.users.find_one({"_id": ObjectId(user_id)})

    # Seta cookie httpOnly — JS não consegue ler, seguro contra XSS
    # SameSite=none + Secure necessário para cross-origin (Vercel → Railway)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/",
    )

    user_data = serialize(user)
    user_data["isAdmin"] = is_admin(user)
    return {
        "token":  token,   # mantido para a extensão Chrome
        "user":   user_data,
        "is_new": is_new
    }

@app.get("/api/auth/me")
async def get_me(response: Response, background_tasks: BackgroundTasks, user=Depends(get_current_user)):
    # Devolve um JWT fresco e renova o cookie. Necessário porque o frontend
    # mantém o JWT apenas em memória — após reload da página ele some, e sem
    # token as chamadas a /api/links e /api/categories cairiam no cache local
    # (dando a falsa impressão de que o usuário perdeu todos os dados).
    # Aproveita o load do app para disparar o backup automático diário (background).
    background_tasks.add_task(maybe_auto_backup, str(user["_id"]))
    token = create_jwt(str(user["_id"]))
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/",
    )
    data = serialize(user)
    data["token"] = token
    data["isAdmin"] = is_admin(user)
    return data

@app.post("/api/auth/logout")
async def logout(response: Response, user=Depends(get_current_user)):
    response.delete_cookie(
        key=COOKIE_NAME,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
    )
    return {"ok": True}

# ─── BILLING / PREMIUM ─────────────────────────────────────────────────────
# Dois provedores que convivem:
#   • Stripe → assinatura mensal por cartão (self-service)
#   • Efí    → Pix Automático recorrente (OAuth2 + mTLS — ver EFI_SETUP.md)
@app.get("/api/billing/config")
async def billing_config():
    """Informa ao frontend quais provedores/métodos de pagamento estão ativos."""
    methods = []
    if stripe_ready(): methods.append({"provider": "stripe", "method": "card",          "label": "Cartão"})
    if efi_ready():    methods.append({"provider": "efi",    "method": "pix_automatico", "label": "Pix Automático"})
    return {
        "enabled": bool(methods),
        "default": PAYMENT_PROVIDER,
        "methods": methods,
    }

@app.post("/api/billing/checkout")
@limiter.limit("10/hour")
async def billing_checkout(request: Request, user=Depends(get_current_user)):
    """Cria uma sessão de checkout do Stripe (assinatura mensal) e devolve a URL."""
    if not stripe_ready():
        raise HTTPException(status_code=503, detail="Pagamento não configurado no servidor.")
    uid = str(user["_id"])

    # Reaproveita o customer do Stripe se já existir (evita duplicar clientes)
    customer_id = user.get("stripeCustomerId")
    try:
        if not customer_id:
            customer = stripe.Customer.create(
                email=user.get("email"),
                name=user.get("name"),
                metadata={"userId": uid},
            )
            customer_id = customer.id
            await db.users.update_one({"_id": user["_id"]}, {"$set": {"stripeCustomerId": customer_id}})

        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            client_reference_id=uid,
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            allow_promotion_codes=True,
            success_url=f"{APP_PUBLIC_URL}/?checkout=success",
            cancel_url=f"{APP_PUBLIC_URL}/?checkout=cancel",
            metadata={"userId": uid},
            subscription_data={"metadata": {"userId": uid}},
        )
        return {"url": session.url}
    except Exception as e:
        print(f"[billing] checkout falhou para {uid}: {e}")
        raise HTTPException(status_code=502, detail="Não foi possível iniciar o pagamento.")

@app.post("/api/billing/portal")
@limiter.limit("10/hour")
async def billing_portal(request: Request, user=Depends(get_current_user)):
    """Abre o portal do Stripe para o usuário gerenciar/cancelar a assinatura."""
    if not stripe_ready():
        raise HTTPException(status_code=503, detail="Pagamento não configurado no servidor.")
    customer_id = user.get("stripeCustomerId")
    if not customer_id:
        raise HTTPException(status_code=400, detail="Nenhuma assinatura encontrada.")
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{APP_PUBLIC_URL}/",
        )
        return {"url": session.url}
    except Exception as e:
        print(f"[billing] portal falhou: {e}")
        raise HTTPException(status_code=502, detail="Não foi possível abrir o portal de assinatura.")

async def _set_plan(customer_id: str, plan: str, sub_id: str = None, status: str = None):
    """Atualiza o plano do usuário a partir do customer do Stripe."""
    update = {"plan": plan}
    if sub_id is not None:    update["stripeSubscriptionId"] = sub_id
    if status is not None:    update["planStatus"] = status
    r = await db.users.update_one({"stripeCustomerId": customer_id}, {"$set": update})
    if r.matched_count == 0:
        print(f"[billing] webhook: nenhum usuário com customer {customer_id}")

@app.post("/api/billing/webhook")
async def stripe_webhook(request: Request):
    """
    Recebe eventos do Stripe. É a ÚNICA fonte de verdade para conceder/remover
    Premium — nunca confiar no redirect de sucesso do frontend.
    Verifica a assinatura do webhook para garantir que veio do Stripe.
    """
    if not (stripe and STRIPE_WEBHOOK_SECRET):
        raise HTTPException(status_code=503, detail="Webhook não configurado.")
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        print(f"[billing] webhook inválido: {e}")
        raise HTTPException(status_code=400, detail="Assinatura do webhook inválida.")

    etype = event["type"]
    obj   = event["data"]["object"]

    if etype == "checkout.session.completed":
        customer_id = obj.get("customer")
        sub_id      = obj.get("subscription")
        if customer_id:
            await _set_plan(customer_id, "premium", sub_id, "active")
            print(f"[billing] Premium ativado para customer {customer_id}")

    elif etype in ("customer.subscription.updated", "customer.subscription.created"):
        customer_id = obj.get("customer")
        status      = obj.get("status")  # active, trialing, past_due, canceled, unpaid...
        sub_id      = obj.get("id")
        # Premium só enquanto a assinatura estiver ativa/em teste
        plan = "premium" if status in ("active", "trialing") else "free"
        if customer_id:
            await _set_plan(customer_id, plan, sub_id, status)

    elif etype == "customer.subscription.deleted":
        customer_id = obj.get("customer")
        if customer_id:
            await _set_plan(customer_id, "free", None, "canceled")
            print(f"[billing] Assinatura cancelada para customer {customer_id}")

    elif etype == "invoice.payment_failed":
        customer_id = obj.get("customer")
        if customer_id:
            await _set_plan(customer_id, "free", status="past_due")

    return {"received": True}

# ─── EFÍ: Pix Automático (assinatura recorrente via Pix) ───────────────────────
# A API Pix da Efí exige OAuth2 + certificado mTLS (.p12/.pem) em TODA requisição.
# Endpoints (docs públicas dev.efipay.com.br):
#   POST /oauth/token   → token (Basic base64(client_id:client_secret) + cert)
#   POST /v2/rec        → cria a recorrência (mandato), devolve idRec
#   webhook             → confirma autorização/cobrança → define o plano
# Inativo até EFI_CLIENT_ID/SECRET/PIX_KEY + certificado estarem configurados.
# Pix Automático requer conta Efí Empresas (PJ). Ver EFI_SETUP.md.

_efi_token_cache = {"token": None, "exp": 0}

async def _efi_token() -> str:
    """Obtém (e cacheia) o access_token OAuth2 da Efí, usando o certificado mTLS."""
    import time as _time, base64 as _b64
    if _efi_token_cache["token"] and _efi_token_cache["exp"] > _time.time() + 30:
        return _efi_token_cache["token"]
    auth = _b64.b64encode(f"{EFI_CLIENT_ID}:{EFI_CLIENT_SECRET}".encode()).decode()
    async with httpx.AsyncClient(cert=_EFI_CERT_PATH, timeout=20) as client:
        r = await client.post(
            f"{EFI_BASE_URL}/oauth/token",
            headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
            json={"grant_type": "client_credentials"},
        )
        r.raise_for_status()
        d = r.json()
    _efi_token_cache["token"] = d["access_token"]
    _efi_token_cache["exp"]   = _time.time() + int(d.get("expires_in", 3600))
    return _efi_token_cache["token"]

async def _set_plan_by_field(field: str, value: str, plan: str, status: str = None):
    """Atualiza o plano localizando o usuário por um campo arbitrário (ex.: efiIdRec)."""
    update = {"plan": plan}
    if status is not None:
        update["planStatus"] = status
    r = await db.users.update_one({field: value}, {"$set": update})
    if r.matched_count == 0:
        print(f"[efi] webhook: nenhum usuário com {field}={value}")

class EfiCheckout(BaseModel):
    cpf: str   # Pix Automático exige o CPF do pagador para o mandato

@app.post("/api/billing/efi/checkout")
@limiter.limit("10/hour")
async def efi_checkout(request: Request, body: EfiCheckout, user=Depends(get_current_user)):
    """
    Cria a recorrência (mandato) de Pix Automático na Efí.
    O usuário autoriza a recorrência no app do banco; o webhook libera o Premium.
    """
    if not efi_ready():
        raise HTTPException(status_code=503, detail="Pix Automático não configurado no servidor.")
    uid = str(user["_id"])
    cpf = "".join(filter(str.isdigit, body.cpf or ""))
    if len(cpf) != 11:
        raise HTTPException(status_code=400, detail="CPF inválido.")

    today = datetime.utcnow().strftime("%Y-%m-%d")
    # Corpo conforme docs do Pix Automático da Efí (POST /v2/rec).
    rec_body = {
        "vinculo": {
            "contrato": f"wl-{uid}",
            "devedor":  {"cpf": cpf, "nome": user.get("name", "Cliente")},
            "objeto":   "WatchList Premium",
        },
        "calendario": {"dataInicial": today, "periodicidade": "MENSAL"},
        "valor":      {"valorRec": str(EFI_PRICE)},
        "politicaRetentativa": "PERMITE_3R_7D",
        "recebedor":  {"chave": EFI_PIX_KEY},
    }
    try:
        token = await _efi_token()
        async with httpx.AsyncClient(cert=_EFI_CERT_PATH, timeout=20) as client:
            r = await client.post(
                f"{EFI_BASE_URL}/v2/rec",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=rec_body,
            )
        r.raise_for_status()
        data = r.json()
        id_rec = data.get("idRec")
        if id_rec:
            await db.users.update_one({"_id": user["_id"]}, {"$set": {"efiIdRec": id_rec, "efiCpf": cpf}})
        # A jornada de autorização do mandato é concluída no app do banco do pagador.
        # Retornamos o idRec e o que a Efí devolver (location/QR quando disponível).
        return {
            "provider": "efi",
            "idRec":    id_rec,
            "pixCode":  data.get("pixCopiaECola") or (data.get("loc") or {}).get("pixCopiaECola"),
            "qrLoc":    (data.get("loc") or {}).get("location"),
            "raw":      data,
        }
    except httpx.HTTPStatusError as e:
        print(f"[efi] checkout HTTP {e.response.status_code}: {e.response.text[:300]}")
        raise HTTPException(status_code=502, detail="Não foi possível iniciar o Pix Automático.")
    except Exception as e:
        print(f"[efi] checkout falhou para {uid}: {e}")
        raise HTTPException(status_code=502, detail="Não foi possível iniciar o Pix Automático.")

@app.post("/api/billing/efi/webhook")
async def efi_webhook(request: Request):
    """
    Recebe notificações de recorrência/cobrança da Efí. Fonte de verdade do plano.
    A Efí valida a origem via mTLS (o webhook é entregue com o certificado dela);
    casamos o usuário pelo idRec salvo no checkout.
    """
    if not efi_ready():
        raise HTTPException(status_code=503, detail="Pix Automático não configurado.")
    try:
        data = await request.json()
    except Exception:
        data = {}

    # A Efí pode enviar lotes de recorrências/cobranças. Normaliza para uma lista.
    items = data.get("recs") or data.get("rec") or data.get("pix") or [data]
    if isinstance(items, dict):
        items = [items]

    for it in items:
        id_rec = it.get("idRec") or (it.get("vinculo") or {}).get("idRec")
        status = (it.get("status") or "").upper()
        if not id_rec:
            continue
        # Status de recorrência da Efí: APROVADA/ATIVA liberam; demais encerram.
        if status in ("APROVADA", "ATIVA", "CONFIRMADA"):
            await _set_plan_by_field("efiIdRec", id_rec, "premium", status)
            print(f"[efi] Premium ativado (idRec={id_rec})")
        elif status in ("CANCELADA", "REJEITADA", "EXPIRADA"):
            await _set_plan_by_field("efiIdRec", id_rec, "free", status)
            print(f"[efi] Assinatura encerrada (idRec={id_rec})")

    return {"received": True}

# ─── CATEGORIES ──────────────────────────────────────────────────────────────
@app.get("/api/categories")
async def get_categories(user=Depends(get_current_user)):
    cursor = db.categories.find({"userId": str(user["_id"])}).sort("order", 1)
    return [serialize(c) async for c in cursor]

@app.post("/api/categories")
async def create_category(body: CategoryCreate, user=Depends(get_current_user)):
    doc = {
        "userId":   str(user["_id"]),
        "name":     body.name,
        "parentId": body.parentId,
        "order":    body.order,
        "createdAt": datetime.utcnow()
    }
    result = await db.categories.insert_one(doc)
    return {"catId": str(result.inserted_id)}

@app.patch("/api/categories/{cat_id}")
async def update_category(cat_id: str, body: CategoryUpdate, user=Depends(get_current_user)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="Nada para atualizar")
    result = await db.categories.find_one_and_update(
        {"_id": ObjectId(cat_id), "userId": str(user["_id"])},
        {"$set": updates},
        return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Categoria não encontrada")
    return serialize(result)

@app.delete("/api/categories/by-name/{name}")
async def delete_category_by_name(name: str, user=Depends(get_current_user)):
    """
    Deleta TODAS as categorias do usuário com este nome (case-insensitive).
    Saída de emergência quando o delete por _id falha (ex: id em formato inválido).
    Também remove subcategorias (filhas recursivas) e mantém os links como órfãos.
    """
    uid = str(user["_id"])
    # Encontra todas com este nome (case-insensitive)
    found = await db.categories.find({
        "userId": uid,
        "name": {"$regex": f"^{name}$", "$options": "i"}
    }).to_list(None)

    if not found:
        return {"ok": True, "deleted": 0, "message": "Nenhuma categoria encontrada"}

    total_deleted = 0
    deleted_names = []

    async def recurse_delete(cid):
        nonlocal total_deleted
        # Apaga filhas primeiro
        children = await db.categories.find({"userId": uid, "parentId": cid}).to_list(None)
        for child in children:
            await recurse_delete(str(child["_id"]))
        # Apaga a categoria (links viram órfãos)
        try:
            r = await db.categories.delete_one({"_id": ObjectId(cid)})
            total_deleted += r.deleted_count
        except Exception as e:
            print(f"[by-name] delete_one falhou para {cid}: {e}")

    for cat in found:
        cid = str(cat["_id"])
        deleted_names.append(cat.get("name", "?"))
        await recurse_delete(cid)

    print(f"[by-name DELETE] user={uid} name='{name}' deleted_count={total_deleted} names={deleted_names}")
    return {"ok": True, "deleted": total_deleted, "names": deleted_names}

@app.post("/api/categories/nuke-all")
@limiter.limit("2/hour")
async def nuke_all_categories(request: Request, user=Depends(get_current_user)):
    """
    🧨 Apaga TODAS as categorias do usuário. Use com cuidado.
    Os links viram órfãos (aparecem em 'Sem categoria').
    """
    uid = str(user["_id"])
    r = await db.categories.delete_many({"userId": uid})
    print(f"[NUKE] user={uid} apagou {r.deleted_count} categorias")
    return {"ok": True, "deleted": r.deleted_count}

@app.delete("/api/categories/{cat_id}")
async def delete_category(cat_id: str, user=Depends(get_current_user)):
    uid = str(user["_id"])
    deleted_count = 0

    async def delete_recursive(cid: str):
        nonlocal deleted_count
        # Find children using all possible parentId formats
        query_filter = {"userId": uid, "parentId": cid}
        children = await db.categories.find(query_filter).to_list(None)
        for child in children:
            await delete_recursive(str(child["_id"]))
        # NOTE: Links are NOT deleted when category is deleted
        # They become orphaned and appear in "Sem Categoria" row in the app
        # This prevents accidental data loss
        # await db.links.delete_many({"userId": uid, "categoryId": cid})  # intentionally commented out
        # Attempt 1: standard _id + userId match
        try:
            r = await db.categories.delete_one({"_id": ObjectId(cid), "userId": uid})
            deleted_count += r.deleted_count
        except Exception as e:
            print(f"[DELETE] ObjectId attempt failed for {cid}: {e}")
        # Attempt 2: delete without userId constraint (in case userId format mismatch)
        if deleted_count == 0:
            try:
                r = await db.categories.delete_one({"_id": ObjectId(cid)})
                deleted_count += r.deleted_count
                print(f"[DELETE] Deleted {cid} without userId constraint")
            except Exception as e:
                print(f"[DELETE] No-userId attempt failed: {e}")

    await delete_recursive(cat_id)
    print(f"[DELETE] Category {cat_id}: deleted_count={deleted_count}")
    return {"ok": True, "deleted": deleted_count}

@app.put("/api/categories/batch")
async def batch_update_categories(body: dict, user=Depends(get_current_user)):
    """Reorder multiple categories at once."""
    updates = body.get("categories", [])
    for u in updates:
        cat_id = u.pop("catId", None)
        if cat_id:
            await db.categories.update_one(
                {"_id": ObjectId(cat_id), "userId": str(user["_id"])},
                {"$set": u}
            )
    cursor = db.categories.find({"userId": str(user["_id"])}).sort("order", 1)
    return [serialize(c) async for c in cursor]

# ─── LINKS ────────────────────────────────────────────────────────────────────

@app.get("/api/links/counts")
async def get_links_counts(user=Depends(get_current_user)):
    """Contagens leves — usado pela extensão Chrome para evitar carregar todos os links."""
    uid = str(user["_id"])
    total = await db.links.count_documents({"userId": uid})
    pipeline = [
        {"$match": {"userId": uid}},
        {"$group": {"_id": "$categoryId", "count": {"$sum": 1}}},
    ]
    by_cat = {}
    async for r in db.links.aggregate(pipeline):
        by_cat[r["_id"]] = r["count"]
    return {"total": total, "byCategory": by_cat}

@app.get("/api/links")
async def get_links(
    user=Depends(get_current_user),
    watched: Optional[bool] = None,
    limit: int = 100,
    skip: int = 0,
):
    """
    Retorna links paginados.
    - limit: máx 200 por chamada (default 100)
    - skip: offset
    - Resposta: { items, total, skip, limit, hasMore }
    """
    limit = min(limit, 200)
    query: dict = {"userId": str(user["_id"])}
    if watched is not None:
        query["watched"] = watched
    total   = await db.links.count_documents(query)
    cursor  = db.links.find(query).sort("createdAt", -1).skip(skip).limit(limit)
    items   = [serialize(l) async for l in cursor]
    return {
        "items":   items,
        "total":   total,
        "skip":    skip,
        "limit":   limit,
        "hasMore": (skip + limit) < total,
    }

@app.post("/api/links")
async def create_link(body: LinkCreate, user=Depends(get_current_user)):
    # Free plan limit
    if user.get("plan", "free") == "free":
        count = await db.links.count_documents({"userId": str(user["_id"])})
        if count >= 300:
            raise HTTPException(status_code=403, detail="Limite de 300 links no plano Free atingido")
    
    doc = {
        "userId":     str(user["_id"]),
        "url":        body.url,
        "title":      body.title,
        "thumbnail":  body.thumbnail,
        "rawThumb":   body.rawThumb,
        "platform":   body.platform,
        "videoId":    body.videoId,
        "categoryId": body.categoryId,
        "watched":    body.watched,
        "notes":      body.notes,
        "tags":       body.tags,
        "order":      body.order,
        "createdAt":  datetime.utcnow(),
        "watchedAt":  None,
        "durationSeconds": body.durationSeconds,
        "watchedSeconds":  body.watchedSeconds or 0,
        "lastWatchedAt":   body.lastWatchedAt,
        "watchCount":      body.watchCount or 0,
        "isFavorite":      bool(body.isFavorite),
    }
    result = await db.links.insert_one(doc)
    return {"linkId": str(result.inserted_id)}

@app.patch("/api/links/{link_id}")
async def update_link(link_id: str, body: LinkUpdate, user=Depends(get_current_user)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if body.watched is True and "watchedAt" not in updates:
        updates["watchedAt"] = datetime.utcnow()
    elif body.watched is False:
        updates["watchedAt"] = None
    
    result = await db.links.find_one_and_update(
        {"_id": ObjectId(link_id), "userId": str(user["_id"])},
        {"$set": updates},
        return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Link não encontrado")
    return serialize(result)

@app.delete("/api/links/{link_id}")
async def delete_link(link_id: str, user=Depends(get_current_user)):
    result = await db.links.delete_one(
        {"_id": ObjectId(link_id), "userId": str(user["_id"])}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Link não encontrado")
    return {"ok": True}

# ─── NOTE FOLDERS ─────────────────────────────────────────────────────────────
@app.get("/api/note-folders")
async def get_note_folders(user=Depends(get_current_user)):
    cursor = db.note_folders.find({"userId": str(user["_id"])}).sort("order", 1)
    return [serialize(f) async for f in cursor]

@app.post("/api/note-folders")
async def create_note_folder(body: NoteFolderCreate, user=Depends(get_current_user)):
    doc = {
        "userId":   str(user["_id"]),
        "name":     body.name.strip() or "Sem nome",
        "parentId": body.parentId,
        "color":    body.color,
        "order":    body.order,
        "createdAt": datetime.utcnow()
    }
    result = await db.note_folders.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize(doc)

@app.patch("/api/note-folders/{folder_id}")
async def update_note_folder(folder_id: str, body: NoteFolderUpdate, user=Depends(get_current_user)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="Nada para atualizar")
    result = await db.note_folders.find_one_and_update(
        {"_id": ObjectId(folder_id), "userId": str(user["_id"])},
        {"$set": updates},
        return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Pasta não encontrada")
    return serialize(result)

@app.delete("/api/note-folders/{folder_id}")
async def delete_note_folder(folder_id: str, user=Depends(get_current_user)):
    """
    Deleta pasta de notas e move todas as notas dela para a Caixa de entrada
    (folderId = None). Subpastas também são removidas em cascata.
    """
    uid = str(user["_id"])
    to_delete = [folder_id]
    queue = [folder_id]
    while queue:
        pid = queue.pop()
        async for child in db.note_folders.find({"userId": uid, "parentId": pid}):
            cid = str(child["_id"])
            if cid not in to_delete:
                to_delete.append(cid)
                queue.append(cid)

    # Move notas dessas pastas para inbox (folderId = None)
    await db.notes.update_many(
        {"userId": uid, "folderId": {"$in": to_delete}},
        {"$set": {"folderId": None}}
    )
    # Remove pastas
    obj_ids = [ObjectId(x) for x in to_delete]
    r = await db.note_folders.delete_many({"_id": {"$in": obj_ids}, "userId": uid})
    return {"ok": True, "deleted": r.deleted_count}

# ─── NOTES ────────────────────────────────────────────────────────────────────
@app.get("/api/notes")
async def get_notes(
    user=Depends(get_current_user),
    folderId:       Optional[str] = None,    # "__inbox__" filtra só Inbox; "__all__" = tudo
    includeDeleted: bool = False
):
    query: dict = {"userId": str(user["_id"])}
    if not includeDeleted:
        query["deletedAt"] = None
    else:
        query["deletedAt"] = {"$ne": None}

    if folderId == "__inbox__":
        query["folderId"] = None
    elif folderId and folderId != "__all__":
        query["folderId"] = folderId

    cursor = db.notes.find(query).sort("updatedAt", -1)
    return [serialize(n) async for n in cursor]

@app.post("/api/notes")
async def create_note(body: NoteCreate, user=Depends(get_current_user)):
    now = datetime.utcnow()
    folder_id = body.folderId if body.folderId not in ("__inbox__", "") else None
    # Default position: ms timestamp (newer = higher rank = appears at top)
    pos = body.position if body.position is not None else now.timestamp() * 1000
    doc = {
        "userId":       str(user["_id"]),
        "title":        body.title or "",
        "body":         body.body or "",
        "folderId":     folder_id,
        "linkedItemId": body.linkedItemId,
        "priority":     body.priority,
        "dueDate":      body.dueDate,
        "tags":         body.tags,
        "isCompleted":  body.isCompleted,
        "position":     pos,
        "deletedAt":    None,
        "createdAt":    now,
        "updatedAt":    now,
    }
    result = await db.notes.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize(doc)

@app.patch("/api/notes/{note_id}")
async def update_note(note_id: str, body: NoteUpdate, user=Depends(get_current_user)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="Nada para atualizar")
    # Normaliza inbox sentinel
    if updates.get("folderId") in ("__inbox__", ""):
        updates["folderId"] = None
    updates["updatedAt"] = datetime.utcnow()
    result = await db.notes.find_one_and_update(
        {"_id": ObjectId(note_id), "userId": str(user["_id"])},
        {"$set": updates},
        return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Nota não encontrada")
    return serialize(result)

@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str, user=Depends(get_current_user)):
    """Soft delete — move para lixeira (deletedAt = agora). Permanente após 30 dias."""
    result = await db.notes.find_one_and_update(
        {"_id": ObjectId(note_id), "userId": str(user["_id"])},
        {"$set": {"deletedAt": datetime.utcnow()}},
        return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Nota não encontrada")
    return {"ok": True}

@app.post("/api/notes/{note_id}/restore")
async def restore_note(note_id: str, user=Depends(get_current_user)):
    result = await db.notes.find_one_and_update(
        {"_id": ObjectId(note_id), "userId": str(user["_id"])},
        {"$set": {"deletedAt": None, "updatedAt": datetime.utcnow()}},
        return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Nota não encontrada")
    return serialize(result)

@app.delete("/api/notes/{note_id}/permanent")
async def delete_note_permanent(note_id: str, user=Depends(get_current_user)):
    """Apaga permanentemente da lixeira."""
    r = await db.notes.delete_one({"_id": ObjectId(note_id), "userId": str(user["_id"])})
    if r.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Nota não encontrada")
    return {"ok": True}

@app.post("/api/notes/empty-trash")
async def empty_trash(user=Depends(get_current_user)):
    """Esvazia toda a lixeira de notas."""
    r = await db.notes.delete_many({
        "userId": str(user["_id"]),
        "deletedAt": {"$ne": None}
    })
    return {"ok": True, "deleted": r.deleted_count}

# ─── HOME (Server-Driven UI) ──────────────────────────────────────────────────
# O backend decide QUAIS coleções e em que ORDEM aparecem (+ user_state). O front
# hidrata os itens com os dados locais frescos (data_source kind=collection).
def _parse_dt(v):
    if not v:
        return None
    if isinstance(v, datetime):
        return v.replace(tzinfo=None)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "")).replace(tzinfo=None)
    except Exception:
        return None

def _progress_pct(l: dict) -> int:
    if l.get("watched"):
        return 100
    d = l.get("durationSeconds") or 0
    w = l.get("watchedSeconds") or 0
    if d > 0:
        return max(0, min(100, round(w / d * 100)))
    return 0

def _video_status(l: dict) -> str:
    p = _progress_pct(l)
    if l.get("watched") or p >= 90:
        return "completed"
    if p < 5:
        return "not_started"
    lw = _parse_dt(l.get("lastWatchedAt"))
    days = (datetime.utcnow() - lw).days if lw else 0
    if p <= 85 and days > 14:
        return "abandoned"
    return "in_progress"

def _days_since(l: dict) -> float:
    c = _parse_dt(l.get("createdAt"))
    return (datetime.utcnow() - c).days if c else 9999

# Catálogo do sistema (espelha src/lib/collections.ts). min = min_items_to_show.
HOME_COLLECTIONS = [
    {"id": "continue",   "name": "Continue assistindo",     "min": 1, "pred": lambda l: _video_status(l) == "in_progress"},
    {"id": "almost",     "name": "Quase lá",                "min": 1, "pred": lambda l: 75 <= _progress_pct(l) < 90},
    {"id": "week",       "name": "Salvos esta semana",      "min": 3, "pred": lambda l: _days_since(l) <= 7},
    {"id": "abandoned",  "name": "Resgatar — abandonados",  "min": 2, "pred": lambda l: _video_status(l) == "abandoned"},
    {"id": "short",      "name": "Vídeos curtos",           "min": 3, "pred": lambda l: 0 < (l.get("durationSeconds") or 0) < 600},
    {"id": "notstarted", "name": "Ainda não assistidos",    "min": 4, "pred": lambda l: _video_status(l) == "not_started"},
]

@app.get("/api/home")
async def home_layout(user=Depends(get_current_user)):
    uid = str(user["_id"])
    cats = await db.categories.find({"userId": uid}).to_list(2000)
    cat_ids = {str(c["_id"]) for c in cats}
    links = await db.links.find({"userId": uid}).to_list(5000)
    valid = [l for l in links if l.get("categoryId") in cat_ids]

    sections = []
    order = 1
    for c in HOME_COLLECTIONS:
        n = sum(1 for l in valid if c["pred"](l))
        if n >= c["min"]:
            sections.append({
                "id": c["id"], "type": "smart_collection", "order": order,
                "title": c["name"],
                "data_source": {"kind": "collection", "collection_id": c["id"]},
            })
            order += 1

    user_state = "new" if len(links) < 3 else "returning"
    return {"version": 1, "user_state": user_state, "generated_at": datetime.utcnow().isoformat(), "sections": sections}

# ─── METADATA FETCH ──────────────────────────────────────────────────────────
def _yt_video_id(url: str) -> str:
    import re
    m = re.search(r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})", url or "")
    return m.group(1) if m else ""

def _iso8601_to_seconds(s: str) -> int:
    import re
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m:
        return 0
    h, mi, se = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + se

async def _yt_duration(video_id: str):
    """Duração (s) via YouTube Data API. Só funciona se YOUTUBE_API_KEY estiver setada."""
    if not YOUTUBE_API_KEY or not video_id:
        return None
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get("https://www.googleapis.com/youtube/v3/videos",
                            params={"id": video_id, "part": "contentDetails", "key": YOUTUBE_API_KEY})
            if r.status_code == 200:
                items = r.json().get("items", [])
                if items:
                    return _iso8601_to_seconds(items[0]["contentDetails"]["duration"])
    except Exception:
        pass
    return None

@app.post("/api/fetch-metadata")
@limiter.limit("20/minute")
async def fetch_metadata(request: Request, body: dict):
    """Fetch title + thumbnail for any URL (bypasses CORS for frontend)."""
    url = body.get("url", "")
    if not url:
        raise HTTPException(status_code=400, detail="URL obrigatória")
    
    # YouTube oEmbed
    if "youtube.com" in url or "youtu.be" in url:
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(f"https://www.youtube.com/oembed?url={url}&format=json")
                if r.status_code == 200:
                    d = r.json()
                    dur = await _yt_duration(_yt_video_id(url))
                    return {
                        "title":     d.get("title", ""),
                        "thumbnail": d.get("thumbnail_url", ""),
                        "platform":  "youtube",
                        "author":    d.get("author_name", ""),
                        "durationSeconds": dur,
                    }
        except:
            pass
    
    # Generic Open Graph fallback
    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as c:
            r = await c.get(url, headers={"User-Agent": "WatchListBot/1.0"})
            html = r.text[:50000]
            
            def og(prop):
                import re
                m = re.search(f'<meta[^>]+property=["\']og:{prop}["\'][^>]+content=["\']([^"\']+)', html)
                if m: return m.group(1)
                m = re.search(f'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:{prop}["\']', html)
                return m.group(1) if m else ""
            
            title = og("title") or ""
            if not title:
                import re
                m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
                title = m.group(1).strip() if m else ""
            
            return {
                "title":     title,
                "thumbnail": og("image"),
                "platform":  "other"
            }
    except Exception as e:
        return {"title": "", "thumbnail": "", "platform": "other", "error": str(e)}

# ─── ADMIN ─────────────────────────────────────────────────────────────────
# Todas as rotas exigem get_admin_user (e-mail em ADMIN_EMAILS).

@app.get("/api/admin/stats")
async def admin_stats(admin=Depends(get_admin_user)):
    """Métricas agregadas para o dashboard administrativo."""
    now   = datetime.utcnow()
    d1    = now - timedelta(days=1)
    d7    = now - timedelta(days=7)
    d30   = now - timedelta(days=30)

    total_users   = await db.users.count_documents({})
    premium_users = await db.users.count_documents({"plan": "premium"})
    new_1d        = await db.users.count_documents({"createdAt": {"$gte": d1}})
    new_7d        = await db.users.count_documents({"createdAt": {"$gte": d7}})
    new_30d       = await db.users.count_documents({"createdAt": {"$gte": d30}})
    active_7d     = await db.users.count_documents({"lastLogin": {"$gte": d7}})
    active_30d    = await db.users.count_documents({"lastLogin": {"$gte": d30}})

    total_links   = await db.links.count_documents({})
    total_cats    = await db.categories.count_documents({})
    total_notes   = await db.notes.count_documents({})

    free_users = total_users - premium_users
    conversion = round((premium_users / total_users) * 100, 1) if total_users else 0.0
    mrr        = round(premium_users * 19.0, 2)

    # Série de novos usuários por dia (últimos 14 dias)
    signups = []
    pipeline = [
        {"$match": {"createdAt": {"$gte": now - timedelta(days=14)}}},
        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}}, "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    async for r in db.users.aggregate(pipeline):
        signups.append({"date": r["_id"], "count": r["count"]})

    return {
        "users":   {"total": total_users, "premium": premium_users, "free": free_users,
                    "new1d": new_1d, "new7d": new_7d, "new30d": new_30d,
                    "active7d": active_7d, "active30d": active_30d},
        "content": {"links": total_links, "categories": total_cats, "notes": total_notes},
        "revenue": {"mrr": mrr, "conversionPct": conversion, "pricePerMonth": 19.0},
        "signups14d": signups,
    }

@app.get("/api/admin/users")
async def admin_list_users(
    admin=Depends(get_admin_user),
    search: Optional[str] = None,
    plan: Optional[str] = None,
    skip: int = 0,
    limit: int = 25,
):
    """Lista paginada de usuários com busca por nome/e-mail e filtro por plano."""
    limit = min(limit, 100)
    query: dict = {}
    if search:
        rx = {"$regex": search, "$options": "i"}
        query["$or"] = [{"name": rx}, {"email": rx}]
    if plan in ("free", "premium"):
        query["plan"] = plan

    total  = await db.users.count_documents(query)
    cursor = db.users.find(query).sort("createdAt", -1).skip(skip).limit(limit)
    users  = []
    async for u in cursor:
        uid = str(u["_id"])
        users.append({
            "id":        uid,
            "name":      u.get("name", ""),
            "email":     u.get("email", ""),
            "avatar":    u.get("avatar", ""),
            "plan":      u.get("plan", "free"),
            "planStatus": u.get("planStatus"),
            "createdAt": u["createdAt"].isoformat() if isinstance(u.get("createdAt"), datetime) else u.get("createdAt"),
            "lastLogin": u["lastLogin"].isoformat() if isinstance(u.get("lastLogin"), datetime) else u.get("lastLogin"),
            "isAdmin":   is_admin(u),
        })
    return {"items": users, "total": total, "skip": skip, "limit": limit, "hasMore": (skip + limit) < total}

@app.get("/api/admin/users/{user_id}")
async def admin_user_detail(user_id: str, admin=Depends(get_admin_user)):
    """Detalhe de um usuário, incluindo contagem de conteúdo."""
    try:
        u = await db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="ID inválido")
    if not u:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    uid = str(u["_id"])
    data = serialize(dict(u))
    data["counts"] = {
        "links":      await db.links.count_documents({"userId": uid}),
        "categories": await db.categories.count_documents({"userId": uid}),
        "notes":      await db.notes.count_documents({"userId": uid}),
        "backups":    await db.backups.count_documents({"userId": uid}),
    }
    data["isAdmin"] = is_admin(u)
    return data

class AdminUserUpdate(BaseModel):
    plan: Optional[str] = None   # "free" | "premium"

@app.patch("/api/admin/users/{user_id}")
async def admin_update_user(user_id: str, body: AdminUserUpdate, admin=Depends(get_admin_user)):
    """Atualiza o plano de um usuário manualmente (cortesia/suporte)."""
    update = {}
    if body.plan in ("free", "premium"):
        update["plan"] = body.plan
        update["planStatus"] = "admin_grant" if body.plan == "premium" else "admin_revoke"
    if not update:
        raise HTTPException(status_code=400, detail="Nada para atualizar.")
    try:
        r = await db.users.update_one({"_id": ObjectId(user_id)}, {"$set": update})
    except Exception:
        raise HTTPException(status_code=400, detail="ID inválido")
    if r.matched_count == 0:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return {"ok": True, "updated": update}

@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: str, admin=Depends(get_admin_user)):
    """Exclui um usuário e TODOS os seus dados. Irreversível."""
    if str(admin["_id"]) == user_id:
        raise HTTPException(status_code=400, detail="Você não pode excluir a própria conta admin aqui.")
    try:
        oid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="ID inválido")
    u = await db.users.find_one({"_id": oid})
    if not u:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    uid = user_id
    deleted = {
        "links":       (await db.links.delete_many({"userId": uid})).deleted_count,
        "categories":  (await db.categories.delete_many({"userId": uid})).deleted_count,
        "notes":       (await db.notes.delete_many({"userId": uid})).deleted_count,
        "noteFolders": (await db.note_folders.delete_many({"userId": uid})).deleted_count,
        "backups":     (await db.backups.delete_many({"userId": uid})).deleted_count,
    }
    await db.users.delete_one({"_id": oid})
    print(f"[admin] {admin.get('email')} excluiu usuário {uid} ({u.get('email')})")
    return {"ok": True, "deleted": deleted}

# ─── BACKUP / RESTORE ────────────────────────────────────────────────────────
AUTO_BACKUP_INTERVAL = timedelta(hours=24)  # 1 snapshot automático por dia
MAX_AUTO_BACKUPS     = 7                      # retenção de snapshots automáticos
MAX_MANUAL_BACKUPS   = 10                     # retenção de snapshots manuais

async def _collect_user_data(uid: str) -> dict:
    """Coleta todos os dados do usuário para um snapshot."""
    return {
        "categories":  [serialize(c) async for c in db.categories.find({"userId": uid})],
        "links":       [serialize(l) async for l in db.links.find({"userId": uid})],
        "notes":       [serialize(n) async for n in db.notes.find({"userId": uid})],
        "noteFolders": [serialize(f) async for f in db.note_folders.find({"userId": uid})],
    }

async def _prune_backups(uid: str, btype: str, keep: int):
    """Mantém apenas os 'keep' backups mais recentes de um tipo."""
    old = db.backups.find({"userId": uid, "type": btype}).sort("createdAt", -1).skip(keep)
    ids = [b["_id"] async for b in old]
    if ids:
        await db.backups.delete_many({"_id": {"$in": ids}})

async def _create_backup(uid: str, btype: str = "manual", label: str = None) -> dict:
    """Cria um snapshot e aplica a política de retenção. Retorna o metadata."""
    data = await _collect_user_data(uid)
    counts = {k: len(v) for k, v in data.items()}
    doc = {
        "userId":    uid,
        "type":      btype,
        "label":     label,
        "createdAt": datetime.utcnow(),
        "counts":    counts,
        "data":      data,
    }
    result = await db.backups.insert_one(doc)
    # Retenção por tipo
    if btype == "auto":
        await _prune_backups(uid, "auto", MAX_AUTO_BACKUPS)
    elif btype == "manual":
        await _prune_backups(uid, "manual", MAX_MANUAL_BACKUPS)
    return {
        "id":        str(result.inserted_id),
        "type":      btype,
        "label":     label,
        "createdAt": doc["createdAt"].isoformat(),
        "counts":    counts,
    }

async def maybe_auto_backup(uid: str):
    """Cria um backup automático se o último tem mais de 24h. Roda em background."""
    try:
        last = await db.backups.find_one(
            {"userId": uid, "type": "auto"}, sort=[("createdAt", -1)]
        )
        if last and (datetime.utcnow() - last["createdAt"]) < AUTO_BACKUP_INTERVAL:
            return
        # Não cria backup vazio para usuário sem dados ainda
        has_data = await db.links.count_documents({"userId": uid}) > 0 \
            or await db.categories.count_documents({"userId": uid}) > 0
        if not has_data:
            return
        await _create_backup(uid, "auto")
    except Exception as e:
        print(f"[auto-backup] falhou para {uid}: {e}")

@app.get("/api/backups")
async def list_backups(user=Depends(get_current_user)):
    """Lista os backups (apenas metadata — sem o payload pesado)."""
    uid = str(user["_id"])
    cursor = db.backups.find(
        {"userId": uid},
        {"data": 0},  # exclui o campo data (pesado)
    ).sort("createdAt", -1)
    out = []
    async for b in cursor:
        out.append({
            "id":        str(b["_id"]),
            "type":      b.get("type", "manual"),
            "label":     b.get("label"),
            "createdAt": b["createdAt"].isoformat() if isinstance(b["createdAt"], datetime) else b["createdAt"],
            "counts":    b.get("counts", {}),
        })
    return out

@app.post("/api/backups")
@limiter.limit("10/hour")
async def create_backup(request: Request, body: BackupCreate, user=Depends(get_current_user)):
    """Cria um backup manual sob demanda."""
    uid = str(user["_id"])
    meta = await _create_backup(uid, "manual", body.label)
    return {"ok": True, "backup": meta}

@app.get("/api/backups/{backup_id}")
async def get_backup(backup_id: str, user=Depends(get_current_user)):
    """Retorna um backup completo (para download/preview)."""
    uid = str(user["_id"])
    try:
        b = await db.backups.find_one({"_id": ObjectId(backup_id), "userId": uid})
    except Exception:
        raise HTTPException(status_code=400, detail="ID inválido")
    if not b:
        raise HTTPException(status_code=404, detail="Backup não encontrado")
    return {
        "id":        str(b["_id"]),
        "type":      b.get("type", "manual"),
        "label":     b.get("label"),
        "createdAt": b["createdAt"].isoformat() if isinstance(b["createdAt"], datetime) else b["createdAt"],
        "counts":    b.get("counts", {}),
        "data":      b.get("data", {}),
    }

@app.delete("/api/backups/{backup_id}")
async def delete_backup(backup_id: str, user=Depends(get_current_user)):
    uid = str(user["_id"])
    try:
        r = await db.backups.delete_one({"_id": ObjectId(backup_id), "userId": uid})
    except Exception:
        raise HTTPException(status_code=400, detail="ID inválido")
    if r.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Backup não encontrado")
    return {"ok": True}

@app.post("/api/backups/{backup_id}/restore")
@limiter.limit("10/hour")
async def restore_backup(backup_id: str, request: Request, user=Depends(get_current_user)):
    """
    Restaura os dados a partir de um backup.
    SEGURANÇA: antes de sobrescrever, cria um snapshot 'pre-restore' do estado
    atual — assim a restauração em si também é reversível.
    Preserva os _id originais para manter todas as referências intactas.
    """
    uid = str(user["_id"])
    try:
        b = await db.backups.find_one({"_id": ObjectId(backup_id), "userId": uid})
    except Exception:
        raise HTTPException(status_code=400, detail="ID inválido")
    if not b:
        raise HTTPException(status_code=404, detail="Backup não encontrado")

    # 1) Snapshot de segurança do estado atual (reversível)
    await _create_backup(uid, "pre-restore", "Antes da restauração")
    await _prune_backups(uid, "pre-restore", 3)

    data = b.get("data", {})
    collections = {
        "categories":  db.categories,
        "links":       db.links,
        "notes":       db.notes,
        "noteFolders": db.note_folders,
    }

    restored = {}
    for key, coll in collections.items():
        docs = data.get(key, [])
        # 2) Apaga o estado atual desta coleção
        await coll.delete_many({"userId": uid})
        # 3) Reinsere do snapshot (preservando _id e datas)
        if docs:
            prepared = []
            for d in docs:
                doc = deserialize_doc(d)
                doc["userId"] = uid  # garante posse
                prepared.append(doc)
            await coll.insert_many(prepared)
        restored[key] = len(docs)

    return {"ok": True, "restored": restored}

# ─── EXPORT / IMPORT / MIGRATE ───────────────────────────────────────────────
@app.get("/api/export")
async def export_data(user=Depends(get_current_user)):
    uid = str(user["_id"])
    cats  = [serialize(c) async for c in db.categories.find({"userId": uid})]
    links = [serialize(l) async for l in db.links.find({"userId": uid})]
    return {
        "version":    "2.0",
        "exportedAt": datetime.utcnow().isoformat(),
        "categories": cats,
        "links":      links
    }

@app.post("/api/migrate")
@limiter.limit("3/hour")
async def migrate_data(request: Request, body: MigrateRequest, user=Depends(get_current_user)):
    """
    Migração automática do localStorage para MongoDB no primeiro login.
    Chamado automaticamente pelo frontend quando is_new=True e há dados locais.
    Idempotente: verifica duplicatas por URL antes de inserir.
    """
    uid = str(user["_id"])

    # Verifica limite do plano free
    plan = user.get("plan", "free")

    # Filtra links já existentes (por URL) para evitar duplicatas em re-migrações
    existing_urls = set()
    async for link in db.links.find({"userId": uid}, {"url": 1}):
        existing_urls.add(link["url"])

    cats_inserted  = 0
    links_inserted = 0
    links_skipped  = 0

    # Migra categorias (upsert por nome para evitar duplicatas)
    cat_id_map = {}  # old_id → new_id (para remapear categoryId nos links)
    for cat in body.categories:
        old_id = cat.get("id") or cat.get("_id") or ""
        cat_doc = {
            "userId":   uid,
            "name":     cat.get("name", "Sem nome"),
            "parentId": cat.get("parentId"),
            "order":    cat.get("order", 0),
            "createdAt": datetime.utcnow()
        }
        # Tenta encontrar categoria com mesmo nome para evitar duplicata
        existing_cat = await db.categories.find_one({"userId": uid, "name": cat_doc["name"]})
        if existing_cat:
            cat_id_map[old_id] = str(existing_cat["_id"])
        else:
            result = await db.categories.insert_one(cat_doc)
            cat_id_map[old_id] = str(result.inserted_id)
            cats_inserted += 1

    # Migra links (pula duplicatas por URL)
    for link in body.links:
        url = link.get("url", "")
        if url in existing_urls:
            links_skipped += 1
            continue

        # Verifica limite free
        if plan == "free":
            current_count = await db.links.count_documents({"userId": uid})
            if current_count >= 300:
                links_skipped += (len(body.links) - links_inserted - links_skipped)
                break

        old_cat_id = link.get("categoryId", "")
        new_cat_id = cat_id_map.get(old_cat_id, old_cat_id)

        link_doc = {
            "userId":     uid,
            "url":        url,
            "title":      link.get("title", ""),
            "thumbnail":  link.get("thumbnail", ""),
            "rawThumb":   link.get("rawThumb", ""),
            "platform":   link.get("platform", "other"),
            "videoId":    link.get("videoId", ""),
            "categoryId": new_cat_id,
            "watched":    link.get("watched", False),
            "notes":      link.get("notes", ""),
            "tags":       link.get("tags", []),
            "order":      link.get("order", 0),
            "createdAt":  datetime.fromisoformat(link["createdAt"]) if link.get("createdAt") else datetime.utcnow(),
            "watchedAt":  datetime.fromisoformat(link["watchedAt"]) if link.get("watchedAt") else None,
            "migratedFrom": "localStorage"
        }
        await db.links.insert_one(link_doc)
        existing_urls.add(url)
        links_inserted += 1

    # Marca usuário como migrado
    await db.users.update_one({"_id": user["_id"]}, {"$set": {"migratedAt": datetime.utcnow()}})

    return {
        "ok": True,
        "imported": {
            "categories": cats_inserted,
            "links":      links_inserted,
            "skipped":    links_skipped
        }
    }

# ─── STARTUP ─────────────────────────────────────────────────────────────────
@app.get("/api/auth/migration-status")
async def migration_status(user=Depends(get_current_user)):
    """Verifica se o usuário já teve os dados migrados."""
    return {
        "migrated": bool(user.get("migratedAt")),
        "is_new":   not bool(user.get("migratedAt")) and (
            await db.links.count_documents({"userId": str(user["_id"])}) == 0
        )
    }

@app.on_event("startup")
async def startup():
    # Grava o certificado mTLS da Efí em disco (a partir do base64 em env var)
    global _EFI_CERT_PATH
    if EFI_CERT_BASE64:
        try:
            import base64, tempfile, os as _os
            pem = base64.b64decode(EFI_CERT_BASE64)
            path = _os.path.join(tempfile.gettempdir(), "efi_cert.pem")
            with open(path, "wb") as f:
                f.write(pem)
            _EFI_CERT_PATH = path
            print("✅ Certificado Efí carregado")
        except Exception as e:
            print(f"[startup] falha ao carregar certificado Efí: {e}")

    # Create indexes
    await db.users.create_index("email", unique=True)
    await db.categories.create_index([("userId", 1), ("order", 1)])
    await db.links.create_index([("userId", 1), ("createdAt", -1)])
    await db.links.create_index([("userId", 1), ("categoryId", 1)])
    # Notes indexes
    await db.notes.create_index([("userId", 1), ("position", -1)])
    await db.notes.create_index([("userId", 1), ("updatedAt", -1)])
    await db.notes.create_index([("userId", 1), ("folderId", 1)])
    await db.notes.create_index([("userId", 1), ("deletedAt", 1)])
    await db.notes.create_index([("userId", 1), ("linkedItemId", 1)])
    await db.note_folders.create_index([("userId", 1), ("order", 1)])
    # Backups
    await db.backups.create_index([("userId", 1), ("type", 1), ("createdAt", -1)])

    # Migração: backfill 'position' em notas antigas (a partir do updatedAt em ms)
    try:
        await db.notes.update_many(
            {"position": {"$exists": False}},
            [{"$set": {"position": {"$toLong": "$updatedAt"}}}]
        )
    except Exception as e:
        print(f"[startup] position backfill skipped: {e}")

    print("✅ WatchList API iniciada — MongoDB conectado (com Notes + reorder)")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
