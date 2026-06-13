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
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")   # opcional: IA via OpenAI
OPENAI_CHAT_MODEL      = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL     = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
# Gemini (Google AI Studio) — preferido se setado; free tier faz tags + embeddings.
# Aceita VÁRIAS chaves (cota diária separada por projeto): GEMINI_API_KEYS com
# chaves separadas por vírgula, OU a antiga GEMINI_API_KEY (uma só).
GEMINI_API_KEY         = os.getenv("GEMINI_API_KEY", "")
GEMINI_KEYS            = [k.strip() for k in (os.getenv("GEMINI_API_KEYS", "") or GEMINI_API_KEY).split(",") if k.strip()]
GEMINI_CHAT_MODEL      = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")
# Fallback de MODELOS: se o primário esgotar a cota do dia (PerDay), tenta o próximo.
# A cota grátis costuma ser POR MODELO, então trocar de modelo ainda rende.
GEMINI_CHAT_FALLBACKS  = os.getenv("GEMINI_CHAT_FALLBACKS", "gemini-2.0-flash,gemini-flash-latest,gemini-1.5-flash")
GEMINI_CHAT_MODELS     = list(dict.fromkeys(
    [GEMINI_CHAT_MODEL] + [m.strip() for m in GEMINI_CHAT_FALLBACKS.split(",") if m.strip()]))
GEMINI_EMBED_MODEL     = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_EMBED_DIMS      = int(os.getenv("GEMINI_EMBED_DIMS", "768"))
# Groq (provedor de CHAT — Llama etc., API compatível com OpenAI, free e rápido).
# NÃO faz embeddings (esses ficam no Gemini). Chaves em GROQ_API_KEYS (vírgula).
GROQ_API_KEY           = os.getenv("GROQ_API_KEY", "")
GROQ_KEYS              = [k.strip() for k in (os.getenv("GROQ_API_KEYS", "") or GROQ_API_KEY).split(",") if k.strip()]
GROQ_MODELS            = list(dict.fromkeys([m.strip() for m in os.getenv(
    "GROQ_MODELS", "llama-3.3-70b-versatile,llama-3.1-8b-instant").split(",") if m.strip()]))
_groq_exhausted: dict  = {}   # (chave, modelo) -> 'YYYY-MM-DD' (cota diária esgotada)

# Cota esgotada HOJE rastreada por (chave, modelo) -> 'YYYY-MM-DD'. Como o limite
# grátis é por MODELO, uma chave esgotada no 2.5-flash-lite ainda serve no 2.0-flash.
# Reseta sozinho ao virar o dia (UTC).
_gemini_exhausted: dict = {}

def _is_daily_quota_429(data: dict) -> bool:
    """Distingue um 429 de cota DIÁRIA (PerDay → chave esgotada o dia todo) de um
    429 de limite POR MINUTO (PerMinute → só um soluço transitório; NÃO queima a
    chave). No corpo do erro do Gemini, as violações de cota trazem um quotaId/
    quotaMetric; só o que contém 'PerDay' significa que o dia acabou de verdade.
    Na dúvida (sem detalhes), tratamos como transitório p/ não matar a chave à toa."""
    try:
        for d in (data.get("error", {}).get("details") or []):
            for v in (d.get("violations") or []):
                qid = (v.get("quotaId") or "") + "|" + (v.get("quotaMetric") or "")
                if "PerDay" in qid:
                    return True
    except Exception:
        pass
    return False

def _ai_enabled() -> bool:
    return bool(GEMINI_KEYS or GROQ_KEYS or OPENAI_API_KEY)

def _gemini_keys_available(model: str) -> list:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return [k for k in GEMINI_KEYS if _gemini_exhausted.get((k, model)) != today]

async def _gemini_post(path: str, body: dict, model: str, timeout: float = 25):
    """POST na API do Gemini com ROTAÇÃO de chaves para um MODELO. Em 429 de cota
    DIÁRIA marca (chave, modelo) como esgotado hoje e passa à próxima chave. Tenta 2x
    por chave. Retorna o JSON da 1ª resposta 200, ou None se todas falharem/esgotarem."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://generativelanguage.googleapis.com/v1beta/{path}"
    async with httpx.AsyncClient(timeout=timeout) as c:
        for key in _gemini_keys_available(model):
            for _attempt in range(2):
                try:
                    r = await c.post(url, params={"key": key}, json=body)
                except Exception:
                    continue
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    try: edata = r.json()
                    except Exception: edata = {}
                    if _is_daily_quota_429(edata):     # cota DIÁRIA (deste modelo) acabou
                        _gemini_exhausted[(key, model)] = today
                    break                              # → próxima chave (não queima em limite/min)
                # outro erro: tenta a 2ª vez; persistindo, vai p/ a próxima chave
    return None

def _gemini_text(j: dict) -> str:
    """Extrai o texto da resposta do Gemini com segurança. Os modelos 2.5 podem
    voltar SEM parts (ex.: gastaram o orçamento em 'thinking', finishReason
    MAX_TOKENS) — nesse caso devolve '' em vez de quebrar com KeyError."""
    try:
        cand = (j.get("candidates") or [])[0]
        parts = ((cand.get("content") or {}).get("parts")) or []
        return "".join(p.get("text", "") for p in parts).strip()
    except Exception:
        return ""

# Desliga o "thinking" dos modelos 2.5 (senão consomem o maxOutputTokens pensando
# e voltam sem texto). Mesclado no generationConfig das chamadas de texto.
_GEN_NO_THINK = {"thinkingConfig": {"thinkingBudget": 0}}

async def _gemini_chat(prompt: str, max_tokens: int, temperature: float = 0.3, timeout: float = 35, models=None):
    """Geração de texto com FALLBACK de modelos: tenta cada modelo de `models` (default
    GEMINI_CHAT_MODELS), cada um rotacionando as chaves, até obter TEXTO. Robusto a cota
    diária por modelo. Retorna o texto (str) ou '' se nada funcionar.
    IMPORTANTE: `thinkingConfig` só existe nos modelos 2.5 — enviá-lo a 2.0/1.5 dá 400 e
    QUEBRA o fallback. Por isso o body é montado POR MODELO."""
    for model in (models or GEMINI_CHAT_MODELS):
        gen = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if "2.5" in model:
            gen["thinkingConfig"] = {"thinkingBudget": 0}   # 2.5 pensa e come os tokens → desliga
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}
        j = await _gemini_post(f"models/{model}:generateContent", body, model, timeout=timeout)
        if j:
            txt = _gemini_text(j)
            if txt:
                return txt
    return ""

# Modelos para DECISÕES que exigem nuance (categoria/tags sugeridas ao usuário):
# começa por um modelo mais forte e cai pros menores. Dedup com a lista padrão.
GEMINI_SMART_MODELS = list(dict.fromkeys(["gemini-2.5-flash", "gemini-2.0-flash"] + GEMINI_CHAT_MODELS))

def _groq_keys_available(model: str) -> list:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return [k for k in GROQ_KEYS if _groq_exhausted.get((k, model)) != today]

async def _groq_chat(prompt: str, max_tokens: int, temperature: float = 0.3, timeout: float = 35):
    """Chat via Groq (API compatível com OpenAI) com rotação de chaves + fallback de
    modelos. 429 de cota DIÁRIA marca (chave,modelo); 429 por-minuto é transitório."""
    if not GROQ_KEYS:
        return ""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    async with httpx.AsyncClient(timeout=timeout) as c:
        for model in GROQ_MODELS:
            for key in _groq_keys_available(model):
                for _attempt in range(2):
                    try:
                        r = await c.post("https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {key}"},
                            json={"model": model, "max_tokens": max_tokens, "temperature": temperature,
                                  "messages": [{"role": "user", "content": prompt}]})
                    except Exception:
                        continue
                    if r.status_code == 200:
                        try:
                            return (r.json()["choices"][0]["message"]["content"] or "").strip()
                        except Exception:
                            return ""
                    if r.status_code == 429:
                        msg = ""
                        try: msg = str((r.json().get("error") or {}).get("message", "")).lower()
                        except Exception: pass
                        if any(s in msg for s in ("per day", "daily", "rpd", "tpd")):
                            _groq_exhausted[(key, model)] = today
                        break   # → próxima chave (não queima em limite/min)
                    # outro erro: tenta 2ª vez; persistindo, vai p/ a próxima chave
    return ""

async def _openai_chat(prompt: str, max_tokens: int, temperature: float = 0.3, timeout: float = 35):
    if not OPENAI_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": OPENAI_CHAT_MODEL, "max_tokens": max_tokens, "temperature": temperature,
                      "messages": [{"role": "user", "content": prompt}]})
            if r.status_code == 200:
                return (r.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        pass
    return ""

async def _chat(prompt: str, max_tokens: int, temperature: float = 0.3, timeout: float = 35, models=None):
    """Texto/chat com MÚLTIPLOS provedores em cascata: Gemini → Groq → OpenAI (o que
    estiver configurado/responder). `models` aplica-se só à etapa Gemini."""
    if GEMINI_KEYS:
        t = await _gemini_chat(prompt, max_tokens, temperature, timeout, models)
        if t: return t
    if GROQ_KEYS:
        t = await _groq_chat(prompt, max_tokens, temperature, timeout)
        if t: return t
    if OPENAI_API_KEY:
        t = await _openai_chat(prompt, max_tokens, temperature, timeout)
        if t: return t
    return ""
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
    doc.pop("topicEmbedding", None)  # vetor grande — fica só no servidor
    doc.pop("contentText", None)     # texto extraído (grande) — só no servidor (busca/RAG)
    doc.pop("noteEmbedding", None)   # vetor da nota — só no servidor (RAG)
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

class StatePut(BaseModel):
    value: Any = None

class BridgeReq(BaseModel):
    a: str
    b: str

class GeminiReq(BaseModel):
    model: Optional[str] = None
    body: dict = {}

class SearchReq(BaseModel):
    q: str = ""
    limit: int = 40

class SuggestCatReq(BaseModel):
    title: str = ""
    url: str = ""
    description: str = ""

class SuggestTagsReq(BaseModel):
    title: str = ""
    url: str = ""
    description: str = ""

class TeachReq(BaseModel):
    advice: str = ""

class AskReq(BaseModel):
    q: str = ""
    history: list = []   # [{"q": "...", "a": "..."}] — turnos anteriores (memória do chat)

class RulesReq(BaseModel):
    rules: list = []

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

@app.get("/api/state/{key}")
async def get_state(key: str, user=Depends(get_current_user)):
    """Armazenamento genérico por usuário (sincroniza estado entre dispositivos)."""
    doc = await db.appstate.find_one({"userId": str(user["_id"]), "key": key})
    return {"key": key, "value": (doc or {}).get("value"), "updatedAt": (doc or {}).get("updatedAt")}

@app.put("/api/state/{key}")
async def put_state(key: str, body: StatePut, user=Depends(get_current_user)):
    await db.appstate.update_one(
        {"userId": str(user["_id"]), "key": key},
        {"$set": {"value": body.value, "updatedAt": datetime.utcnow()}},
        upsert=True,
    )
    return {"ok": True}

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

@app.get("/api/links/exists")
async def link_exists(url: str = "", videoId: str = "", user=Depends(get_current_user)):
    """Checa se um link JÁ está salvo (p/ a extensão indicar como o ⭐ dos favoritos).
    É só uma CONSULTA AO BANCO — NÃO usa IA, NÃO gasta token. Custo ~nada."""
    uid = str(user["_id"])
    vid = (videoId or _yt_video_id(url)) if (videoId or url) else ""
    if not (vid or url):
        return {"saved": False}
    q = {"userId": uid, "videoId": vid} if vid else {"userId": uid, "urlKey": _url_key(url)}
    doc = await db.links.find_one(q, {"categoryId": 1, "title": 1})
    if not doc:
        return {"saved": False}
    cat = ""
    if doc.get("categoryId"):
        c = await db.categories.find_one({"_id": ObjectId(doc["categoryId"]), "userId": uid}, {"name": 1})
        cat = (c or {}).get("name", "")
    return {"saved": True, "id": str(doc["_id"]), "title": doc.get("title", ""), "category": cat}

@app.post("/api/links")
async def create_link(body: LinkCreate, background: BackgroundTasks, user=Depends(get_current_user)):
    # Free plan limit
    if user.get("plan", "free") == "free":
        count = await db.links.count_documents({"userId": str(user["_id"])})
        if count >= 300:
            raise HTTPException(status_code=403, detail="Limite de 300 links no plano Free atingido")

    # Dedup canônico: videoId (vídeos) OU urlKey normalizado (ignora www/barra/
    # protocolo/tracking) → idempotente mesmo com variações da mesma URL.
    ukey = _url_key(body.url)
    dup_q = ({"userId": str(user["_id"]), "videoId": body.videoId}
             if body.videoId else {"userId": str(user["_id"]), "urlKey": ukey})
    existing = await db.links.find_one(dup_q)
    if existing:
        return {"linkId": str(existing["_id"]), "duplicate": True}

    doc = {
        "userId":     str(user["_id"]),
        "url":        body.url,
        "urlKey":     ukey,
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
    # IA: enriquece tags + embedding em background (se houver chave de IA)
    if _ai_enabled():
        background.add_task(_enrich_link, str(result.inserted_id), str(user["_id"]))
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
    await db.chunks.delete_many({"linkId": link_id, "userId": str(user["_id"])})
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
async def create_note(body: NoteCreate, background: BackgroundTasks, user=Depends(get_current_user)):
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
    if _ai_enabled():
        background.add_task(_embed_note, str(result.inserted_id), str(user["_id"]))
    return serialize(doc)

@app.patch("/api/notes/{note_id}")
async def update_note(note_id: str, body: NoteUpdate, background: BackgroundTasks, user=Depends(get_current_user)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="Nada para atualizar")
    # Texto mudou → re-embeda em background (o RAG responde com a versão atual)
    if _ai_enabled() and ("title" in updates or "body" in updates):
        background.add_task(_embed_note, note_id, str(user["_id"]))
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
    cats = await db.categories.find({"userId": uid}, {"_id": 1}).to_list(2000)
    cat_ids = {str(c["_id"]) for c in cats}
    # Projeção: as seções da home só usam status/recência/vetor — sem contentText.
    links = await db.links.find({"userId": uid},
        {"categoryId": 1, "topicEmbedding": 1, "title": 1, "watched": 1, "watchedAt": 1,
         "watchedSeconds": 1, "durationSeconds": 1, "lastWatchedAt": 1, "createdAt": 1}).to_list(5000)
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

    # Semântico (Fase 2): "Porque você assistiu X" — similaridade de embedding ao
    # último vídeo assistido. Manda só os IDs; o front hidrata com dados locais.
    seeds = [l for l in valid if l.get("topicEmbedding") and (l.get("watched") or _progress_pct(l) >= 50)]
    if seeds:
        def _cos(a, b):
            if not a or not b:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0
        src = max(seeds, key=lambda l: _parse_dt(l.get("lastWatchedAt")) or _parse_dt(l.get("watchedAt")) or datetime.min)
        sims = []
        for l in valid:
            if str(l["_id"]) == str(src["_id"]) or not l.get("topicEmbedding"):
                continue
            sims.append((str(l["_id"]), _cos(src["topicEmbedding"], l["topicEmbedding"])))
        sims.sort(key=lambda x: x[1], reverse=True)
        top = [lid for lid, sc in sims[:12] if sc > 0.3]
        if len(top) >= 3:
            title = (src.get("title") or "")[:40]
            sections.insert(min(2, len(sections)), {
                "id": "because", "type": "because_you_watched", "order": 0,
                "title": f'Porque você assistiu "{title}"',
                "data_source": {"kind": "ids", "ids": top},
            })

    # ── "Segundo cérebro": trilha, ângulo oposto, esquecidos, duplicados ──
    emb_items = [l for l in valid if l.get("topicEmbedding")]
    watched_seeds = [l for l in emb_items if (l.get("watched") or _progress_pct(l) >= 50)]
    unwatched = [l for l in emb_items if _video_status(l) == "not_started"]

    def _add(sid, title, subtitle, ids):
        nonlocal order
        ids = list(dict.fromkeys(ids))[:14]
        if len(ids) >= 3:
            sections.append({"id": sid, "type": "ai_reco", "order": order, "title": title,
                             "subtitle": subtitle, "data_source": {"kind": "ids", "ids": ids}})
            order += 1

    if watched_seeds and len(unwatched) >= 3:
        dim = len(watched_seeds[0]["topicEmbedding"])
        cen = [0.0] * dim
        for l in watched_seeds:
            e = l["topicEmbedding"]
            for i in range(min(dim, len(e))):
                cen[i] += e[i]
        cen = [x / len(watched_seeds) for x in cen]
        scored = sorted(((_cosine(cen, l["topicEmbedding"]), str(l["_id"])) for l in unwatched), reverse=True)
        # Continue a trilha: não-assistidos MAIS próximos do que você já viu.
        _add("trilha", "Continue sua trilha", "próximos passos no que você vem estudando",
             [lid for sc, lid in scored if sc > 0.32])
        # Ângulo oposto: os MENOS parecidos (sair da bolha).
        _add("oposto", "Explore um ângulo diferente", "fora da sua bolha — amplie horizontes",
             [lid for sc, lid in reversed(scored) if sc < 0.20])

    # Esquecidos: salvos há tempo e nunca vistos.
    forgotten = sorted([l for l in valid if _video_status(l) == "not_started" and _days_since(l) > 60],
                       key=_days_since, reverse=True)
    _add("esquecidos", "Esquecidos", "salvos há tempos e nunca vistos", [str(l["_id"]) for l in forgotten])

    # Duplicados: pares quase idênticos (cosseno > 0.93). Amostra capada p/ latência.
    pool = emb_items[:180]
    dup = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if _cosine(pool[i]["topicEmbedding"], pool[j]["topicEmbedding"]) > 0.93:
                dup.append(str(pool[i]["_id"])); dup.append(str(pool[j]["_id"]))
    _add("dups", "Possíveis duplicados", "conteúdos muito parecidos — vale revisar", dup)

    user_state = "new" if len(links) < 3 else "returning"
    return {"version": 1, "user_state": user_state, "generated_at": datetime.utcnow().isoformat(), "sections": sections}

# ─── IA (Fase 2): auto-tagging por LLM + embeddings (OpenAI) ───────────────────
import math, json as _json, re as _re, asyncio

_TAG_SYS = ("Você gera TAGS DE TÓPICO para um vídeo a partir do título. Responda APENAS "
            "um array JSON com 2 a 4 tags curtas em português (1-2 palavras, Capitalizadas), "
            "sem texto extra. Ex: [\"Marketing\",\"Facebook Ads\"]")

def _extract_tags(txt: str) -> list:
    m = _re.search(r"\[.*\]", txt or "", _re.S)
    if not m:
        return []
    try:
        arr = _json.loads(m.group(0))
        return [str(t).strip() for t in arr if str(t).strip()][:4]
    except Exception:
        return []

async def _ai_tags(title: str) -> list:
    if not title or not _ai_enabled():
        return []
    try:
        txt = await _chat(f"{_TAG_SYS}\n\nTítulo: {title}", 200, temperature=0.2, timeout=25)
        return _extract_tags(txt) if txt else []
    except Exception:
        return []

async def _ai_embedding(text: str):
    if not text or not _ai_enabled():
        return None
    try:
        if GEMINI_KEYS:
            j = await _gemini_post(f"models/{GEMINI_EMBED_MODEL}:embedContent",
                {"model": f"models/{GEMINI_EMBED_MODEL}", "content": {"parts": [{"text": text[:4000]}]},
                 "outputDimensionality": GEMINI_EMBED_DIMS}, GEMINI_EMBED_MODEL)
            if j:
                return j["embedding"]["values"]
        else:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.post("https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={"model": OPENAI_EMBED_MODEL, "input": text[:4000]})
                if r.status_code == 200:
                    return r.json()["data"][0]["embedding"]
    except Exception:
        pass
    return None

async def _ai_text(prompt: str, max_tokens: int = 600) -> str:
    """Geração de texto livre (resumos) — Gemini → Groq → OpenAI."""
    if not _ai_enabled():
        return ""
    try:
        return await _chat(prompt, max_tokens, temperature=0.3, timeout=35)
    except Exception:
        return ""

async def _yt_description(video_id: str) -> str:
    if not YOUTUBE_API_KEY or not video_id:
        return ""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://www.googleapis.com/youtube/v3/videos",
                            params={"id": video_id, "part": "snippet", "key": YOUTUBE_API_KEY})
            if r.status_code == 200:
                items = r.json().get("items", [])
                if items:
                    return items[0]["snippet"].get("description", "")
    except Exception:
        pass
    return ""

def _html_to_text(html: str) -> str:
    """Readability simples: foca no <article>/<main>, remove ruído e tags → texto."""
    import re, html as _h
    m = (re.search(r"<article[^>]*>(.*?)</article>", html, re.S | re.I)
         or re.search(r"<main[^>]*>(.*?)</main>", html, re.S | re.I))
    chunk = m.group(1) if m else html
    chunk = re.sub(r"<(script|style|nav|header|footer|aside|form|noscript)[^>]*>.*?</\1>", " ", chunk, flags=re.S | re.I)
    chunk = re.sub(r"<br\s*/?>", "\n", chunk, flags=re.I)
    chunk = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", chunk, flags=re.I)
    chunk = re.sub(r"<[^>]+>", " ", chunk)
    chunk = _h.unescape(chunk)
    chunk = re.sub(r"[ \t]+", " ", chunk)
    chunk = re.sub(r"\n\s*\n+", "\n\n", chunk)
    return chunk.strip()

def _parse_timedtext(xml: str) -> list:
    """Parse PURO do XML do timedtext → [{"t": segundos, "text": ...}]. Preserva os
    TIMESTAMPS (antes jogados fora) — são eles que permitem citar o minuto do vídeo."""
    import html as _h
    segs = []
    for m in _re.finditer(r'<text\s+[^>]*start="([\d.]+)"[^>]*>(.*?)</text>', xml or "", _re.S):
        txt = _re.sub(r"\s+", " ", _h.unescape(_re.sub(r"<[^>]+>", " ", m.group(2)))).strip()
        if txt:
            segs.append({"t": float(m.group(1)), "text": txt})
    return segs

async def _yt_transcript_segments(video_id: str) -> list:
    """Best-effort: transcrição COM timestamps via timedtext (sem OAuth). [] se não houver."""
    if not video_id:
        return []
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as c:
            r = await c.get(f"https://www.youtube.com/watch?v={video_id}",
                            headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "pt-BR,pt,en"})
            urls = _re.findall(r'"baseUrl":"(https://www\.youtube\.com/api/timedtext[^"]+)"', r.text)
            if not urls:
                return []
            base = urls[0].replace("\\u0026", "&").replace("\\/", "/")
            tr = await c.get(base)
            return _parse_timedtext(tr.text)
    except Exception:
        return []

async def _yt_transcript(video_id: str) -> str:
    """Transcrição como texto único (compat — resumo/embedding do link inteiro)."""
    segs = await _yt_transcript_segments(video_id)
    return " ".join(s["text"] for s in segs).strip()

def _chunk_segments(segments: list, target_chars: int = 700, cap: int = 24) -> list:
    """Agrupa segmentos timestampados em CHUNKS ~target_chars, carregando o tempo de
    início do 1º segmento → [{"t": seg|None, "text": ...}]. Cap protege a cota."""
    chunks, buf, t0 = [], [], None
    size = 0
    for s in segments:
        if t0 is None:
            t0 = s["t"]
        buf.append(s["text"]); size += len(s["text"]) + 1
        if size >= target_chars:
            chunks.append({"t": int(t0), "text": " ".join(buf)})
            buf, t0, size = [], None, 0
            if len(chunks) >= cap:
                return chunks
    if buf and len(chunks) < cap:
        chunks.append({"t": int(t0), "text": " ".join(buf)})
    return chunks

def _chunk_text(text: str, target_chars: int = 800, cap: int = 24) -> list:
    """Divide texto corrido (artigo/post) em chunks por sentença → [{"t": None, ...}]."""
    text = (text or "").strip()
    if not text:
        return []
    sentences = _re.split(r"(?<=[.!?…])\s+|\n{2,}", text)
    chunks, buf, size = [], [], 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        buf.append(s); size += len(s) + 1
        if size >= target_chars:
            chunks.append({"t": None, "text": " ".join(buf)})
            buf, size = [], 0
            if len(chunks) >= cap:
                return chunks
    if buf and len(chunks) < cap:
        chunks.append({"t": None, "text": " ".join(buf)})
    return chunks

async def _ai_embeddings_batch(texts: list) -> list:
    """Embeddings em LOTE (1 chamada p/ até 100 textos via batchEmbedContents) —
    essencial p/ chunks sem estourar a cota. Devolve lista alinhada (None onde falhou)."""
    if not texts:
        return []
    if GEMINI_KEYS:
        out = []
        for i in range(0, len(texts), 100):
            lote = texts[i:i + 100]
            j = await _gemini_post(f"models/{GEMINI_EMBED_MODEL}:batchEmbedContents",
                {"requests": [{"model": f"models/{GEMINI_EMBED_MODEL}",
                               "content": {"parts": [{"text": t[:4000]}]},
                               "outputDimensionality": GEMINI_EMBED_DIMS} for t in lote]},
                GEMINI_EMBED_MODEL, timeout=60)
            embs = (j or {}).get("embeddings") or []
            out.extend([(e or {}).get("values") for e in embs] + [None] * (len(lote) - len(embs)))
        return out
    return [await _ai_embedding(t) for t in texts]   # fallback sequencial (OpenAI)

# UA de navegador real: muitos sites bloqueiam "bots" declarados; e UA de crawler
# de preview: paywalls/redes fechadas costumam servir as metatags OG pra ele.
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
_CRAWLER_UA = "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)"

def _unwrap_url(url: str) -> str:
    """Desembrulha redirects (Google, Facebook, Instagram, Reddit) → URL real."""
    try:
        from urllib.parse import urlparse, parse_qs, unquote
        p = urlparse(url or "")
        host, qs = p.netloc.lower(), parse_qs(p.query)
        if "google." in host and p.path == "/url":
            for k in ("q", "url"):
                if qs.get(k):
                    return unquote(qs[k][0])
        if host in ("l.facebook.com", "lm.facebook.com", "l.instagram.com") and qs.get("u"):
            return unquote(qs["u"][0])
        if host == "out.reddit.com" and qs.get("url"):
            return unquote(qs["url"][0])
    except Exception:
        pass
    return url

_TRACK_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                 "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "mc_eid", "mc_cid",
                 "igshid", "si", "feature", "ref", "ref_src", "spm", "_ga", "yclid", "ttclid"}

def _url_key(url: str) -> str:
    """Chave CANÔNICA p/ dedup: ignora protocolo, www/m/mobile, barra final, caixa do
    host, porta, fragmento e querystring de tracking. Faz
    'youtube.com/x', 'www.youtube.com/x/', 'https://youtube.com/x', 'youtube.com/x#t'
    virarem a MESMA chave. Para YouTube, ainda casa pelo videoId quando houver."""
    try:
        from urllib.parse import urlparse, parse_qsl, urlencode
        raw = (url or "").strip()
        if not raw:
            return ""
        if "://" not in raw:
            raw = "http://" + raw            # normaliza URLs sem protocolo
        u = urlparse(raw)
        vid = _yt_video_id(raw)
        if vid:
            return f"yt:{vid}"               # qualquer forma do mesmo vídeo → mesma chave
        host = (u.netloc or "").lower().split("@")[-1].split(":")[0]
        for pre in ("www.", "m.", "mobile.", "pt.", "en."):
            if host.startswith(pre):
                host = host[len(pre):]
        path = (u.path or "/").rstrip("/") or "/"
        qs = sorted((k, v) for k, v in parse_qsl(u.query) if k.lower() not in _TRACK_PARAMS)
        q = urlencode(qs)
        return f"{host}{path}" + (f"?{q}" if q else "")
    except Exception:
        return (url or "").strip().lower()

def _og_text(html: str) -> str:
    """og:title + og:description + meta description — o que QUALQUER página séria
    expõe pra previews, mesmo quando o corpo é 100% JavaScript."""
    import html as _h
    out = []
    for pat in (r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)',
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:title["\']',
                r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)',
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:description["\']',
                r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)'):
        m = _re.search(pat, html or "", _re.I)
        if m:
            t = _re.sub(r"\s+", " ", _h.unescape(m.group(1))).strip()
            if t and t not in out:
                out.append(t)
    return "\n".join(out)

def _tiktok_page_text(html: str) -> str:
    """Extrai do JSON EMBUTIDO na página do TikTok (nickname/bio/descrições) — o
    oEmbed público parou de responder (400), mas a página ainda carrega tudo embutido.
    Em página de vídeo, o 1º 'desc' é a legenda; em perfil, vem nome + bio + stats."""
    out = []
    def grab(pat, cap):
        found = []
        for m in _re.finditer(pat, html or ""):
            try:
                v = _json.loads('"' + m.group(1) + '"')
            except Exception:
                v = m.group(1)
            v = v.strip()
            if v and v not in found:
                found.append(v)
            if len(found) >= cap:
                break
        return found
    nick  = grab(r'"nickname":"((?:[^"\\]|\\.){1,80})"', 1)
    sig   = grab(r'"signature":"((?:[^"\\]|\\.){1,300})"', 1)
    descs = grab(r'"desc":"((?:[^"\\]|\\.){5,300})"', 4)
    if nick:
        out.append(f"Criador(a) no TikTok: {nick[0]}")
    if sig:
        out.append(f"Bio: {sig[0]}")
    out.extend(descs)
    return "\n".join(out)

async def _social_text(url: str) -> tuple:
    """Extração POR PLATAFORMA via canais públicos (oEmbed/JSON/OG). ('', '') se não der."""
    low = (url or "").lower()
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as c:
            # TikTok — oEmbed (se voltar a funcionar) e, na prática, o JSON embutido da página
            if "tiktok.com" in low:
                try:
                    r = await c.get("https://www.tiktok.com/oembed", params={"url": url})
                    if r.status_code == 200:
                        d = r.json()
                        cap, author = (d.get("title") or "").strip(), (d.get("author_name") or "").strip()
                        if cap or author:
                            kind = "Vídeo" if "/video/" in low or "/photo/" in low else "Perfil"
                            return (f"{kind} de {author} no TikTok. {('Legenda: ' + cap) if cap else ''}".strip(), "post")
                except Exception:
                    pass
                r = await c.get(url, headers={"User-Agent": _BROWSER_UA, "Accept-Language": "pt-BR,pt,en"})
                txt = _tiktok_page_text(r.text[:600000])
                if txt:
                    return (txt, "post")
                og = _og_text(r.text[:300000])
                if og and "make your day" not in og.lower():
                    return (f"Perfil no TikTok. {og}", "post")
            # X / Twitter — oEmbed público devolve o TEXTO do tweet
            if "twitter.com" in low or "://x.com" in low:
                r = await c.get("https://publish.twitter.com/oembed",
                                params={"url": url, "omit_script": "1", "lang": "pt"})
                if r.status_code == 200:
                    d = r.json()
                    txt = _re.sub(r"\s+", " ", _re.sub(r"<[^>]+>", " ", d.get("html") or "")).strip()
                    if txt:
                        return (f"Post de {d.get('author_name', '')} no X: {txt}", "post")
            # Reddit — .json público: título + texto + top comentários
            if "reddit.com" in low and "/comments/" in low:
                r = await c.get(url.split("?")[0].rstrip("/") + ".json",
                                headers={"User-Agent": _BROWSER_UA})
                if r.status_code == 200:
                    j = r.json()
                    post = j[0]["data"]["children"][0]["data"] if isinstance(j, list) and j else {}
                    parts = [post.get("title", ""), post.get("selftext", "")]
                    try:
                        for ch in j[1]["data"]["children"][:3]:
                            b = ch.get("data", {}).get("body", "")
                            if b:
                                parts.append("Comentário: " + b)
                    except Exception:
                        pass
                    txt = "\n".join(p for p in parts if p).strip()
                    if txt:
                        return (txt, "post")
            # Vimeo — oEmbed com título + autor + descrição
            if "vimeo.com" in low:
                r = await c.get("https://vimeo.com/api/oembed.json", params={"url": url})
                if r.status_code == 200:
                    d = r.json()
                    txt = "\n".join(x for x in (d.get("title", ""), d.get("author_name", ""), d.get("description", "")) if x)
                    if txt:
                        return (txt, "vídeo")
            # Instagram / Facebook / LinkedIn — fechados: OG com UA de crawler de preview
            if any(s in low for s in ("instagram.com", "facebook.com", "fb.watch", "linkedin.com")):
                for ua in (_CRAWLER_UA, _BROWSER_UA):
                    try:
                        r = await c.get(url, headers={"User-Agent": ua, "Accept-Language": "pt-BR,pt,en"})
                        og = _og_text(r.text[:300000])
                        if len(og) > 30:
                            return (og, "post")
                    except Exception:
                        continue
    except Exception:
        pass
    return ("", "")

async def _extract_content(url: str, platform: str, video_id: str):
    """Texto REAL do conteúdo (o multiplicador): YouTube→transcrição; redes sociais→
    canais públicos por plataforma; Google Docs→export; página→readability com UA de
    navegador + fallback OG. Retorna (texto, tipo, nº_palavras)."""
    url = _unwrap_url(url)
    text, ctype = "", "web"
    low = (url or "").lower()
    try:
        if platform == "youtube" or video_id:
            ctype = "vídeo"
            text = (await _yt_transcript(video_id)) or (await _yt_description(video_id)) or ""
        elif url:
            social = any(s in low for s in ("tiktok.", "twitter.", "://x.com", "instagram.",
                                            "facebook.", "fb.watch", "linkedin.", "reddit.", "vimeo."))
            if social:
                text, st = await _social_text(url)
                if st:
                    ctype = st
            if not text and "docs.google.com/document" in low:
                m = _re.search(r"/document/d/([\w-]+)", url)
                if m:
                    async with httpx.AsyncClient(timeout=12, follow_redirects=True) as c:
                        r = await c.get(f"https://docs.google.com/document/d/{m.group(1)}/export?format=txt")
                        if r.status_code == 200 and r.text.strip():
                            text, ctype = r.text, "artigo"
            if not text:
                async with httpx.AsyncClient(timeout=12, follow_redirects=True) as c:
                    r = await c.get(url, headers={"User-Agent": _BROWSER_UA, "Accept-Language": "pt-BR,pt,en"})
                    html = r.text[:500000]
                    text = _html_to_text(html)
                    if len(text) < 200:   # corpo raso (página JS/paywall) → OG salva algo
                        og = _og_text(html)
                        if len(og) > len(text):
                            text = og
                    if len(text) < 60:    # última carta: UA de crawler (paywalls servem OG)
                        r2 = await c.get(url, headers={"User-Agent": _CRAWLER_UA})
                        alt = _og_text(r2.text[:300000]) or _html_to_text(r2.text[:300000])
                        if len(alt) > len(text):
                            text = alt
                if ctype == "web":
                    if any(s in low for s in ("medium.", "substack", "/blog", "/article", "news", "dev.to")):
                        ctype = "artigo"
                    elif social:
                        ctype = "post"
    except Exception:
        pass
    text = (text or "").strip()[:8000]
    return text, ctype, len(text.split())

async def _summarize(title: str, text: str) -> str:
    """Resumo objetivo a partir do CONTEÚDO real (artigo/transcrição) — 1 parágrafo + bullets."""
    prompt = (
        "Resuma este conteúdo em português, de forma útil e objetiva. Formato:\n"
        "1) Um parágrafo curto (2-3 frases) explicando do que se trata e a ideia central.\n"
        "2) 3 a 5 pontos-chave em bullets (cada um começando com \"- \").\n"
        "Sem saudações, sem enrolação, sem inventar nada além do conteúdo.\n\n"
        f"TÍTULO: {title}\n\nCONTEÚDO:\n{text[:6000]}"
    )
    return await _chat(prompt, 600, temperature=0.3)

async def _link_meta_line(doc: dict, user_id: str) -> str:
    """Linha com o que o USUÁRIO sabe sobre o link: título + categoria + tags +
    plataforma. É o que torna respondível 'o vídeo do TikTok na categoria W' —
    mesmo quando a página não entrega texto nenhum (TikTok/Instagram)."""
    cat = ""
    if doc.get("categoryId"):
        try:
            c = await db.categories.find_one({"_id": ObjectId(doc["categoryId"]), "userId": user_id}, {"name": 1})
            cat = (c or {}).get("name", "")
        except Exception:
            pass
    parts = [doc.get("title") or doc.get("url", "")]
    if cat:
        parts.append(f"Categoria: {cat}")
    if doc.get("tags"):
        parts.append("Tags: " + ", ".join(doc["tags"][:6]))
    if doc.get("platform") and doc["platform"] != "other":
        parts.append(f"Plataforma: {doc['platform']}")
    return " · ".join(p for p in parts if p)

async def _embed_note(note_id: str, user_id: str):
    """Nota → embedding (segundo cérebro inclui o que VOCÊ escreveu, não só o que salvou).
    Grava None quando o texto é curto demais — marca como processada sem re-tentar."""
    if not _ai_enabled():
        return
    try:
        doc = await db.notes.find_one({"_id": ObjectId(note_id), "userId": user_id},
                                      {"title": 1, "body": 1})
        if not doc:
            return
        text = ((doc.get("title") or "") + "\n" + (doc.get("body") or "")).strip()
        vec = await _ai_embedding(text) if len(text) >= 20 else None
        await db.notes.update_one({"_id": doc["_id"], "userId": user_id},
                                  {"$set": {"noteEmbedding": vec}})
    except Exception:
        pass

async def _enrich_link(link_id: str, user_id: str):
    """Job em background: extrai o CONTEÚDO REAL (1x), gera aiTopics + embedding SOBRE o
    conteúdo (não só o título), resumo, tipo e tempo de leitura. Tudo melhora com isto."""
    if not _ai_enabled():
        return
    try:
        doc = await db.links.find_one({"_id": ObjectId(link_id), "userId": user_id})
    except Exception:
        doc = None
    if not doc:
        return
    title = doc.get("title", "") or ""
    content = doc.get("contentText")
    ctype = doc.get("contentType")
    words = doc.get("contentWords")
    if not content:   # nunca extraiu OU ficou vazio (re-tenta: extratores melhoram)
        content, ctype, words = await _extract_content(doc.get("url", ""), doc.get("platform", ""), doc.get("videoId", ""))
    snippet = (content or "")[:1500]
    topics = await _ai_tags(title + (("\n" + snippet) if snippet else ""))
    emb = await _ai_embedding((title + "\n" + (content or "")).strip())
    updates = {
        "aiEnrichedAt": datetime.utcnow(),
        "contentText": content or "",
        "contentType": ctype or "web",
        "contentWords": words or 0,
        "readingTimeMin": max(1, round((words or 0) / 200)),
    }
    if not doc.get("urlKey"):   # dedup canônico p/ links antigos (sem custo de IA)
        updates["urlKey"] = _url_key(doc.get("url", ""))
    if topics:
        updates["aiTopics"] = list(dict.fromkeys(topics))   # temas de IA, separados das tags do usuário
    if emb:
        updates["topicEmbedding"] = emb
    # Resumo automático SOBRE o conteúdo real (não só a descrição). Gerado 1x p/ TODOS.
    if not doc.get("aiSummary"):
        base = (content or "").strip()
        if len(base) < 120:   # pouco/nenhum texto extraído → usa descrição do vídeo
            base = (base + "\n" + await _yt_description(doc.get("videoId", ""))).strip() if doc.get("videoId") else base
        if len(base) >= 60:
            summary = await _summarize(title, base)
            if summary:
                updates["aiSummary"] = summary

    # CHUNKS com timestamp (1x por link, versão 2): vídeo → segmentos do timedtext
    # (carregam o MINUTO → o RAG cita "aos 12:30"); artigo → chunks por sentença.
    # SEMPRE inclui um chunk-meta (idx 0) com título+categoria+tags+plataforma —
    # garante que TODO link exista no Q&A, até TikTok/Instagram sem texto extraível.
    if doc.get("chunksVer") != 3:
        vid = doc.get("videoId", "")
        if vid:
            segs = await _yt_transcript_segments(vid)
            chunks = _chunk_segments(segs) if segs else _chunk_text(content or "")
        else:
            chunks = _chunk_text(content or "")
        meta = await _link_meta_line(doc, user_id)
        if meta:
            chunks = [{"t": None, "text": meta}] + chunks
        if chunks:
            vecs = await _ai_embeddings_batch([c["text"] for c in chunks])
            rows = [{"userId": user_id, "linkId": link_id, "idx": i,
                     "t": c["t"], "text": c["text"], "embedding": v}
                    for i, (c, v) in enumerate(zip(chunks, vecs)) if v]
            if rows:
                try:
                    await db.chunks.delete_many({"linkId": link_id, "userId": user_id})
                    await db.chunks.insert_many(rows)
                    updates["chunksAt"] = datetime.utcnow()
                    updates["chunksVer"] = 3
                except Exception:
                    pass
        else:
            updates["chunksAt"] = datetime.utcnow()
            updates["chunksVer"] = 3   # sem nada p/ indexar → não fica re-tentando
    try:
        await db.links.update_one({"_id": doc["_id"], "userId": user_id},
                                  {"$set": updates, "$inc": {"aiTries": 1}})
    except Exception:
        pass
    # Save AVULSO sem categoria → categoriza agora (1 chamada). Itens de PLAYLIST
    # (importBatch) NÃO entram aqui: vão pelo lote do backfill (mais barato).
    has_cat = updates.get("categoryId") or doc.get("categoryId")
    if not has_cat and not doc.get("importBatch") and _ai_enabled():
        try:
            await _auto_categorize_pending(user_id)
        except Exception:
            pass

@app.get("/api/ai/status")
@limiter.limit("10/minute")
async def ai_status(request: Request):
    """Diagnóstico: confirma se a IA responde (faz 1 chamada mínima de tags+embedding)."""
    prov = "gemini" if GEMINI_KEYS else ("groq" if GROQ_KEYS else ("openai" if OPENAI_API_KEY else "none"))
    if prov == "none":
        return {"provider": "none", "configured": False}
    emb = await _ai_embedding("teste de embedding")
    probe = {"embed_ok": bool(emb), "embed_dims": (len(emb) if emb else 0)}
    tags = await _ai_tags("Tutorial de Marketing Digital e Facebook Ads para iniciantes")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    # Uma chave conta como "disponível" se ainda tem cota em ALGUM modelo de chat hoje.
    usable = [k for k in GEMINI_KEYS
              if any(_gemini_exhausted.get((k, m)) != today for m in GEMINI_CHAT_MODELS)]
    dead = [k for k in GEMINI_KEYS
            if all(_gemini_exhausted.get((k, m)) == today for m in GEMINI_CHAT_MODELS)]
    keys_info = {
        "total": len(GEMINI_KEYS),
        "available_today": len(usable),
        "models": GEMINI_CHAT_MODELS,
        "exhausted_today": [k[:6] + "…" for k in dead],
    } if GEMINI_KEYS else None
    groq_info = {"keys": len(GROQ_KEYS), "models": GROQ_MODELS} if GROQ_KEYS else None
    return {"provider": prov, "configured": True,
            "keys": keys_info,
            "groq": groq_info,
            "chat_model": (GEMINI_CHAT_MODELS[0] if GEMINI_KEYS else (GROQ_MODELS[0] if GROQ_KEYS else OPENAI_CHAT_MODEL)),
            "embed_model": (GEMINI_EMBED_MODEL if GEMINI_KEYS else OPENAI_EMBED_MODEL),
            "tags_ok": bool(tags), "tags_sample": tags, "probe": probe}

@app.post("/api/ai/summary/{link_id}")
async def ai_summary(link_id: str, user=Depends(get_current_user)):
    """Resumo + pontos-chave do vídeo (a partir de título + descrição). Sob demanda."""
    if not _ai_enabled():
        raise HTTPException(status_code=400, detail="IA não configurada")
    link = await db.links.find_one({"_id": ObjectId(link_id), "userId": str(user["_id"])})
    if not link:
        raise HTTPException(status_code=404, detail="Link não encontrado")
    if link.get("aiSummary"):
        return {"summary": link["aiSummary"]}
    # Prefere o CONTEÚDO real (Fase 1); cai pra descrição do vídeo se não houver texto.
    base = (link.get("contentText") or "").strip()
    if len(base) < 120 and link.get("videoId"):
        base = (base + "\n" + await _yt_description(link.get("videoId", ""))).strip()
    summary = await _summarize(link.get("title", ""), base) if len(base) >= 60 else ""
    if summary:
        await db.links.update_one({"_id": link["_id"]}, {"$set": {"aiSummary": summary}})
    return {"summary": summary}

@app.post("/api/ai/bridge")
async def ai_bridge(body: BridgeReq, user=Depends(get_current_user)):
    """Sugere uma PONTE entre dois assuntos pouco conectados na biblioteca."""
    if not _ai_enabled():
        raise HTTPException(status_code=400, detail="IA não configurada")
    a = (body.a or "").strip()[:80]
    b = (body.b or "").strip()[:80]
    if not a or not b:
        raise HTTPException(status_code=400, detail="Temas inválidos")
    prompt = (
        f"Um usuário tem muitos conteúdos (vídeos) sobre '{a}' e sobre '{b}', "
        "mas quase nada que conecte os dois assuntos. Em português, de forma curta e prática:\n"
        "1) Sugira um TEMA-PONTE que liga os dois (1 linha começando com \"Ponte: \").\n"
        "2) Sugira 1 ou 2 ideias concretas de conteúdo/vídeo para assistir ou criar que "
        "preencham essa lacuna (bullets começando com \"- \").\n"
        "Sem saudações, sem enrolação. Máximo 4 linhas."
    )
    text = await _ai_text(prompt, 300)
    # tema-ponte (linha "Ponte: ...") vira a busca de um VÍDEO concreto p/ preencher.
    query = ""
    m = _re.search(r"Ponte:\s*(.+)", text or "")
    if m:
        query = m.group(1).strip().rstrip(".").strip()[:90]
    if not query:
        query = f"{a} {b}"
    videos = await _yt_search(query, 1)
    return {"suggestion": text, "query": query, "video": (videos[0] if videos else None)}

async def _yt_search(query: str, n: int = 1) -> list:
    """Busca vídeos reais no YouTube (Data API) p/ a 'ponte' das lacunas. Vazio sem chave."""
    if not YOUTUBE_API_KEY or not (query or "").strip():
        return []
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get("https://www.googleapis.com/youtube/v3/search", params={
                "part": "snippet", "q": query, "type": "video", "maxResults": n,
                "relevanceLanguage": "pt", "key": YOUTUBE_API_KEY})
            if r.status_code == 200:
                out = []
                for it in r.json().get("items", []):
                    vid = (it.get("id") or {}).get("videoId")
                    sn = it.get("snippet") or {}
                    if vid:
                        out.append({
                            "videoId": vid,
                            "title": sn.get("title", ""),
                            "channel": sn.get("channelTitle", ""),
                            "thumb": ((sn.get("thumbnails") or {}).get("medium") or {}).get("url", ""),
                            "url": f"https://www.youtube.com/watch?v={vid}",
                        })
                return out
    except Exception:
        pass
    return []

def _cosine(a, b) -> float:
    if not a or not b:   # None/vazio (doc sem embedding) → sem similaridade, sem crash
        return 0.0
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return s / (na * nb) if na and nb else 0.0

# "Guia de organização": regras que o usuário ENSINOU à IA. Persistem e são injetadas
# nos prompts de categoria/tags (a IA não treina; isto é a "memória" prática).
async def _get_org_rules(uid: str) -> list:
    doc = await db.ai_prefs.find_one({"userId": uid})
    return (doc.get("rules") or []) if doc else []

def _rules_block(rules: list) -> str:
    if not rules:
        return ""
    return ("REGRAS DO USUÁRIO (conselhos dele sobre como organizar — siga quando fizer sentido):\n"
            + "\n".join(f"- {r}" for r in rules[:20]) + "\n\n")

@app.get("/api/ai/rules")
async def get_rules(user=Depends(get_current_user)):
    return {"rules": await _get_org_rules(str(user["_id"]))}

@app.put("/api/ai/rules")
async def put_rules(req: RulesReq, user=Depends(get_current_user)):
    rules = [str(r).strip()[:160] for r in (req.rules or []) if str(r).strip()][:25]
    await db.ai_prefs.update_one({"userId": str(user["_id"])},
                                 {"$set": {"rules": rules, "updatedAt": datetime.utcnow()}}, upsert=True)
    return {"rules": rules}

@app.post("/api/ai/teach")
async def ai_teach(req: TeachReq, user=Depends(get_current_user)):
    """Recebe um CONSELHO do usuário, destila numa REGRA curta e guarda no guia dele."""
    uid = str(user["_id"])
    advice = (req.advice or "").strip()[:400]
    if not advice:
        raise HTTPException(status_code=400, detail="Conselho vazio")
    rule = advice
    if _ai_enabled():
        prompt = ("Transforme o conselho do usuário sobre como organizar a biblioteca dele em UMA "
                  "regra curta e imperativa (1 linha, em português, sem aspas, sem numerar). Mantenha "
                  "o sentido exato.\nConselho: " + advice)
        r = await _chat(prompt, 60, models=GEMINI_SMART_MODELS)
        if r:
            rule = r.strip().strip('"').splitlines()[0][:160]
    rules = await _get_org_rules(uid)
    if rule and rule.lower() not in [x.lower() for x in rules]:
        rules = (rules + [rule])[-25:]
    await db.ai_prefs.update_one({"userId": uid},
                                 {"$set": {"rules": rules, "updatedAt": datetime.utcnow()}}, upsert=True)
    return {"rule": rule, "rules": rules}

@app.post("/api/links/search")
async def semantic_search(req: SearchReq, user=Depends(get_current_user)):
    """Busca SEMÂNTICA: embeda a query e ranqueia os links do usuário por similaridade
    de cosseno com o topicEmbedding. Sem IA/sem vetor → cai p/ busca textual simples."""
    q = (req.q or "").strip()
    if not q:
        return {"semantic": False, "results": []}
    uid = str(user["_id"])
    # Projeção: só o que a busca usa. SEM contentText (8KB/doc) — com biblioteca
    # grande, puxar o doc inteiro custava dezenas de MB por busca digitada.
    docs = await db.links.find({"userId": uid},
        {"title": 1, "url": 1, "rawThumb": 1, "videoId": 1, "platform": 1,
         "tags": 1, "aiTopics": 1, "topicEmbedding": 1}).to_list(8000)
    by_id = {str(d["_id"]): d for d in docs}
    def _fmt(did, score):
        d = by_id[did]
        thumb = d.get("rawThumb") or (f"https://img.youtube.com/vi/{d.get('videoId')}/hqdefault.jpg" if d.get("videoId") else "")
        return {"id": did, "score": round(score, 3), "title": d.get("title", ""), "url": d.get("url", ""),
                "thumb": thumb, "videoId": d.get("videoId", ""), "platform": d.get("platform", "other")}
    emb = await _ai_embedding(q)
    if not emb:
        ql = q.lower()
        hits = [str(d["_id"]) for d in docs
                if ql in (d.get("title", "") + " " + " ".join(d.get("tags", []) or []) + " " + " ".join(d.get("aiTopics", []) or [])).lower()]
        return {"semantic": False, "results": [_fmt(i, 1.0) for i in hits[:req.limit]]}
    sims = []
    for d in docs:
        e = d.get("topicEmbedding")
        if e:
            sims.append((str(d["_id"]), _cosine(emb, e)))
    sims.sort(key=lambda x: -x[1])
    return {"semantic": True, "results": [_fmt(i, s) for i, s in sims[:req.limit] if s > 0.22]}

@app.post("/api/ai/suggest-category")
async def suggest_category(req: SuggestCatReq, user=Depends(get_current_user)):
    """Sugere, para um conteúdo novo, a MELHOR categoria EXISTENTE ou propõe UMA nova
    (com pai opcional). O usuário decide usar a sugestão ou fazer manual."""
    uid = str(user["_id"])
    title = (req.title or "").strip()[:240]
    if not _ai_enabled() or not title:
        return {"matchId": None, "newName": None, "newParentId": None, "reason": ""}
    cats = await db.categories.find({"userId": uid}).to_list(2000)
    by_id = {str(c["_id"]): c for c in cats}
    def path(c):
        parts = [c.get("name", "")]; p = c.get("parentId")
        seen = 0
        while p and p in by_id and seen < 12:
            parts.insert(0, by_id[p].get("name", "")); p = by_id[p].get("parentId"); seen += 1
        return " › ".join(parts)
    counts = {}
    async for d in db.links.find({"userId": uid}, {"categoryId": 1}):
        cid = d.get("categoryId")
        if cid:
            counts[cid] = counts.get(cid, 0) + 1
    rows = [f'- [{cid}] {path(c)} ({counts.get(cid, 0)} itens)' for cid, c in by_id.items()]
    listing = "\n".join(rows[:150]) or "(nenhuma ainda)"
    desc = (req.description or "")[:600]
    prompt = (
        "Você é um bibliotecário organizando uma ÁRVORE de categorias (profundidade ilimitada).\n\n"
        "PRIMEIRO entenda o que o conteúdo REALMENTE é: qual o seu ASSUNTO e a sua NATUREZA/uso "
        "(é uma ferramenta? um app? um tutorial? um artigo? um produto? sobre o quê?). NÃO classifique "
        "por palavras soltas do título. Exemplo de erro a evitar: uma FERRAMENTA de design que usa IA "
        "NÃO é um 'Agente de IA' — é uma ferramenta de design; case pelo SIGNIFICADO e USO real, não "
        "pela mera presença da palavra 'IA' no nome.\n\n"
        "Uma boa categoria AGRUPA vários conteúdos do mesmo assunto — nunca é feita para um único item. "
        "Decida o lugar CERTO, nesta ordem de preferência:\n"
        "1) REUTILIZAR: se já existe categoria/subcategoria adequada (em qualquer nível) que combine "
        "com o ASSUNTO/USO real, use-a (matchId). Na dúvida entre reutilizar e criar, REUTILIZE.\n"
        "2) Só se nada servir, CRIE UMA, escolhendo o nível pela ABRANGÊNCIA do assunto:\n"
        "   • PRINCIPAL (newParentId null) → tema amplo e novo, não coberto por nenhuma existente.\n"
        "   • SUB / SUB-SUB (newParentId = nó existente) → faceta MAIS específica de uma existente.\n\n"
        "NÃO faça: categoria 'prima' quase igual a uma existente; sub/sub-sub granular que terá só "
        "este item; aninhar fundo sem necessidade; nem deixar tudo na raiz. Use a contagem de itens "
        "como dica de quais categorias são agrupamentos reais. No 'reason', explique em 1 frase o que "
        "o conteúdo é e por que esse lugar.\n\n"
        "Responda APENAS um JSON em uma linha:\n"
        '{"matchId": "<id existente ou null>", "newName": "<nome curto da nova ou null>", '
        '"newParentId": "<id do pai (pode ser nó profundo) ou null>", "reason": "<o que é + por quê>"}\n\n'
        + _rules_block(await _get_org_rules(uid)) +
        f"ÁRVORE (id › caminho completo (nº de itens)):\n{listing}\n\n"
        f"CONTEÚDO:\nTítulo: {title}\nURL: {(req.url or '')[:200]}\n"
        + (f"Descrição: {desc}\n" if desc else "")
    )
    txt = await _chat(prompt, 260, temperature=0.2, models=GEMINI_SMART_MODELS)
    out = {"matchId": None, "newName": None, "newParentId": None, "reason": ""}
    try:
        m = _re.search(r"\{.*\}", txt, _re.S)
        if m:
            data = _json.loads(m.group(0))
            mid = data.get("matchId")
            out["matchId"] = mid if (isinstance(mid, str) and mid in by_id) else None
            nn = data.get("newName")
            out["newName"] = (str(nn).strip()[:40] or None) if (nn and not out["matchId"]) else None
            npid = data.get("newParentId")
            out["newParentId"] = npid if (isinstance(npid, str) and npid in by_id and out["newName"]) else None
            out["reason"] = str(data.get("reason") or "")[:160]
    except Exception:
        pass
    # nome legível do match (p/ o front exibir sem recalcular)
    out["matchPath"] = path(by_id[out["matchId"]]) if out["matchId"] else None
    return out

@app.post("/api/ai/suggest-tags")
async def suggest_tags(req: SuggestTagsReq, user=Depends(get_current_user)):
    """Sugere TAGS de usuário (estilo curadoria) para o conteúdo, REUTILIZANDO o
    vocabulário de tags que o usuário já usa (consistência) + novas se preciso.
    O usuário aceita por clique (não entra automático)."""
    uid = str(user["_id"])
    title = (req.title or "").strip()[:240]
    if not _ai_enabled() or not title:
        return {"tags": []}
    vocab = set()
    async for l in db.links.find({"userId": uid}, {"tags": 1}).limit(3000):
        for t in (l.get("tags") or []):
            if t: vocab.add(str(t))
    vlist = list(vocab)[:140]
    desc = (req.description or "")[:500]
    prompt = (
        "Entenda o que o conteúdo REALMENTE é (assunto + natureza/uso) e sugira de 3 a 6 TAGS curtas "
        "(1–2 palavras, minúsculas) no estilo de curadoria pessoal (assunto/uso/nicho). Não derive tag "
        "de palavra solta do título; reflita o conteúdo real. PREFIRA reutilizar tags que o usuário JÁ "
        "usa quando couberem; complemente com novas só se necessário. Sem # e sem repetir.\n"
        "Responda APENAS um array JSON de strings. Ex.: [\"marketing\",\"tráfego pago\",\"iniciante\"]\n\n"
        + _rules_block(await _get_org_rules(uid)) +
        f"TAGS JÁ USADAS PELO USUÁRIO: {', '.join(vlist) if vlist else '(nenhuma ainda)'}\n"
        f"Título: {title}\nURL: {(req.url or '')[:200]}\n"
        + (f"Descrição: {desc}\n" if desc else "")
    )
    txt = await _chat(prompt, 140, temperature=0.3, models=GEMINI_SMART_MODELS)
    tags = []
    try:
        m = _re.search(r"\[.*\]", txt, _re.S)
        if m:
            arr = _json.loads(m.group(0))
            seen = set()
            for t in arr:
                s = str(t).strip().lower()[:28]
                if s and s not in seen:
                    seen.add(s); tags.append(s)
    except Exception:
        pass
    return {"tags": tags[:6]}

@app.get("/api/ai/library-map")
async def library_map(user=Depends(get_current_user)):
    """Mapa do conhecimento: no que o usuário é FORTE (categorias por contagem
    recursiva), onde há BURACOS (categorias quase vazias), os TEMAS dominantes, e um
    RESUMO da IA do perfil da biblioteca."""
    uid = str(user["_id"])
    cats = await db.categories.find({"userId": uid}).to_list(3000)
    links = await db.links.find({"userId": uid}, {"categoryId": 1, "aiTopics": 1}).to_list(8000)
    by_id = {str(c["_id"]): c for c in cats}
    direct = {}
    for l in links:
        cid = l.get("categoryId")
        if cid:
            direct[cid] = direct.get(cid, 0) + 1
    children = {}
    for cid, c in by_id.items():
        children.setdefault(c.get("parentId"), []).append(cid)
    def rec_count(cid, seen=None):
        seen = seen or set()
        if cid in seen:
            return 0
        seen.add(cid)
        return direct.get(cid, 0) + sum(rec_count(ch, seen) for ch in children.get(cid, []))
    roots = [cid for cid, c in by_id.items() if not c.get("parentId")]
    cat_stats = sorted(((by_id[cid].get("name", ""), rec_count(cid)) for cid in roots), key=lambda x: -x[1])
    total = len(links)
    strengths = [{"name": n, "count": c} for n, c in cat_stats if c >= 3][:8]
    gaps = [{"name": n, "count": c} for n, c in cat_stats if c <= 2][:8]
    orphan = sum(1 for l in links if not l.get("categoryId") or l.get("categoryId") not in by_id)
    topic_freq = {}
    for l in links:
        for t in (l.get("aiTopics") or []):
            topic_freq[t] = topic_freq.get(t, 0) + 1
    top_topics = sorted(topic_freq.items(), key=lambda x: -x[1])[:12]
    topics = [{"name": t, "count": c} for t, c in top_topics]

    # #12 Mapa hierárquico: temas que aparecem JUNTOS no mesmo link se conectam.
    # Constrói 'Tema ↳ relacionados' a partir da co-ocorrência dos aiTopics.
    cooc = {}
    for l in links:
        ts = list(dict.fromkeys(l.get("aiTopics") or []))
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]
                da = cooc.setdefault(a, {}); da[b] = da.get(b, 0) + 1
                db = cooc.setdefault(b, {}); db[a] = db.get(a, 0) + 1
    min_cooc = 2 if total >= 25 else 1
    tree = []
    for root, _c in sorted(topic_freq.items(), key=lambda x: -x[1])[:6]:
        kids = sorted((cooc.get(root) or {}).items(), key=lambda x: -x[1])
        ch = [{"name": b, "count": cnt} for b, cnt in kids if cnt >= min_cooc][:5]
        if ch:   # só raízes que de fato conectam a algo
            tree.append({"name": root, "count": topic_freq[root], "children": ch})

    summary = ""
    if _ai_enabled() and total >= 3:
        cats_str = ", ".join(f"{n} ({c})" for n, c in cat_stats[:15])
        topics_str = ", ".join(f"{t} ({c})" for t, c in top_topics[:10]) or "(sem temas ainda)"
        prompt = (
            "Em 2 a 3 frases, descreva o PERFIL desta biblioteca de conteúdos de um usuário: sobre o "
            "que ela é, no que ele está mais forte e (se houver) o que está pouco explorado. Tom "
            "direto e pessoal, em português, sem jargão técnico. Não invente além dos dados.\n"
            f"Total: {total} conteúdos.\nCategorias (nome (itens)): {cats_str}\nTemas: {topics_str}"
        )
        try:   # prazo máximo: IA lenta/fora NÃO pode segurar o mapa — abre sem o resumo
            summary = await asyncio.wait_for(_chat(prompt, 240, models=GEMINI_SMART_MODELS), 25)
        except Exception:
            summary = ""
    return {"total": total, "summary": summary, "strengths": strengths, "gaps": gaps, "topics": topics, "orphan": orphan, "tree": tree}

_insights_cache = {}  # uid -> (ts, n_items, result)

@app.get("/api/ai/insights")
async def ai_insights(user=Depends(get_current_user)):
    """Fase 3 — 'O que eu percebi sobre você': padrões em linguagem natural a partir
    do que VOCÊ salvou. Os fatos são calculados de forma determinística (temas, recência,
    plataforma, profundidade, backlog); a IA só os transforma em frases curtas e pessoais.
    Cacheado ~10min por usuário (invalida quando o nº de itens muda)."""
    uid = str(user["_id"])
    now = datetime.utcnow()
    links = await db.links.find(
        {"userId": uid},
        {"aiTopics": 1, "platform": 1, "url": 1, "contentWords": 1, "watched": 1, "createdAt": 1},
    ).to_list(8000)
    total = len(links)
    if total < 5:
        return {"insights": [], "total": total, "need": 5}

    cached = _insights_cache.get(uid)
    if cached and cached[1] == total and (now.timestamp() - cached[0]) < 600:
        return cached[2]

    d30 = now - timedelta(days=30)
    d7  = now - timedelta(days=7)
    def _dt(l):
        v = l.get("createdAt")
        return v if isinstance(v, datetime) else None
    week_count  = sum(1 for l in links if (_dt(l) or now) >= d7)
    month_count = sum(1 for l in links if (_dt(l) or now) >= d30)

    # Temas: frequência total e recente (últimos 30d) → dominante, contraste e "em alta"
    topic_freq, topic_recent = {}, {}
    for l in links:
        recent = (_dt(l) or datetime(2000, 1, 1)) >= d30
        for t in (l.get("aiTopics") or []):
            topic_freq[t] = topic_freq.get(t, 0) + 1
            if recent:
                topic_recent[t] = topic_recent.get(t, 0) + 1
    top_topics = sorted(topic_freq.items(), key=lambda x: -x[1])
    dominant = top_topics[0] if top_topics else None
    # contraste: tema relevante porém com pouca presença (o "47 vs 2")
    small = next(((t, c) for t, c in reversed(top_topics) if 1 <= c <= 2 and total >= 12), None)
    # em alta: tema cuja maioria das adições é recente (e tem massa mínima)
    rising = None
    for t, c in top_topics:
        r = topic_recent.get(t, 0)
        if c >= 3 and r >= 2 and r / c >= 0.5:
            rising = (t, r, c); break

    # Linha do tempo intelectual: agrupa por período (granularidade conforme o alcance
    # dos dados) e acha o tema dominante de cada um — usa a DATA DE SALVAMENTO (quando
    # VOCÊ se interessou), que reflete melhor a evolução dos seus interesses.
    _MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    dated = [(_dt(l), l.get("aiTopics") or []) for l in links if _dt(l)]
    timeline = []
    if len(dated) >= 6:
        dts = [d for d, _ in dated]
        span = (max(dts) - min(dts)).days
        if span > 540:      # > ~18 meses → por ano
            keyf, labf = (lambda d: f"{d.year:04d}"), (lambda d: f"{d.year}")
        elif span > 120:    # > ~4 meses → por trimestre
            keyf, labf = (lambda d: f"{d.year:04d}-{(d.month-1)//3:01d}"), (lambda d: f"T{(d.month-1)//3+1} {d.year}")
        else:               # curto → por mês
            keyf, labf = (lambda d: f"{d.year:04d}-{d.month:02d}"), (lambda d: f"{_MESES[d.month-1]}/{d.year}")
        buckets = {}
        for d, topics in dated:
            k = keyf(d)
            b = buckets.setdefault(k, {"label": labf(d), "n": 0, "topics": {}})
            b["n"] += 1
            for t in topics:
                b["topics"][t] = b["topics"].get(t, 0) + 1
        for k in sorted(buckets):
            b = buckets[k]
            top = sorted(b["topics"].items(), key=lambda x: -x[1])[:2]
            timeline.append({"period": k, "label": b["label"], "count": b["n"],
                             "top": [{"name": t, "count": c} for t, c in top]})

    # Plataforma dominante
    plat_freq = {}
    for l in links:
        p = (l.get("platform") or "").strip()
        if not p or p == "other":
            try:
                p = _re.sub(r"^www\.", "", (l.get("url") or "").split("/")[2]).split(".")[0]
            except Exception:
                p = "outros"
        plat_freq[p] = plat_freq.get(p, 0) + 1
    top_plat = max(plat_freq.items(), key=lambda x: x[1]) if plat_freq else None

    deep    = sum(1 for l in links if (l.get("contentWords") or 0) > 200)
    watched = sum(1 for l in links if l.get("watched"))
    backlog = total - watched

    # Monta os FATOS (números reais) para a IA narrar — nada além disto pode ser dito.
    facts = [f"Total salvo: {total}.",
             f"Adicionados nos últimos 30 dias: {month_count}; nesta semana: {week_count}.",
             f"Já consumidos: {watched}; ainda na fila: {backlog}.",
             f"Conteúdos com leitura profunda (texto extraído): {deep}."]
    if dominant:
        facts.append(f"Tema dominante: '{dominant[0]}' com {dominant[1]} conteúdos.")
    if small:
        facts.append(f"Tema com pouca presença: '{small[0]}' com apenas {small[1]}.")
    if rising:
        facts.append(f"Tema em alta: '{rising[0]}' — {rising[1]} dos {rising[2]} são recentes (últimos 30 dias).")
    if top_plat:
        facts.append(f"Plataforma mais usada: {top_plat[0]} ({top_plat[1]} conteúdos).")
    other_topics = ", ".join(f"{t} ({c})" for t, c in top_topics[1:8])
    if other_topics:
        facts.append(f"Outros temas: {other_topics}.")
    # Evolução temporal: só se houver >=2 períodos com tema dominante
    evo = [p for p in timeline if p["top"]]
    if len(evo) >= 2:
        seq = " → ".join(f"{p['label']}: {p['top'][0]['name']}" for p in evo)
        facts.append(f"Evolução dos interesses ao longo do tempo (período: tema dominante): {seq}.")

    insights = []
    if _ai_enabled():
        prompt = (
            "Você analisa a biblioteca pessoal de conteúdos de alguém e aponta PADRÕES que a "
            "pessoa talvez não tenha notado — como um amigo observador e inteligente. "
            "Gere de 3 a 5 percepções CURTAS (1 frase cada), em português, na 2ª pessoa ('você'), "
            "tom pessoal e direto, SEM jargão técnico. Cada percepção deve usar números REAIS dos "
            "fatos abaixo e, quando fizer sentido, sugerir sutilmente um próximo passo. "
            "NÃO invente nada que não esteja nos fatos. Responda APENAS um array JSON: "
            '[{"icon":"<1 emoji>","text":"<frase>"}].\n\nFATOS:\n' + "\n".join(facts)
        )
        try:   # prazo máximo: sem IA, cai no fallback determinístico abaixo
            txt = await asyncio.wait_for(_chat(prompt, 420, temperature=0.4, models=GEMINI_SMART_MODELS), 25)
        except Exception:
            txt = ""
        try:
            m = _re.search(r"\[.*\]", txt, _re.S)
            if m:
                for it in _json.loads(m.group(0)):
                    t = str(it.get("text", "")).strip()
                    if t:
                        insights.append({"icon": str(it.get("icon", "✨"))[:4], "text": t[:240]})
        except Exception:
            pass

    # Fallback determinístico (IA indisponível ou parse falhou)
    if not insights:
        if dominant:
            insights.append({"icon": "🎯", "text": f"Seu maior foco é '{dominant[0]}': {dominant[1]} conteúdos salvos."})
        if rising:
            insights.append({"icon": "📈", "text": f"'{rising[0]}' está em alta — {rising[1]} dos {rising[2]} são das últimas semanas."})
        if backlog > 0:
            insights.append({"icon": "📚", "text": f"Você tem {backlog} conteúdos esperando — que tal escolher 1 pra hoje?"})
        if small and dominant:
            insights.append({"icon": "🔍", "text": f"Muito '{dominant[0]}', mas só {small[1]} sobre '{small[0]}' — um ponto a explorar."})

    result = {"insights": insights[:5], "total": total, "timeline": timeline,
              "stats": {"week": week_count, "month": month_count, "watched": watched, "backlog": backlog, "deep": deep}}
    _insights_cache[uid] = (now.timestamp(), total, result)
    return result

@app.get("/api/ai/related/{link_id}")
async def ai_related(link_id: str, user=Depends(get_current_user)):
    """#14 Recomendação contextual: dado um link, devolve os mais SIMILARES do próprio
    acervo por similaridade de cosseno do topicEmbedding. 'Vendo X → veja também Y, Z'."""
    uid = str(user["_id"])
    try:
        base = await db.links.find_one({"_id": ObjectId(link_id), "userId": uid}, {"topicEmbedding": 1})
    except Exception:
        base = None
    if not base:
        raise HTTPException(status_code=404, detail="Link não encontrado")
    emb = base.get("topicEmbedding")
    if not emb:
        return {"results": []}   # ainda não enriquecido — sem vetor, sem similaridade
    docs = await db.links.find({"userId": uid, "topicEmbedding": {"$exists": True}},
        {"topicEmbedding": 1, "title": 1, "url": 1, "rawThumb": 1,
         "videoId": 1, "platform": 1, "watched": 1}).to_list(8000)
    scored = []
    for d in docs:
        if str(d["_id"]) == link_id:
            continue
        v = d.get("topicEmbedding")
        if not v:
            continue
        scored.append((_cosine(emb, v), d))
    scored.sort(key=lambda x: -x[0])
    results = []
    for s, d in scored[:6]:
        if s < 0.45:   # corta relações fracas (evita "relacionado" sem relação)
            break
        thumb = d.get("rawThumb") or (f"https://img.youtube.com/vi/{d.get('videoId')}/hqdefault.jpg" if d.get("videoId") else "")
        results.append({"id": str(d["_id"]), "score": round(s, 3), "title": d.get("title", ""),
                        "url": d.get("url", ""), "thumb": thumb, "videoId": d.get("videoId", ""),
                        "platform": d.get("platform", "other"), "watched": bool(d.get("watched"))})
    return {"results": results}

# ─── REVISÃO ATIVA (repetição espaçada) ──────────────────────────────────────
# Transforma a biblioteca de arquivo passivo em APRENDIZADO: conteúdo consumido
# vira card de revisão ("lembra qual era a tese?"). Intervalos: lembrei→x2.5,
# vago→x1.3, esqueci→volta p/ 1 dia. 1ª revisão 3 dias após consumir.

@app.get("/api/review/queue")
async def review_queue(user=Depends(get_current_user)):
    uid = str(user["_id"])
    now = datetime.utcnow()
    links = await db.links.find(
        {"userId": uid, "watched": True, "aiSummary": {"$exists": True, "$ne": ""}},
        {"title": 1, "rawThumb": 1, "videoId": 1, "url": 1, "aiSummary": 1, "watchedAt": 1}).to_list(4000)
    if not links:
        return {"cards": [], "total": 0}
    by_id = {str(l["_id"]): l for l in links}
    revs = {r["linkId"]: r for r in await db.reviews.find({"userId": uid}).to_list(8000)}
    due = []
    for lid, l in by_id.items():
        r = revs.get(lid)
        if r is None:
            wa = l.get("watchedAt")
            first = (wa if isinstance(wa, datetime) else now) + timedelta(days=3)
            if first <= now:
                due.append((first, lid, None))
        elif isinstance(r.get("due"), datetime) and r["due"] <= now:
            due.append((r["due"], lid, r))
    due.sort(key=lambda x: x[0])
    batch = due[:5]
    if not batch:
        return {"cards": [], "total": 0}

    # Pergunta de recall: gerada UMA vez por card (1 chamada p/ o lote), salva no review.
    need_q = [(lid, by_id[lid]) for _, lid, r in batch if not (r or {}).get("question")]
    questions = {}
    if need_q and _ai_enabled():
        lst = "\n\n".join(f"{i+1}. Título: {l.get('title','')}\nResumo: {(l.get('aiSummary') or '')[:400]}"
                          for i, (_, l) in enumerate(need_q))
        try:   # prazo máximo: sem IA, cai na pergunta-fallback por título
            txt = await asyncio.wait_for(_chat(
                "Para cada conteúdo abaixo, gere UMA pergunta curta de revisão ativa, em português, que "
                "teste se a pessoa LEMBRA da ideia central — sem entregar a resposta na pergunta. "
                'Responda APENAS um array JSON de strings, na mesma ordem: ["pergunta 1", ...]\n\n' + lst,
                400, temperature=0.4), 25)
        except Exception:
            txt = ""
        try:
            m = _re.search(r"\[.*\]", txt, _re.S)
            for (lid, _), q_ in zip(need_q, _json.loads(m.group(0)) if m else []):
                questions[lid] = str(q_).strip()[:300]
        except Exception:
            pass

    cards = []
    for due_at, lid, r in batch:
        l = by_id[lid]
        q_ = (r or {}).get("question") or questions.get(lid) \
            or f"O que você lembra de \"{(l.get('title') or '')[:80]}\"?"
        await db.reviews.update_one({"userId": uid, "linkId": lid},
            {"$setOnInsert": {"interval": 3, "due": due_at, "createdAt": now},
             "$set": {"question": q_}}, upsert=True)
        thumb = l.get("rawThumb") or (f"https://img.youtube.com/vi/{l.get('videoId')}/hqdefault.jpg" if l.get("videoId") else "")
        cards.append({"id": lid, "title": l.get("title", ""), "url": l.get("url", ""),
                      "thumb": thumb, "question": q_, "answer": l.get("aiSummary", "")})
    return {"cards": cards, "total": len(due)}

class ReviewAnswer(BaseModel):
    linkId: str = ""
    result: str = "lembrei"   # lembrei | vago | esqueci

@app.post("/api/review/answer")
async def review_answer(req: ReviewAnswer, user=Depends(get_current_user)):
    uid = str(user["_id"])
    r = await db.reviews.find_one({"userId": uid, "linkId": req.linkId}) or {}
    interval = float(r.get("interval") or 3)
    if req.result == "esqueci":
        interval = 1.0
    elif req.result == "vago":
        interval = max(2.0, interval * 1.3)
    else:
        interval = min(180.0, interval * 2.5)   # teto de ~6 meses
    await db.reviews.update_one({"userId": uid, "linkId": req.linkId},
        {"$set": {"interval": interval, "due": datetime.utcnow() + timedelta(days=interval),
                  "lastResult": req.result, "answeredAt": datetime.utcnow()}}, upsert=True)
    return {"ok": True, "nextDays": round(interval)}

_topics_cache = {}  # uid -> (ts, n_items, result) — evita reclusterizar/renomear a cada abertura

@app.get("/api/ai/topics")
async def ai_topics(user=Depends(get_current_user)):
    """Temas por SIGNIFICADO: clusteriza os vídeos pelos embeddings (greedy por centroide)
    e a IA NOMEIA cada grupo (1 chamada só). Retorna no formato dos tópicos do Universo
    ({name,count,videoIds}). Cacheado ~10min por usuário."""
    uid = str(user["_id"])
    links = await db.links.find({"userId": uid}, {"topicEmbedding": 1, "title": 1, "aiTopics": 1}).to_list(8000)
    items = [l for l in links if l.get("topicEmbedding")]
    now = datetime.utcnow().timestamp()
    cache = _topics_cache.get(uid)
    if cache and now - cache[0] < 600 and cache[1] == len(items):
        return cache[2]
    if len(items) < 4:
        res = {"topics": []}; _topics_cache[uid] = (now, len(items), res); return res
    THRESH = 0.55
    clusters = []
    for l in items:
        e = l["topicEmbedding"]
        best, bs = None, -1.0
        for c in clusters:
            sim = _cosine(e, c["cen"])
            if sim > bs:
                bs, best = sim, c
        if best and bs >= THRESH:
            for i in range(len(e)):
                best["sum"][i] += e[i]
            best["n"] += 1
            best["cen"] = [s / best["n"] for s in best["sum"]]
            best["ids"].append(str(l["_id"])); best["titles"].append(l.get("title", "") or "")
            for t in (l.get("aiTopics") or []):
                best["topics"][t] = best["topics"].get(t, 0) + 1
        else:
            clusters.append({"sum": list(e), "cen": list(e), "n": 1, "ids": [str(l["_id"])],
                             "titles": [l.get("title", "") or ""], "topics": {t: 1 for t in (l.get("aiTopics") or [])}})
    clusters = sorted([c for c in clusters if c["n"] >= 2], key=lambda c: -c["n"])[:14]
    names = []
    if clusters and _ai_enabled():
        lines = []
        for i, c in enumerate(clusters):
            tt = ", ".join(t for t, _ in sorted(c["topics"].items(), key=lambda x: -x[1])[:5])
            ex = "; ".join(s[:50] for s in c["titles"][:3])
            lines.append(f"{i}: temas=[{tt}] exemplos=[{ex}]")
        prompt = ("Dê um NOME curto (1 a 3 palavras, em português) para cada GRUPO de conteúdos "
                  "abaixo, refletindo o tema em comum. Responda APENAS um array JSON de strings, na "
                  "MESMA ORDEM e quantidade dos grupos.\n" + "\n".join(lines))
        txt = await _chat(prompt, 220, models=GEMINI_SMART_MODELS)
        try:
            m = _re.search(r"\[.*\]", txt, _re.S)
            if m:
                names = [str(x).strip()[:30] for x in _json.loads(m.group(0))]
        except Exception:
            pass
    topics = []
    for i, c in enumerate(clusters):
        nm = (names[i] if i < len(names) and names[i]
              else (sorted(c["topics"].items(), key=lambda x: -x[1])[0][0] if c["topics"] else f"Grupo {i + 1}"))
        topics.append({"name": nm, "count": c["n"], "videoIds": c["ids"]})
    res = {"topics": topics}
    _topics_cache[uid] = (now, len(items), res)
    return res

@app.post("/api/ai/ask")
async def ai_ask(req: AskReq, user=Depends(get_current_user)):
    """Assistente do segundo cérebro. A IA recebe a ESTRUTURA REAL da biblioteca
    (categorias + itens + plataformas) E os TRECHOS relevantes do conteúdo — então
    responde tanto perguntas de ORGANIZAÇÃO ('quantas categorias tenho', 'qual item
    na categoria W') quanto de CONTEÚDO ('o que aprendi sobre X', com citações [n])."""
    uid = str(user["_id"])
    q = (req.q or "").strip()[:400]
    if not q:
        return {"answer": "", "sources": []}
    if not _ai_enabled():
        return {"answer": "IA não configurada.", "sources": []}
    ql = q.lower()
    uname = (user.get("name") or "").split(" ")[0]

    # ── Carrega a biblioteca inteira (leve) + categorias ──────────────────────
    cats_all = await db.categories.find({"userId": uid}, {"name": 1, "parentId": 1}).to_list(3000)
    cname = {str(c["_id"]): (c.get("name") or "") for c in cats_all}
    links = await db.links.find({"userId": uid},
        {"title": 1, "url": 1, "rawThumb": 1, "videoId": 1, "platform": 1,
         "tags": 1, "categoryId": 1, "aiSummary": 1, "topicEmbedding": 1}).to_list(8000)
    total = len(links)
    if total == 0:
        return {"answer": "Sua biblioteca ainda está vazia — salve alguns conteúdos e eu passo a responder sobre eles.", "sources": []}

    def _src(d, t=None):
        thumb = d.get("rawThumb") or (f"https://img.youtube.com/vi/{d.get('videoId')}/hqdefault.jpg" if d.get("videoId") else "")
        return {"kind": "link", "id": str(d["_id"]), "title": d.get("title", ""), "url": d.get("url", ""),
                "thumb": thumb, "videoId": d.get("videoId", ""), "t": t}

    # ── ESTRUTURA: categorias (com contagem) + plataformas ────────────────────
    by_cat, plat_count = {}, {}
    for l in links:
        by_cat.setdefault(l.get("categoryId") or "__none__", []).append(l)
        p = (l.get("platform") or "outros")
        plat_count[p] = plat_count.get(p, 0) + 1
    cat_counts = sorted(((cname.get(cid, "Sem categoria") if cid != "__none__" else "Sem categoria",
                          cid, len(items)) for cid, items in by_cat.items()), key=lambda x: -x[2])

    # categorias citadas na pergunta (match por nome, com fronteira de palavra)
    mentioned = []
    for nm, cid, n in cat_counts:
        low = (nm or "").strip().lower()
        if low and low != "sem categoria" and _re.search(rf"(?<!\w){_re.escape(low)}(?!\w)", ql):
            mentioned.append(cid)

    struct = [f"Total: {total} conteúdos · {len([c for c in cat_counts if c[1] != '__none__'])} categorias.",
              "PLATAFORMAS: " + ", ".join(f"{p} ({n})" for p, n in sorted(plat_count.items(), key=lambda x: -x[1])),
              "CATEGORIAS (nome · qtd):"]
    shown_items = 0
    for nm, cid, n in cat_counts:
        struct.append(f"- {nm} ({n})")
        # lista os títulos dos itens p/ categorias citadas ou pequenas (até um teto global)
        if (cid in mentioned or n <= 4) and shown_items < 80:
            for it in by_cat[cid][:25]:
                struct.append(f"    • {(it.get('title') or it.get('url',''))[:90]}"
                              + (f"  [{it.get('platform')}]" if it.get("platform") and it["platform"] != "other" else ""))
                shown_items += 1
    struct_text = "\n".join(struct)[:6000]

    # ── Montagem das FONTES (numeradas e DEDUPADAS por link) ──────────────────
    blocks, sources, seen = [], [], set()
    def _add_link(d, body="", t=None):
        did = str(d["_id"])
        if did in seen:
            return
        seen.add(did)
        i = len(blocks) + 1
        head = d.get("title") or d.get("url", "")
        cn = cname.get(d.get("categoryId") or "", "")
        meta = head + (f" · Categoria: {cn}" if cn else "") + (f" · {d.get('platform')}" if d.get("platform") and d["platform"] != "other" else "")
        stamp = f" (aos {int(t)//60}:{int(t)%60:02d})" if isinstance(t, (int, float)) else ""
        blocks.append(f"[{i}] {meta}{stamp}" + (f"\n{body[:800]}" if body else ""))
        sources.append(_src(d, int(t) if isinstance(t, (int, float)) else None))

    # 1) PRIORIDADE: itens das categorias CITADAS na pergunta (é o que ele pediu)
    for cid in mentioned:
        for it in by_cat.get(cid, [])[:8]:
            _add_link(it, (it.get("aiSummary") or "")[:300])

    # 2) CONTEÚDO relevante por embedding (chunks com timestamp) — 1 fonte por link
    emb = await _ai_embedding(q)
    if emb:
        cvecs = await db.chunks.find({"userId": uid}, {"embedding": 1, "linkId": 1}).to_list(20000)
        csims = sorted(((_cosine(emb, c.get("embedding")), c) for c in cvecs), key=lambda x: -x[0])
        ldocs = {str(d["_id"]): d for d in links}
        added = 0
        for s, c in csims:
            if s < 0.22 or added >= 8:
                break
            if c["linkId"] in seen:
                continue
            ld = ldocs.get(c["linkId"])
            if not ld:
                continue
            ch = await db.chunks.find_one({"_id": c["_id"]}, {"text": 1, "t": 1})
            _add_link(ld, (ch or {}).get("text", ""), (ch or {}).get("t"))
            added += 1
        # notas do próprio usuário
        nvecs = await db.notes.find({"userId": uid, "noteEmbedding": {"$exists": True}, "deletedAt": None},
                                    {"noteEmbedding": 1}).to_list(4000)
        nsims = sorted(((_cosine(emb, n.get("noteEmbedding")), n["_id"]) for n in nvecs), key=lambda x: -x[0])
        nids = [nid for s, nid in nsims[:2] if s > 0.3]
        for n in (await db.notes.find({"_id": {"$in": nids}}, {"title": 1, "body": 1}).to_list(len(nids)) if nids else []):
            i = len(blocks) + 1
            blocks.append(f"[{i}] SUA NOTA: {n.get('title') or 'sem título'}\n{(n.get('body') or '')[:600]}".strip())
            sources.append({"kind": "note", "id": str(n["_id"]), "title": n.get("title") or "Sua nota",
                            "url": "", "thumb": "", "videoId": "", "t": None})

    content_text = ("\n\n".join(blocks))[:7000] if blocks else "(nenhum trecho de conteúdo recuperado)"

    # Memória do chat: turnos anteriores (a IA junta a pergunta nova com o contexto)
    hist = ""
    for turn in (req.history or [])[-4:]:
        uq = str(turn.get("q", ""))[:300]
        ua = str(turn.get("a", ""))[:600]
        if uq:
            hist += f"\nUsuário: {uq}\nVocê: {ua}"

    prompt = (
        f"Você é o assistente pessoal do segundo cérebro de {uname or 'do usuário'} — inteligente, "
        "direto e perspicaz, como o ChatGPT/Claude, mas que CONHECE a biblioteca dele. Responda em "
        "português, de forma natural e útil (sem ser robótico).\n\n"
        "COMO USAR O CONTEXTO:\n"
        "• Esta é uma CONVERSA: se houver histórico abaixo, leve-o em conta — uma mensagem curta nova "
        "geralmente COMPLEMENTA a pergunta anterior (ex.: depois de 'vídeos da categoria W' o usuário "
        "digita 'a asiática' → ele quer refinar a MESMA busca, não começar outra).\n"
        "• Perguntas de ORGANIZAÇÃO (quantas/quais categorias, o que está na categoria X, plataforma, "
        "quantos itens) → responda pela ESTRUTURA DA BIBLIOTECA (verdade completa; conte e liste com "
        "confiança).\n"
        "• Perguntas de CONTEÚDO (o que aprendi sobre X, resuma) → use os TRECHOS e CITE com [n]. "
        "Trecho com tempo ('aos 12:30') → mencione.\n"
        "• LOCALIZAR algo → cruze estrutura + trechos, aponte o item mais provável citando [n], aceite "
        "correspondência parcial. Só diga que não achou se realmente não houver candidato.\n"
        "• NÃO invente conteúdo fora do acervo, mas seja proativo: sugira o item/categoria mais próximo.\n\n"
        f"=== ESTRUTURA DA BIBLIOTECA ===\n{struct_text}\n\n"
        f"=== TRECHOS DE CONTEÚDO (fontes citáveis) ===\n{content_text}\n"
        + (f"\n=== CONVERSA ATÉ AGORA ==={hist}\n" if hist else "")
        + f"\nPERGUNTA ATUAL: {q}"
    )
    try:
        answer = await asyncio.wait_for(_chat(prompt, 900, temperature=0.35, models=GEMINI_SMART_MODELS), 30)
    except Exception:
        answer = ""
    return {"answer": answer or "Não consegui gerar a resposta agora — tente de novo.", "sources": sources}

@app.post("/api/ai/gemini")
async def ai_gemini_proxy(req: GeminiReq, user=Depends(get_current_user)):
    """Proxy do Gemini com ROTAÇÃO de chaves — usado pela IA do Financeiro (e
    qualquer chamada que precise do generateContent cru). Devolve {status, data}
    espelhando a resposta do Gemini para o front aplicar a lógica dele (fallback
    de modelo etc.). Em 429 rotaciona as chaves; só devolve 429 quando TODAS
    esgotaram no dia."""
    if not GEMINI_KEYS:
        raise HTTPException(status_code=400, detail="IA não configurada (sem chaves Gemini)")
    model = (req.model or GEMINI_CHAT_MODELS[0]).strip()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    last = {"status": 502, "data": {"error": {"message": "sem chaves disponíveis"}}}
    async with httpx.AsyncClient(timeout=60) as c:
        for key in _gemini_keys_available(model):
            try:
                r = await c.post(url, params={"key": key}, json=req.body)
            except Exception as e:
                last = {"status": 599, "data": {"error": {"message": str(e)[:200]}}}
                continue
            if r.status_code == 200:
                return {"status": 200, "data": r.json()}
            if r.status_code == 429:
                try: edata = r.json()
                except Exception: edata = {}
                if _is_daily_quota_429(edata):     # só marca esgotada se for cota DIÁRIA
                    _gemini_exhausted[(key, model)] = today
                last = {"status": 429, "data": edata}
                continue
            # erro não-cota (sobrecarga 503, 400 etc.) → devolve p/ o front decidir
            try: data = r.json()
            except Exception: data = {"error": {"message": r.text[:300]}}
            return {"status": r.status_code, "data": data}
    return last

class GroqReq(BaseModel):
    prompt: str = ""
    max_tokens: int = 2048
    temperature: float = 0.4
    json_mode: bool = True

@app.post("/api/ai/groq")
async def ai_groq_proxy(req: GroqReq, user=Depends(get_current_user)):
    """Proxy do Groq (compatível com OpenAI) com rotação de chaves + fallback de modelos.
    Usado como REDE DE SEGURANÇA quando o Gemini esgota a cota (ex.: Apresentação do
    Financeiro). Com json_mode liga o response_format JSON do Groq. Devolve {status, text}."""
    if not GROQ_KEYS:
        return {"status": 400, "text": "", "error": "sem chaves Groq"}
    today = datetime.utcnow().strftime("%Y-%m-%d")
    payload = {"max_tokens": max(256, min(req.max_tokens, 8000)), "temperature": req.temperature,
               "messages": [{"role": "user", "content": req.prompt}]}
    if req.json_mode:
        payload["response_format"] = {"type": "json_object"}
    last = {"status": 502, "text": "", "error": "sem chaves disponíveis"}
    async with httpx.AsyncClient(timeout=60) as c:
        for model in GROQ_MODELS:
            for key in _groq_keys_available(model):
                try:
                    r = await c.post("https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}"}, json={**payload, "model": model})
                except Exception as e:
                    last = {"status": 599, "text": "", "error": str(e)[:200]}
                    continue
                if r.status_code == 200:
                    try:
                        return {"status": 200, "text": (r.json()["choices"][0]["message"]["content"] or "").strip()}
                    except Exception:
                        return {"status": 200, "text": ""}
                if r.status_code == 429:
                    msg = ""
                    try: msg = str((r.json().get("error") or {}).get("message", "")).lower()
                    except Exception: pass
                    if any(s in msg for s in ("per day", "daily", "rpd", "tpd")):
                        _groq_exhausted[(key, model)] = today
                    last = {"status": 429, "text": "", "error": "rate limit"}
                    continue
                try: msg = str((r.json().get("error") or {}).get("message", ""))[:200]
                except Exception: msg = r.text[:200]
                last = {"status": r.status_code, "text": "", "error": msg}
    return last

async def _cat_paths(uid: str):
    """(by_id, paths) onde paths[cid] = 'Pai > Filho > Neto' e contagem de itens por cat."""
    cats = await db.categories.find({"userId": uid}).to_list(3000)
    by_id = {str(c["_id"]): c for c in cats}
    counts = {}
    async for l in db.links.find({"userId": uid}, {"categoryId": 1}):
        cid = l.get("categoryId")
        if cid:
            counts[cid] = counts.get(cid, 0) + 1
    def path_of(cid, seen=None):
        seen = seen or set()
        c = by_id.get(cid)
        if not c or cid in seen:
            return ""
        seen.add(cid)
        p = c.get("parentId")
        pre = path_of(str(p), seen) if p else ""
        return (pre + " > " if pre else "") + (c.get("name") or "")
    return by_id, {cid: path_of(cid) for cid in by_id}, counts

async def _resolve_path(uid: str, names: list) -> str:
    """Acha/cria a cadeia de categorias (aninhada) e devolve o id da FOLHA. Reusa por
    nome+pai (case-insensitive) — não duplica categoria que já existe."""
    parent = None
    leaf = None
    for raw in names[:4]:   # profundidade máx 4 (sub-sub-sub)
        name = (raw or "").strip()[:60]
        if not name:
            continue
        existing = await db.categories.find_one(
            {"userId": uid, "parentId": parent,
             "name": {"$regex": f"^{_re.escape(name)}$", "$options": "i"}})
        if existing:
            leaf = str(existing["_id"])
        else:
            order = await db.categories.count_documents({"userId": uid, "parentId": parent})
            res = await db.categories.insert_one(
                {"userId": uid, "name": name, "parentId": parent, "order": order,
                 "createdAt": datetime.utcnow(), "autoCreated": True})
            leaf = str(res.inserted_id)
        parent = leaf
    return leaf or ""

async def _auto_categorize_pending(uid: str, limit: int = 12) -> int:
    """B (automático, alta qualidade): pega itens SEM categoria e, EM LOTE, decide o
    melhor caminho de categoria pra cada um — reusando os existentes ou criando novos
    (aninhados) quando faz sentido. Lote = 1 chamada de IA (barato e mais coerente que
    decidir item a item). Itens ambíguos ficam sem categoria (não geram lixo)."""
    if not _ai_enabled():
        return 0
    # Só categoriza itens JÁ enriquecidos (têm resumo/temas) → qualidade alta E a barra
    # de "Categorias" anda junto com a leitura, em vez de ficar parada até o fim.
    items = await db.links.find(
        {"userId": uid, "autoCatAt": {"$exists": False}, "aiEnrichedAt": {"$exists": True},
         "$or": [{"categoryId": None}, {"categoryId": {"$exists": False}}, {"categoryId": ""}]},
        {"title": 1, "platform": 1, "aiSummary": 1, "aiTopics": 1, "url": 1}).limit(limit).to_list(limit)
    if not items:
        return 0
    _by, paths, counts = await _cat_paths(uid)
    tree = "\n".join(f"- {p}  ({counts.get(cid, 0)} itens)" for cid, p in sorted(paths.items(), key=lambda x: x[1]) if p) or "(você ainda não tem categorias)"
    lst = "\n".join(
        f'{i}. "{(it.get("title") or it.get("url",""))[:120]}"'
        + (f' — resumo: {(it.get("aiSummary") or "")[:160]}' if it.get("aiSummary") else "")
        + (f' — temas: {", ".join(it.get("aiTopics") or [])}' if it.get("aiTopics") else "")
        + (f' [{it.get("platform")}]' if it.get("platform") and it["platform"] != "other" else "")
        for i, it in enumerate(items))
    prompt = (
        "Você é um BIBLIOTECÁRIO especialista organizando uma biblioteca pessoal. Sua tarefa: dar a "
        "CADA item abaixo o melhor CAMINHO de categoria. A QUALIDADE é crítica — categoria errada vira "
        "lixo e estraga a biblioteca, então pense no ASSUNTO e no USO REAL de cada item (não em "
        "palavras soltas do título).\n\n"
        "REGRAS:\n"
        "1) REUTILIZE um caminho que já existe sempre que ele servir (use exatamente o mesmo nome). Na "
        "dúvida entre reusar e criar, REUTILIZE.\n"
        "2) Só CRIE um caminho novo quando nenhum existente couber. Pode aninhar até 4 níveis "
        "(ex.: ['Marketing','Tráfego Pago','Facebook Ads']). AGRUPE itens semelhantes no MESMO caminho "
        "novo — não crie uma categoria para um único item se vários combinam.\n"
        "3) Nomes curtos, claros, Capitalizados, em português. Não crie variações quase iguais "
        "('IA' e 'Inteligência Artificial' devem ser UMA só).\n"
        "4) Se um item for ambíguo/genérico demais pra classificar bem, devolva path: [] (deixa sem "
        "categoria — melhor vazio que errado).\n"
        + _rules_block(await _get_org_rules(uid)) +
        f"\nCATEGORIAS EXISTENTES (caminho · contagem):\n{tree}\n\n"
        f"ITENS A CLASSIFICAR:\n{lst}\n\n"
        'Responda APENAS um array JSON, um objeto por item: [{"i":0,"path":["Pai","Filho"]}, ...]. '
        "Use exatamente os nomes dos caminhos existentes quando reutilizar."
    )
    try:
        txt = await asyncio.wait_for(_chat(prompt, 1200, temperature=0.2, models=GEMINI_SMART_MODELS), 40)
        m = _re.search(r"\[.*\]", txt, _re.S)
        arr = _json.loads(m.group(0)) if m else []
    except Exception:
        arr = []
    assigned = {}
    for a in arr:
        if isinstance(a, dict) and isinstance(a.get("i"), int) and isinstance(a.get("path"), list):
            assigned[a["i"]] = [str(x) for x in a["path"] if str(x).strip()]
    done = 0
    for i, it in enumerate(items):
        upd = {"autoCatAt": datetime.utcnow()}   # marca como tentado (não reprocessa sempre)
        path = assigned.get(i)
        if path:
            cid = await _resolve_path(uid, path)
            if cid:
                upd["categoryId"] = cid
                done += 1
        await db.links.update_one({"_id": it["_id"], "userId": uid}, {"$set": upd})
    return done

@app.post("/api/ai/backfill")
async def ai_backfill(background: BackgroundTasks, user=Depends(get_current_user)):
    """Garante que TODO conteúdo (novo e antigo) tenha IA: enriquece em lotes os links
    sem aiEnrichedAt OU sem topicEmbedding (busca semântica/recomendações dependem do
    vetor), desistindo após 3 tentativas. Devolve `remaining` p/ o front repetir até 0."""
    if not _ai_enabled():
        return {"ok": False, "reason": "no_key", "remaining": 0}
    uid = str(user["_id"])
    q = {"userId": uid, "$and": [
        {"$or": [{"aiEnrichedAt": {"$exists": False}}, {"topicEmbedding": {"$exists": False}}, {"contentText": {"$exists": False}}, {"aiSummary": {"$exists": False}}, {"chunksVer": {"$ne": 3}}]},
        {"$or": [{"aiTries": {"$exists": False}}, {"aiTries": {"$lt": 6}}]},
    ]}
    remaining = await db.links.count_documents(q)
    pending = await db.links.find(q).limit(40).to_list(40)   # lote por chamada (extração é mais pesada)
    for l in pending:
        background.add_task(_enrich_link, str(l["_id"]), uid)
    # Notas antigas sem embedding entram no mesmo ciclo (RAG inclui o que VOCÊ escreveu)
    nq = {"userId": uid, "deletedAt": None, "noteEmbedding": {"$exists": False}}
    n_remaining = await db.notes.count_documents(nq)
    for n in await db.notes.find(nq, {"_id": 1}).limit(20).to_list(20):
        background.add_task(_embed_note, str(n["_id"]), uid)
    # Auto-categorização (B): itens SEM categoria são organizados em lote pela IA.
    # Roda APÓS o enriquecimento (usa resumo/temas) → só conta como pendente quando
    # não há mais enriquecimento na fila, p/ categorizar com o máximo de contexto.
    cq = {"userId": uid, "autoCatAt": {"$exists": False},
          "$or": [{"categoryId": None}, {"categoryId": {"$exists": False}}, {"categoryId": ""}]}
    cat_remaining = await db.links.count_documents(cq)
    # Roda JUNTO com o enriquecimento (categoriza só os já lidos) → barra anda ao vivo.
    if cat_remaining:
        background.add_task(_auto_categorize_pending, uid)
    return {"ok": True, "queued": len(pending), "remaining": remaining + n_remaining + cat_remaining}

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
    url = _unwrap_url(body.get("url", ""))
    if not url:
        raise HTTPException(status_code=400, detail="URL obrigatória")
    low = url.lower()

    # TikTok oEmbed → título vira "autor: legenda" (em vez de "TikTok - Make Your Day")
    if "tiktok.com" in low:
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=True) as c:
                r = await c.get("https://www.tiktok.com/oembed", params={"url": url})
                if r.status_code == 200:
                    d = r.json()
                    author = (d.get("author_name") or "").strip()
                    cap = (d.get("title") or "").strip()
                    title = (f"{author}: {cap}" if author and cap else cap or author)[:160]
                    if title:
                        return {"title": title, "thumbnail": d.get("thumbnail_url", ""),
                                "description": cap[:500], "platform": "tiktok", "author": author}
        except Exception:
            pass

    # X/Twitter oEmbed → texto do post como título/descrição
    if "twitter.com" in low or "://x.com" in low:
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=True) as c:
                r = await c.get("https://publish.twitter.com/oembed",
                                params={"url": url, "omit_script": "1", "lang": "pt"})
                if r.status_code == 200:
                    d = r.json()
                    txt = _re.sub(r"\s+", " ", _re.sub(r"<[^>]+>", " ", d.get("html") or "")).strip()
                    author = (d.get("author_name") or "").strip()
                    if txt:
                        return {"title": (f"{author}: {txt}" if author else txt)[:160],
                                "thumbnail": "", "description": txt[:500],
                                "platform": "twitter", "author": author}
        except Exception:
            pass

    # Vimeo oEmbed
    if "vimeo.com" in low:
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=True) as c:
                r = await c.get("https://vimeo.com/api/oembed.json", params={"url": url})
                if r.status_code == 200:
                    d = r.json()
                    return {"title": d.get("title", ""), "thumbnail": d.get("thumbnail_url", ""),
                            "description": (d.get("description") or "")[:500],
                            "platform": "vimeo", "author": d.get("author_name", ""),
                            "durationSeconds": d.get("duration")}
        except Exception:
            pass

    # YouTube oEmbed
    if "youtube.com" in url or "youtu.be" in url:
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(f"https://www.youtube.com/oembed?url={url}&format=json")
                if r.status_code == 200:
                    d = r.json()
                    vid = _yt_video_id(url)
                    dur = await _yt_duration(vid)
                    ydesc = await _yt_description(vid)
                    return {
                        "title":       d.get("title", ""),
                        "thumbnail":   d.get("thumbnail_url", ""),
                        "description": (ydesc or "")[:500],
                        "platform":    "youtube",
                        "author":      d.get("author_name", ""),
                        "durationSeconds": dur,
                    }
        except:
            pass
    
    # Generic Open Graph fallback (UA de navegador real — bots declarados são barrados)
    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as c:
            r = await c.get(url, headers={"User-Agent": _BROWSER_UA, "Accept-Language": "pt-BR,pt,en"})
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
            # descrição (og:description ou <meta name=description>) → ajuda a IA a
            # entender o que o conteúdo REALMENTE é (categoria/tags melhores).
            import re as _re2
            desc = og("description") or ""
            if not desc:
                m = _re2.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)', html, _re2.I)
                desc = m.group(1).strip() if m else ""
            return {
                "title":       title,
                "thumbnail":   og("image"),
                "description": desc[:500],
                "platform":    "other"
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

# ─── IMPORTAÇÃO EM MASSA (onboarding: biblioteca instantânea) ─────────────────
def _parse_bookmarks_html(html: str) -> list:
    """Bookmarks Netscape (export do Chrome/Edge/Firefox) → [{url, title}]. Puro."""
    import html as _h
    out, seen = [], set()
    for m in _re.finditer(r'<A[^>]+HREF="([^"]+)"[^>]*>(.*?)</A>', html or "", _re.S | _re.I):
        url = m.group(1).strip()
        if not url.lower().startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        title = _re.sub(r"\s+", " ", _h.unescape(_re.sub(r"<[^>]+>", " ", m.group(2)))).strip()
        out.append({"url": url, "title": title or url})
    return out

def _parse_playlist_page(html: str) -> list:
    """HTML da página de playlist do YouTube → [{videoId, title}] (ordem, dedup). Puro.
    Formato NOVO (2025+): lockupViewModel → contentId VIDEO + accessibilityContext.label
    (a label = 'Título  duração'; removemos a duração). Fallback: playlistVideoRenderer."""
    html = html or ""
    out, seen = [], set()

    def _push(vid, title):
        if vid and vid not in seen:
            seen.add(vid)
            out.append({"videoId": vid, "title": (title or "").strip() or "Vídeo do YouTube"})

    # Novo: contentId (vídeo) + label de acessibilidade logo em seguida
    for m in _re.finditer(
        r'"contentId":"([A-Za-z0-9_-]{11})","contentType":"LOCKUP_CONTENT_TYPE_VIDEO".{0,400}?'
        r'"accessibilityContext":\{"label":"((?:[^"\\]|\\.)*)"', html, _re.S):
        try:
            label = _json.loads(f'"{m.group(2)}"')
        except Exception:
            label = m.group(2)
        # tira o sufixo de duração ("3 minutos e 55 segundos" / "3 minutes, 55 seconds")
        title = _re.sub(r"\s+\d+\s+(hora|minuto|segundo|hour|minute|second)s?\b.*$", "", label).strip()
        _push(m.group(1), title or label)
    if out:
        return out

    # Antigo: playlistVideoRenderer com title.runs
    for m in _re.finditer(r'"videoId":"([A-Za-z0-9_-]{11})".{0,2000}?"title":\{"runs":\[\{"text":"((?:[^"\\]|\\.)*)"', html):
        try:
            title = _json.loads(f'"{m.group(2)}"')
        except Exception:
            title = m.group(2)
        _push(m.group(1), title)
    return out

class BulkImportReq(BaseModel):
    kind: str = "bookmarks"           # bookmarks | playlist
    url: str = ""                     # playlist do YouTube (?list=)
    html: str = ""                    # conteúdo do arquivo de bookmarks
    categoryId: Optional[str] = None

@app.post("/api/import/bulk")
@limiter.limit("10/hour")
async def import_bulk(request: Request, req: BulkImportReq, user=Depends(get_current_user)):
    """Importa MUITOS links de uma vez (playlist do YouTube ou bookmarks do navegador).
    Dedup contra o acervo, respeita o limite do plano free. O enriquecimento de IA
    acontece pelo backfill normal (lotes em background) — nada trava aqui."""
    uid = str(user["_id"])
    if req.kind == "playlist":
        m = _re.search(r"[?&]list=([A-Za-z0-9_-]+)", req.url or "")
        if not m:
            raise HTTPException(status_code=400, detail="URL de playlist inválida (precisa conter ?list=)")
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
                r = await c.get(f"https://www.youtube.com/playlist?list={m.group(1)}",
                                headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "pt-BR,pt,en"})
            vids = _parse_playlist_page(r.text)[:200]
        except Exception:
            raise HTTPException(status_code=502, detail="Não consegui ler a playlist agora")
        items = [{"url": f"https://www.youtube.com/watch?v={v['videoId']}", "title": v["title"],
                  "videoId": v["videoId"]} for v in vids]
    else:
        items = [{"url": b["url"], "title": b["title"], "videoId": _yt_video_id(b["url"])}
                 for b in _parse_bookmarks_html(req.html or "")[:500]]
    if not items:
        return {"ok": True, "found": 0, "imported": 0, "skipped": 0}

    existing = await db.links.find({"userId": uid}, {"videoId": 1, "urlKey": 1, "url": 1}).to_list(8000)
    have_v = {d.get("videoId") for d in existing if d.get("videoId")}
    have_k = {d.get("urlKey") or _url_key(d.get("url", "")) for d in existing}
    limit = 300 if user.get("plan", "free") == "free" else None
    cat = req.categoryId if req.categoryId else None
    now = datetime.utcnow()
    import uuid as _uuid
    batch = _uuid.uuid4().hex[:12]   # identifica ESTE lote (p/ a barra de progresso)
    docs, skipped = [], 0
    for it in items:
        ukey = _url_key(it["url"])
        if (it["videoId"] and it["videoId"] in have_v) or ukey in have_k:
            skipped += 1
            continue
        if limit is not None and len(existing) + len(docs) >= limit:
            break
        docs.append({
            "userId": uid, "url": it["url"], "urlKey": ukey, "title": (it["title"] or it["url"])[:300],
            "thumbnail": "", "rawThumb": "", "importBatch": batch,
            "platform": "youtube" if it["videoId"] else "other",
            "videoId": it["videoId"] or "", "categoryId": cat,
            "watched": False, "notes": "", "tags": [], "order": 0,
            "createdAt": now, "watchedAt": None,
            "durationSeconds": None, "watchedSeconds": 0,
            "lastWatchedAt": None, "watchCount": 0, "isFavorite": False,
        })
        if it["videoId"]:
            have_v.add(it["videoId"])
        have_k.add(ukey)
    if docs:
        await db.links.insert_many(docs)
    return {"ok": True, "found": len(items), "imported": len(docs), "skipped": skipped,
            "batch": batch if docs else "", "preCategorized": bool(cat),
            "limited": limit is not None and (len(existing) + len(docs)) >= limit}

@app.get("/api/import/progress")
async def import_progress(batch: str = "", user=Depends(get_current_user)):
    """Progresso do enriquecimento de um LOTE de import (p/ a barra ao vivo):
    quantos já foram lidos (conteúdo), ganharam temas/tags e foram categorizados."""
    uid = str(user["_id"])
    if not batch:
        return {"total": 0, "done": True}
    base = {"userId": uid, "importBatch": batch}
    total = await db.links.count_documents(base)
    enriched = await db.links.count_documents({**base, "aiEnrichedAt": {"$exists": True}})
    tagged = await db.links.count_documents({**base, "aiTopics.0": {"$exists": True}})
    categorized = await db.links.count_documents({**base, "categoryId": {"$nin": [None, ""]}})
    cat_done = await db.links.count_documents({**base, "autoCatAt": {"$exists": True}})
    done = total > 0 and enriched >= total and cat_done >= total
    return {"total": total, "enriched": enriched, "tagged": tagged,
            "categorized": categorized, "catProcessed": cat_done, "done": done}

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
    await db.links.create_index([("userId", 1), ("urlKey", 1)])   # dedup canônico
    # Notes indexes
    await db.notes.create_index([("userId", 1), ("position", -1)])
    await db.notes.create_index([("userId", 1), ("updatedAt", -1)])
    await db.notes.create_index([("userId", 1), ("folderId", 1)])
    await db.notes.create_index([("userId", 1), ("deletedAt", 1)])
    await db.notes.create_index([("userId", 1), ("linkedItemId", 1)])
    await db.note_folders.create_index([("userId", 1), ("order", 1)])
    # Backups
    await db.backups.create_index([("userId", 1), ("type", 1), ("createdAt", -1)])
    # Chunks (RAG com timestamp)
    await db.chunks.create_index([("userId", 1), ("linkId", 1)])
    # Revisão ativa
    await db.reviews.create_index([("userId", 1), ("linkId", 1)], unique=True)

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
