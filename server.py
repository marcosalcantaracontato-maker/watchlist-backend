"""
WatchList — Backend API
FastAPI + MongoDB (Motor) + JWT + Google OAuth

Deploy: Railway, Render, ou qualquer VPS
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from motor.motor_asyncio import AsyncIOMotorClient
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime, timedelta
from bson import ObjectId
import os, httpx
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MONGODB_URL            = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
JWT_SECRET             = os.getenv("JWT_SECRET", "TROQUE_ISSO_POR_UM_SECRET_FORTE")
GOOGLE_CLIENT_ID       = os.getenv("GOOGLE_CLIENT_ID", "")
ALGORITHM              = "HS256"
TOKEN_EXPIRE_DAYS      = 30
FRONTEND_URL           = os.getenv("FRONTEND_URL", "*")   # ex: https://watchlist.vercel.app

# ─── APP ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="WatchList API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://localhost:5173"],
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

# ─── AUTH MIDDLEWARE ──────────────────────────────────────────────────────────
security = HTTPBearer()

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        user    = await db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=401, detail="Usuário não encontrado")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido ou expirado")

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

class LinkUpdate(BaseModel):
    title:      Optional[str] = None
    thumbnail:  Optional[str] = None
    rawThumb:   Optional[str] = None
    categoryId: Optional[str] = None
    watched:    Optional[bool] = None
    notes:      Optional[str] = None
    tags:       Optional[List[str]] = None
    order:      Optional[int] = None

class MigrateRequest(BaseModel):
    categories: List[dict]
    links:      List[dict]

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

@app.post("/api/auth/google")
async def login_with_google(body: GoogleLoginRequest):
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
    
    return {
        "token":  token,
        "user":   serialize(user),
        "is_new": is_new
    }

@app.get("/api/auth/me")
async def get_me(user=Depends(get_current_user)):
    return serialize(user)

@app.post("/api/auth/logout")
async def logout(user=Depends(get_current_user)):
    # JWT is stateless — client just discards the token
    # For extra security, you could maintain a token blacklist in Redis
    return {"ok": True}

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
async def nuke_all_categories(user=Depends(get_current_user)):
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
@app.get("/api/links")
async def get_links(user=Depends(get_current_user), watched: Optional[bool] = None):
    query: dict = {"userId": str(user["_id"])}
    if watched is not None:
        query["watched"] = watched
    cursor = db.links.find(query).sort("createdAt", -1)
    return [serialize(l) async for l in cursor]

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
        "watchedAt":  None
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

# ─── METADATA FETCH ──────────────────────────────────────────────────────────
@app.post("/api/fetch-metadata")
async def fetch_metadata(body: dict):
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
                    return {
                        "title":     d.get("title", ""),
                        "thumbnail": d.get("thumbnail_url", ""),
                        "platform":  "youtube",
                        "author":    d.get("author_name", "")
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
async def migrate_data(body: MigrateRequest, user=Depends(get_current_user)):
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
