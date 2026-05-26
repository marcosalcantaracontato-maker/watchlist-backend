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
    categoryId: str
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

# ─── METADATA FETCH ──────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
# NOTES ENDPOINTS
# ═══════════════════════════════════════════════════════════════

class NoteBody(BaseModel):
    title: str
    content: str = ""
    priority: int = 4          # 1=alta, 4=sem
    dueDate: Optional[str] = None
    linkedItemId: Optional[str] = None
    tags: list = []
    folderName: Optional[str] = None
    isCompleted: bool = False

@app.get("/api/notes")
async def get_notes(user=Depends(get_current_user)):
    uid = str(user["_id"])
    cursor = db.notes.find({"userId": uid, "deletedAt": None}).sort("updatedAt", -1)
    notes = await cursor.to_list(None)
    return [serialize(n) for n in notes]

@app.post("/api/notes")
async def create_note(body: NoteBody, user=Depends(get_current_user)):
    uid = str(user["_id"])
    now = datetime.utcnow()
    doc = {
        "userId": uid,
        "title": body.title or "Sem título",
        "content": body.content,
        "priority": body.priority,
        "dueDate": body.dueDate,
        "linkedItemId": body.linkedItemId,
        "tags": body.tags,
        "folderName": body.folderName,
        "isCompleted": body.isCompleted,
        "createdAt": now,
        "updatedAt": now,
        "deletedAt": None,
    }
    result = await db.notes.insert_one(doc)
    return {"noteId": str(result.inserted_id)}

@app.patch("/api/notes/{note_id}")
async def update_note(note_id: str, body: dict, user=Depends(get_current_user)):
    uid = str(user["_id"])
    body["updatedAt"] = datetime.utcnow()
    body.pop("_id", None); body.pop("id", None)
    await db.notes.update_one(
        {"_id": ObjectId(note_id), "userId": uid},
        {"$set": body}
    )
    return {"ok": True}

@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str, user=Depends(get_current_user)):
    uid = str(user["_id"])
    await db.notes.update_one(
        {"_id": ObjectId(note_id), "userId": uid},
        {"$set": {"deletedAt": datetime.utcnow()}}
    )
    return {"ok": True}

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

# ─── NOTES ─────────────────────────────────────────────────────────────────────

class NoteCreate(BaseModel):
    title: str
    content: str = ""
    folder_id: Optional[str] = None
    linked_item_id: Optional[str] = None
    priority: int = 4
    tags: List[str] = []
    due_date: Optional[str] = None

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    folder_id: Optional[str] = None
    linked_item_id: Optional[str] = None
    priority: Optional[int] = None
    tags: Optional[List[str]] = None
    due_date: Optional[str] = None
    is_archived: Optional[bool] = None

class NoteFolderCreate(BaseModel):
    name: str
    parent_id: Optional[str] = None

@app.get("/api/notes")
async def get_notes(folder: str = None, user=Depends(get_current_user)):
    uid = str(user["_id"])
    query = {"userId": uid, "deleted_at": None}
    if folder == "inbox":    query["folder_id"] = None
    elif folder == "today":
        from datetime import date
        query["due_date"] = date.today().isoformat()
    elif folder and folder not in ("all",):
        query["folder_id"] = folder
    notes = await db.notes.find(query).sort("updated_at", -1).to_list(None)
    return [serialize(n) for n in notes]

@app.post("/api/notes")
async def create_note(body: NoteCreate, user=Depends(get_current_user)):
    uid = str(user["_id"])
    doc = {
        "userId": uid, "title": body.title, "content": body.content,
        "folder_id": body.folder_id, "linked_item_id": body.linked_item_id,
        "priority": body.priority, "tags": body.tags, "due_date": body.due_date,
        "is_archived": False, "deleted_at": None,
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    }
    result = await db.notes.insert_one(doc)
    return {"noteId": str(result.inserted_id)}

@app.patch("/api/notes/{note_id}")
async def update_note(note_id: str, body: NoteUpdate, user=Depends(get_current_user)):
    uid = str(user["_id"])
    upd = {k: v for k, v in body.dict().items() if v is not None}
    upd["updated_at"] = datetime.utcnow()
    await db.notes.update_one({"_id": ObjectId(note_id), "userId": uid}, {"$set": upd})
    return {"ok": True}

@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str, user=Depends(get_current_user)):
    uid = str(user["_id"])
    await db.notes.update_one(
        {"_id": ObjectId(note_id), "userId": uid},
        {"$set": {"deleted_at": datetime.utcnow()}}
    )
    return {"ok": True}

@app.get("/api/notes/folders")
async def get_note_folders(user=Depends(get_current_user)):
    uid = str(user["_id"])
    folders = await db.note_folders.find({"userId": uid}).to_list(None)
    return [serialize(f) for f in folders]

@app.post("/api/notes/folders")
async def create_note_folder(body: NoteFolderCreate, user=Depends(get_current_user)):
    uid = str(user["_id"])
    doc = {"userId": uid, "name": body.name, "parent_id": body.parent_id,
           "created_at": datetime.utcnow()}
    result = await db.note_folders.insert_one(doc)
    return {"folderId": str(result.inserted_id)}

@app.delete("/api/notes/folders/{folder_id}")
async def delete_note_folder(folder_id: str, user=Depends(get_current_user)):
    uid = str(user["_id"])
    await db.notes.update_many(
        {"userId": uid, "folder_id": folder_id},
        {"$set": {"folder_id": None}}
    )
    await db.note_folders.delete_one({"_id": ObjectId(folder_id), "userId": uid})
    return {"ok": True}


@app.on_event("startup")
async def startup():
    # Create indexes
    await db.users.create_index("email", unique=True)
    await db.categories.create_index([("userId", 1), ("order", 1)])
    await db.links.create_index([("userId", 1), ("createdAt", -1)])
    await db.links.create_index([("userId", 1), ("categoryId", 1)])
    print("✅ WatchList API iniciada — MongoDB conectado")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
