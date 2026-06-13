# Testes das funções PURAS do server.py — as que quebram silenciosamente:
# similaridade, classificação de 429, extração de texto/tags, status de vídeo,
# serialização. Nenhum teste toca rede ou Mongo.
from datetime import datetime, timedelta

from bson import ObjectId
from jose import jwt as jose_jwt

import server


# ── _cosine ───────────────────────────────────────────────────────────────────
def test_cosine_identical_is_one():
    v = [0.5, -0.2, 0.8]
    assert abs(server._cosine(v, v) - 1.0) < 1e-9

def test_cosine_orthogonal_is_zero():
    assert abs(server._cosine([1, 0], [0, 1])) < 1e-9

def test_cosine_opposite_is_minus_one():
    assert abs(server._cosine([1, 2], [-1, -2]) + 1.0) < 1e-9

def test_cosine_empty_or_none_is_zero():
    assert server._cosine([], [1, 2]) == 0.0
    assert server._cosine(None, [1, 2]) == 0.0
    assert server._cosine([0, 0], [1, 2]) == 0.0   # norma zero não divide por zero


# ── _is_daily_quota_429 (PerDay queima a chave; PerMinute NÃO) ────────────────
def _gemini_429(quota_id="", metric=""):
    return {"error": {"details": [{"violations": [{"quotaId": quota_id, "quotaMetric": metric}]}]}}

def test_quota_429_per_day_is_daily():
    assert server._is_daily_quota_429(_gemini_429("GenerateRequestsPerDayPerProjectPerModel")) is True

def test_quota_429_per_day_in_metric_is_daily():
    assert server._is_daily_quota_429(_gemini_429("", "generate_requests_PerDay")) is True

def test_quota_429_per_minute_is_transient():
    assert server._is_daily_quota_429(_gemini_429("GenerateRequestsPerMinutePerProjectPerModel")) is False

def test_quota_429_without_details_is_transient():
    # Na dúvida, NÃO matar a chave o dia todo.
    assert server._is_daily_quota_429({}) is False
    assert server._is_daily_quota_429({"error": {"message": "rate limited"}}) is False
    assert server._is_daily_quota_429({"error": {"details": "not-a-list"}}) is False


# ── rotação de chaves: exaustão é por (chave, modelo) e por DIA ───────────────
def test_gemini_keys_available_excludes_exhausted_today(monkeypatch):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    monkeypatch.setattr(server, "GEMINI_KEYS", ["k1", "k2"])
    monkeypatch.setattr(server, "_gemini_exhausted", {("k1", "m"): today})
    assert server._gemini_keys_available("m") == ["k2"]
    # outro modelo não é afetado pela exaustão de "m"
    assert server._gemini_keys_available("outro") == ["k1", "k2"]

def test_gemini_keys_available_resets_next_day(monkeypatch):
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    monkeypatch.setattr(server, "GEMINI_KEYS", ["k1"])
    monkeypatch.setattr(server, "_gemini_exhausted", {("k1", "m"): yesterday})
    assert server._gemini_keys_available("m") == ["k1"]


# ── _html_to_text (readability) ───────────────────────────────────────────────
def test_html_to_text_prefers_article():
    html = "<nav>menu ruim</nav><article><p>Conteúdo &amp; ideia.</p></article><footer>x</footer>"
    out = server._html_to_text(html)
    assert "Conteúdo & ideia." in out
    assert "menu ruim" not in out

def test_html_to_text_strips_script_and_style():
    html = "<main><script>alert(1)</script><style>.a{}</style><p>Texto real</p></main>"
    out = server._html_to_text(html)
    assert "Texto real" in out
    assert "alert" not in out and ".a{}" not in out

def test_html_to_text_blocks_become_newlines():
    out = server._html_to_text("<p>um</p><p>dois</p>")
    assert out.splitlines()[0].strip() == "um"
    assert "dois" in out


# ── _extract_tags (parse defensivo da resposta do LLM) ────────────────────────
def test_extract_tags_parses_array():
    assert server._extract_tags('["Marketing", "Facebook Ads"]') == ["Marketing", "Facebook Ads"]

def test_extract_tags_ignores_text_around():
    assert server._extract_tags('Claro! Aqui: ["IA"] espero que ajude') == ["IA"]

def test_extract_tags_caps_at_four():
    assert len(server._extract_tags('["a","b","c","d","e","f"]')) == 4

def test_extract_tags_garbage_is_empty():
    assert server._extract_tags("desculpe, não sei") == []
    assert server._extract_tags("") == []


# ── _yt_video_id ──────────────────────────────────────────────────────────────
def test_yt_video_id_variants():
    vid = "dQw4w9WgXcQ"
    assert server._yt_video_id(f"https://www.youtube.com/watch?v={vid}") == vid
    assert server._yt_video_id(f"https://youtu.be/{vid}") == vid
    assert server._yt_video_id(f"https://www.youtube.com/embed/{vid}") == vid
    assert server._yt_video_id(f"https://www.youtube.com/shorts/{vid}") == vid
    assert server._yt_video_id("https://vimeo.com/123") == ""


# ── progresso / status de vídeo ───────────────────────────────────────────────
def test_progress_pct():
    assert server._progress_pct({"watched": True}) == 100
    assert server._progress_pct({"durationSeconds": 100, "watchedSeconds": 50}) == 50
    assert server._progress_pct({"durationSeconds": 0, "watchedSeconds": 50}) == 0
    assert server._progress_pct({}) == 0

def test_video_status_completed_and_not_started():
    assert server._video_status({"watched": True}) == "completed"
    assert server._video_status({"durationSeconds": 100, "watchedSeconds": 95}) == "completed"
    assert server._video_status({}) == "not_started"

def test_video_status_abandoned_vs_in_progress():
    old = (datetime.utcnow() - timedelta(days=20)).isoformat()
    halfway_old = {"durationSeconds": 100, "watchedSeconds": 50, "lastWatchedAt": old}
    assert server._video_status(halfway_old) == "abandoned"
    recent = (datetime.utcnow() - timedelta(days=2)).isoformat()
    halfway_recent = {"durationSeconds": 100, "watchedSeconds": 50, "lastWatchedAt": recent}
    assert server._video_status(halfway_recent) == "in_progress"


# ── serialize / deserialize_doc ───────────────────────────────────────────────
def test_serialize_strips_server_only_fields_and_converts():
    oid = ObjectId()
    when = datetime(2026, 6, 1, 12, 0, 0)
    doc = {"_id": oid, "title": "x", "createdAt": when,
           "topicEmbedding": [0.1] * 8, "contentText": "segredo grande"}
    out = server.serialize(dict(doc))
    assert out["id"] == str(oid)
    assert "topicEmbedding" not in out and "contentText" not in out
    assert out["createdAt"] == when.isoformat()

def test_serialize_none_passthrough():
    assert server.serialize(None) is None

def test_deserialize_doc_roundtrip_preserves_id_and_dates():
    oid = ObjectId()
    when = datetime(2026, 6, 1, 12, 0, 0)
    ser = server.serialize({"_id": oid, "createdAt": when, "title": "t"})
    back = server.deserialize_doc(ser)
    assert back["_id"] == oid
    assert isinstance(back["createdAt"], datetime)
    assert back["createdAt"] == when


# ── JWT ───────────────────────────────────────────────────────────────────────
def test_create_jwt_roundtrip():
    token = server.create_jwt("user-123")
    claims = jose_jwt.decode(token, server.JWT_SECRET, algorithms=[server.ALGORITHM])
    assert claims["sub"] == "user-123"
    assert claims["exp"] > datetime.utcnow().timestamp()


# ── chunks com timestamp (citação por minuto) ─────────────────────────────────
def test_parse_timedtext_extracts_time_and_text():
    xml = ('<transcript><text start="1.5" dur="3">Ol&amp;aacute; <b>mundo</b></text>'
           '<text start="12.3" dur="2">segundo trecho</text>'
           '<text start="20" dur="2">   </text></transcript>')
    segs = server._parse_timedtext(xml)
    assert len(segs) == 2                       # o vazio é descartado
    assert segs[0]["t"] == 1.5 and "mundo" in segs[0]["text"]
    assert segs[1] == {"t": 12.3, "text": "segundo trecho"}

def test_parse_timedtext_garbage_is_empty():
    assert server._parse_timedtext("") == []
    assert server._parse_timedtext("<html>not captions</html>") == []

def test_chunk_segments_groups_and_keeps_first_timestamp():
    segs = [{"t": float(i * 10), "text": "palavra " * 20} for i in range(10)]   # ~160 chars cada
    chunks = server._chunk_segments(segs, target_chars=300)
    assert len(chunks) >= 2
    assert chunks[0]["t"] == 0                  # tempo do 1º segmento do grupo
    assert chunks[1]["t"] > 0
    assert all(isinstance(c["t"], int) for c in chunks)

def test_chunk_segments_respects_cap():
    segs = [{"t": float(i), "text": "x" * 200} for i in range(100)]
    assert len(server._chunk_segments(segs, target_chars=100, cap=5)) == 5

def test_chunk_text_splits_on_sentences():
    text = ("Primeira frase completa. " * 30).strip()
    chunks = server._chunk_text(text, target_chars=200)
    assert len(chunks) >= 2
    assert all(c["t"] is None for c in chunks)
    assert all(c["text"] for c in chunks)

def test_chunk_text_empty_is_empty():
    assert server._chunk_text("") == []
    assert server._chunk_text("   ") == []


# ── importação em massa (parsers) ─────────────────────────────────────────────
def test_parse_bookmarks_html():
    html = '''<DL><p>
      <DT><A HREF="https://example.com/a" ADD_DATE="123">Artigo &amp; Cia</A>
      <DT><A HREF="https://example.com/a">duplicado</A>
      <DT><A HREF="javascript:alert(1)">ruim</A>
      <DT><A HREF="https://youtu.be/dQw4w9WgXcQ">Vídeo</A>
    </p></DL>'''
    out = server._parse_bookmarks_html(html)
    assert [b["url"] for b in out] == ["https://example.com/a", "https://youtu.be/dQw4w9WgXcQ"]
    assert out[0]["title"] == "Artigo & Cia"

def test_parse_bookmarks_empty():
    assert server._parse_bookmarks_html("") == []
    assert server._parse_bookmarks_html("<html>sem links</html>") == []

def test_parse_playlist_page():
    html = ('{"videoId":"dQw4w9WgXcQ","thumbnail":{},"title":{"runs":[{"text":"Primeiro v\u00eddeo"}]}}'
            '{"videoId":"dQw4w9WgXcQ","title":{"runs":[{"text":"dup"}]}}'
            '{"videoId":"abcdefghijk","x":1,"title":{"runs":[{"text":"Segundo"}]}}')
    out = server._parse_playlist_page(html)
    assert len(out) == 2
    assert out[0] == {"videoId": "dQw4w9WgXcQ", "title": "Primeiro vídeo"}
    assert out[1]["videoId"] == "abcdefghijk"

def test_parse_playlist_garbage():
    assert server._parse_playlist_page("") == []
    assert server._parse_playlist_page("<html>not a playlist</html>") == []


# ── extração por plataforma (helpers puros) ───────────────────────────────────
def test_unwrap_url_google_redirect():
    u = "https://www.google.com/url?q=https%3A%2F%2Fexemplo.com%2Fpost&sa=D"
    assert server._unwrap_url(u) == "https://exemplo.com/post"

def test_unwrap_url_facebook_and_passthrough():
    u = "https://l.facebook.com/l.php?u=https%3A%2F%2Fblog.com%2Fa"
    assert server._unwrap_url(u) == "https://blog.com/a"
    assert server._unwrap_url("https://site.com/x?q=1") == "https://site.com/x?q=1"
    assert server._unwrap_url("") == ""

def test_og_text_extracts_title_and_description():
    html = ('<meta property="og:title" content="T&iacute;tulo Bonito"/>'
            '<meta property="og:description" content="Uma descri&ccedil;&atilde;o."/>'
            '<meta name="description" content="Uma descri&ccedil;&atilde;o."/>')
    out = server._og_text(html)
    assert "Título Bonito" in out
    assert out.count("Uma descrição.") == 1   # dedup

def test_og_text_empty():
    assert server._og_text("") == ""
    assert server._og_text("<html><body>nada</body></html>") == ""

def test_tiktok_page_text_extracts_embedded_json():
    html = ('{"user":{"nickname":"\uc11c\uc5f0","signature":"daily vlogs"},'
            '"itemList":[{"desc":"@_sxxxyeon_ 346.4k seguidores - v\u00eddeos incr\u00edveis"},'
            '{"desc":"@_sxxxyeon_ 346.4k seguidores - v\u00eddeos incr\u00edveis"}]}')
    out = server._tiktok_page_text(html)
    assert "서연" in out
    assert "daily vlogs" in out
    assert out.count("346.4k seguidores") == 1   # dedup

def test_tiktok_page_text_empty():
    assert server._tiktok_page_text("") == ""
    assert server._tiktok_page_text("<html>nada</html>") == ""


def test_parse_playlist_new_lockup_format():
    # Formato 2025+ (lockupViewModel): contentId VIDEO + accessibility label c/ duração
    html = ('xx"contentId":"dQw4w9WgXcQ","contentType":"LOCKUP_CONTENT_TYPE_VIDEO",'
            '"rendererContext":{"accessibilityContext":{"label":"Rick Astley - Never Gonna Give You Up 3 minutos e 32 segundos"}}'
            'yy"contentId":"abcdefghijk","contentType":"LOCKUP_CONTENT_TYPE_VIDEO",'
            '"accessibilityContext":{"label":"Outro Video 10 minutes, 5 seconds"}')
    out = server._parse_playlist_page(html)
    assert len(out) == 2
    assert out[0] == {"videoId": "dQw4w9WgXcQ", "title": "Rick Astley - Never Gonna Give You Up"}
    assert out[1]["title"] == "Outro Video"
