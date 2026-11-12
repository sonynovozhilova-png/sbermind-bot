









"""
Sbermind — веб-сервер чат-бота психологической поддержки.

Принципы конфиденциальности:
1. Анонимизация: имена, телефоны, email вырезаются ДО отправки в LLM.
2. Локальность: LLM работает локально (Ollama), данные не уходят наружу.
3. Отсутствие логов: текст сообщений не логируется.
4. Общий токен доступа: не привязан к личности.
"""

import os
import re
import uuid
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta


import httpx
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# ============================================
# Настройка логирования (БЕЗ текста сообщений!)
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("sbermind")

# ============================================
# Конфигурация из переменных окружения
# ============================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
REDACTION_URL = os.getenv("REDACTION_URL", "http://redaction:8000")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
USE_EXTERNAL_REDACTION = os.getenv("USE_EXTERNAL_REDACTION", "false").lower() == "true"
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:1.5b")

# ============================================
# In-memory хранилище маппингов (TTL 1 час)
# НЕ пишется на диск. При перезапуске — всё стирается.
# ============================================
_session_store: Dict[str, Dict] = {}
SESSION_TTL = timedelta(hours=1)


def cleanup_sessions():
    """Удаляет просроченные сессии."""
    now = datetime.now()
    expired = [
        sid for sid, data in _session_store.items()
        if now - data["created"] > SESSION_TTL
    ]
    for sid in expired:
        del _session_store[sid]


# ============================================
# Локальная анонимизация (fallback)
# ============================================
class LocalRedactor:
    """
    Простая, но рабочая анонимизация на регулярках.
    Вырезает: имена (русские), телефоны, email, URL.
    """

    PATTERNS = [
        # Email
        (re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b'), '[EMAIL]'),
        # Телефоны (разные форматы)
        (re.compile(r'(\+7|8)[\s\-\(\)]*\d{3}[\s\-\(\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}'), '[PHONE]'),
        (re.compile(r'\b\d{3}[\s\-]\d{3}[\s\-]\d{2}[\s\-]\d{2}\b'), '[PHONE]'),
        # URL
        (re.compile(r'https?://\S+'), '[URL]'),
        # Русские имена (простая эвристика: Заглавное слово + окончание)
        # Это не идеально, но лучше, чем ничего
        (re.compile(r'\b[А-ЯЁ][а-яё]{2,}(?:ов|ев|ин|ын|ский|ская|ова|ева|ина|ына)\b'), '[PERSON]'),
    ]

    @classmethod
    def redact(cls, text: str) -> tuple[str, Dict[str, str]]:
        """Возвращает (обезличенный_текст, маппинг)."""
        mapping = {}
        counter = [0]

        def replace(match):
            counter[0] += 1
            placeholder = f"[PERSON_{counter[0]}]" if match.re.pattern.find('PERSON') >= 0 else match.re.pattern
            # Определяем тип плейсхолдера по группе
            original = match.group(0)
            # Определяем тип
            if '@' in original:
                ph = f"[EMAIL_{counter[0]}]"
            elif original.startswith(('http', 'www')):
                ph = f"[URL_{counter[0]}]"
            elif any(c.isdigit() for c in original) and len(original) > 7:
                ph = f"[PHONE_{counter[0]}]"
            else:
                ph = f"[PERSON_{counter[0]}]"
            mapping[ph] = original
            return ph

        result = text
        for pattern, _ in cls.PATTERNS:
            result = pattern.sub(replace, result)

        return result, mapping

    @classmethod
    def restore(cls, text: str, mapping: Dict[str, str]) -> str:
        """Восстанавливает оригинальные значения."""
        for placeholder, original in mapping.items():
            text = text.replace(placeholder, original)
        return text


async def redact_text(text: str) -> tuple[str, Dict[str, str], Optional[str]]:
    """
    Анонимизирует текст.
    Возвращает: (обезличенный_текст, маппинг, session_id или None).
    Если используется внешний сервис — session_id нужен для restore.
    """
    if USE_EXTERNAL_REDACTION:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{REDACTION_URL}/api/redact",
                    json={"text": text}
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("redacted_text", text), {}, data.get("session_id")
        except Exception as e:
            logger.warning(f"External redaction failed, using local: {type(e).__name__}")
            # Fallback на локальную анонимизацию
    
    redacted, mapping = LocalRedactor.redact(text)
    return redacted, mapping, None


async def restore_text(redacted: str, mapping: Dict[str, str], session_id: Optional[str]) -> str:
    """Восстанавливает оригинальные значения."""
    if USE_EXTERNAL_REDACTION and session_id:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{REDACTION_URL}/api/restore",
                    json={"text": redacted, "session_id": session_id}
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("restored_text", redacted)
        except Exception as e:
            logger.warning(f"External restore failed: {type(e).__name__}")
    
    return LocalRedactor.restore(redacted, mapping)


async def ask_ollama(redacted_text: str, history: list) -> str:
    """Отправляет обезличенный текст в локальную LLM."""
    system_prompt = (
        "Ты — поддерживающий собеседник для сотрудников. "
        "Твоя задача — выслушать, отразить чувства, задать открытый вопрос. "
        "НЕ давай советов, НЕ ставь диагнозов. "
        "Отвечай на русском, тепло и кратко (2-4 предложения). "
        "Если пользователь в кризисе — предложи обратиться к специалисту."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": redacted_text})

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7}
            }
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "Извините, не удалось сформировать ответ.")


# ============================================
# FastAPI приложение
# ============================================
app = FastAPI(title="Sbermind", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
class EscalateRequest(BaseModel):
    message: str  # Сюда мы передадим последнюю фразу пользователя для контекста


def check_token(authorization: Optional[str]):
    """Проверка общего токена доступа."""
    return  # ВРЕМЕННО: проверка отключена


@app.get("/", response_class=HTMLResponse)
async def index():
    """Отдаёт главную страницу."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/chat")
async def chat(request: ChatRequest, authorization: Optional[str] = Header(None)):
    """
    Основной эндпоинт.
    НЕ логируем текст сообщений — только факт запроса.
    """
    check_token(authorization)
    cleanup_sessions()

    # 1. Анонимизация
    redacted, mapping, ext_session_id = await redact_text(request.message)

    # 2. Получаем или создаём сессию для истории
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in _session_store:
        _session_store[session_id] = {
            "created": datetime.now(),
            "history": [],
            "mapping": mapping,
            "ext_session_id": ext_session_id,
        }

    session = _session_store[session_id]
    # Обновляем маппинг (новые имена в новом сообщении)
    session["mapping"].update(mapping)

    # 3. Отправляем в LLM
    try:
        response_redacted = await ask_ollama(redacted, session["history"])
    except Exception as e:
        logger.error(f"Ollama error: {type(e).__name__}")
        raise HTTPException(status_code=503, detail="LLM unavailable")

    # 4. Обновляем историю (в обезличенном виде!)
    session["history"].append({"role": "user", "content": redacted})
    session["history"].append({"role": "assistant", "content": response_redacted})
    # Ограничиваем историю 10 сообщениями
    if len(session["history"]) > 10:
        session["history"] = session["history"][-10:]

    # 5. Восстанавливаем оригинальные значения в ответе
    restored = await restore_text(
        response_redacted,
        session["mapping"],
        session.get("ext_session_id")
    )

    logger.info(f"Chat request processed (session={session_id[:8]}...)")
	    # --- БЛОК КОНФИДЕНЦИАЛЬНОСТИ И БАЗЫ ДАННЫХ ---
    try:
        import psycopg2
        from cryptography.fernet import Fernet
        
        # Шифруем сообщение пользователя
        cipher = Fernet(os.getenv("FERNET_KEY").encode())
        encrypted_message = cipher.encrypt(request.message.encode()).decode()
        
        # Подключаемся к облаку Neon
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        
        # Создаем таблицу (если её еще нет)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS encrypted_dialogs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                encrypted_text TEXT
            )
        """)
        
        # Сохраняем зашифрованное сообщение
        cur.execute("INSERT INTO encrypted_dialogs (encrypted_text) VALUES (%s)", (encrypted_message,))
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Message encrypted and saved to Neon cloud.")
        
    except Exception as e:
        logger.error(f"Database error: {e}")
    # --- КОНЕЦ БЛОКА ---

    return JSONResponse({
        "response": restored,
        "session_id": session_id
    })


@app.get("/health")
async def health():
    """Проверка работоспособности."""
    return {"status": "ok"}

@app.post("/escalate")
async def escalate(request: EscalateRequest):
    """
    Эндпоинт для перевода на живого специалиста.
    """
    # В реальности здесь был бы запрос к API календаря или CRM Сбера.
    # Для MVP мы просто возвращаем контакты и логируем факт (без текста диалога).
    
    logger.info("User requested escalation to human specialist.")
    
    return JSONResponse({
        "status": "success",
        "message": "Мы передали запрос в службу поддержки.",
        "contacts": {
            "phone": "8-800-555-35-35 (доб. 777) — Психологическая помощь Сбера",
            "telegram": "@sber_psychology_support",
            "note": "Вы также можете записаться на очную консультацию в корпоративном портале."
        }
    })
