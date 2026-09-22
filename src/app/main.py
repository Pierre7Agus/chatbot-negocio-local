import os
# --- FIX CRÍTICO PARA VERCEL ---
# Forzar el transporte de Gemini a REST para evitar fallos de gRPC HTTP/2 en Vercel Lambda
os.environ["GEMINI_TRANSPORT"] = "rest"

import httpx
import csv
import re
import json
import asyncio
import base64
from datetime import datetime
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from fastapi import FastAPI, Request, BackgroundTasks, Response, Form
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import psycopg

from typing_extensions import TypedDict
from contextlib import asynccontextmanager

from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()

# --- Configuración de Entorno ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ADMIN_PHONE = os.getenv("ADMIN_PHONE")

# --- Configuración Meta Cloud API ---
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
GRAPH_API_URL = os.getenv("GRAPH_API_URL", "https://graph.facebook.com/v25.0")

# --- Configuración de Base de Datos Supabase (PostgreSQL + pgvector) ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

# Adapta la URL según la librería (SQLAlchemy/PGVector usa postgresql+psycopg, psycopg_pool usa postgresql://)
DATABASE_URL_PSYCOPG = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://") if DATABASE_URL.startswith("postgresql://") else DATABASE_URL
DATABASE_URL_RAW = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")

COLLECTION_KNOWLEDGE = "diseno_grafico_knowledge"
COLLECTION_CLIENTS = "clientes_memoria"

# Control de Concurrencia para Gemini en Serverless
gemini_semaphore = asyncio.Semaphore(5)

# Variables globales para clientes asíncronos y grafo
http_client: httpx.AsyncClient = None
db_pool: AsyncConnectionPool = None
graph = None

# =================================================================================
# 1. DECLARACIÓN DE APP Y HEALTH CHECK (COMPATIBILIDAD VERCEL SERVERLESS)
# =================================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, db_pool, graph
    http_client = httpx.AsyncClient(
        timeout=10.0,
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200)
    )

    # Pool de conexiones asíncronas a Supabase PostgreSQL
    try:
        db_pool = AsyncConnectionPool(
            conninfo=DATABASE_URL_RAW,
            min_size=1,
            max_size=10,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": None  # FIX CRÍTICO SUPABASE PGBOUNCER: Desactiva prepared statements para evitar DuplicatePreparedStatement
            }
        )
        await db_pool.open()
        await init_db()

        # Compilar Grafo con Checkpointer de PostgreSQL
        checkpointer = AsyncPostgresSaver(db_pool)
        await checkpointer.setup()
        graph = workflow.compile(checkpointer=checkpointer)
    except Exception as e:
        print(f"⚠️ Error inicializando checkpointer Postgres en Supabase, usando checkpointer sin persistencia: {e}")
        graph = workflow.compile()

    yield

    if db_pool:
        await db_pool.close()
    if http_client:
        await http_client.aclose()

app = FastAPI(title="WhatsApp AI Sales Agent (Supabase Multi-Tenant)", lifespan=lifespan)


TWILIO_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

# Variables de entorno de Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")  # Ej: 'whatsapp:+17372508034'

# Inicialización del cliente
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
async def root():
    """Ruta de verificación de estado para Vercel Serverless."""
    return {"status": "online", "message": "Servidor WhatsApp AI Sales Agent 100% activo en Vercel con Supabase."}

# =================================================================================
# 2. INICIALIZACIÓN DE TABLAS EN SUPABASE
# =================================================================================

async def init_db():
    """Crea la tabla ventas multi-tenant en Supabase PostgreSQL."""
    try:
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE EXTENSION IF NOT EXISTS vector;
                    
                    CREATE TABLE IF NOT EXISTS ventas (
                        id SERIAL PRIMARY KEY,
                        tenant_phone TEXT NOT NULL,
                        fecha TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        nombre_cliente TEXT NOT NULL,
                        telefono TEXT NOT NULL,
                        servicio TEXT NOT NULL,
                        monto NUMERIC(10, 2) NOT NULL DEFAULT 0.00,
                        estado TEXT DEFAULT 'Pendiente Confirmación Anticipo'
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_ventas_tenant_fecha ON ventas (tenant_phone, fecha DESC);
                """)
                await conn.commit()
                print("✅ Tablas inicializadas exitosamente en Supabase PostgreSQL.")
    except Exception as e:
        print(f"⚠️ Error inicializando base de datos en Supabase: {e}")


# --- Conexión RAG Multi-Tenant con Gemini Embeddings ---
print("Inicializando Gemini Embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=384
)

# Vectorstore Multi-tenant de Conocimientos del Negocio
vectorstore_knowledge = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_KNOWLEDGE,
    connection=DATABASE_URL_PSYCOPG,
    use_jsonb=True
)

# Vectorstore Multi-tenant de Memoria a Largo Plazo de Clientes
vectorstore_clientes = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_CLIENTS,
    connection=DATABASE_URL_PSYCOPG,
    use_jsonb=True
)


RE_MONTO = re.compile(r"[-+]?\d*\.\d+|\d+")

def limpiar_monto(valor) -> float:
    if valor is None:
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)
    coincidencias = RE_MONTO.findall(str(valor).replace(",", "."))
    return float(coincidencias[0]) if coincidencias else 0.0


def sanitizar_mensajes_para_gemini(messages: list) -> list:
    """Sanea el historial recortando mensajes para la API de Gemini."""
    if not messages:
        return []

    if isinstance(messages[-1], AIMessage) and getattr(messages[-1], "tool_calls", None):
        return messages[-2:] if len(messages) >= 2 else messages

    recientes = []
    limite_mensajes = 8
    
    for msg in reversed(messages):
        if len(recientes) >= limite_mensajes and isinstance(msg, HumanMessage):
            recientes.insert(0, msg)
            break
        recientes.insert(0, msg)

    while recientes and not isinstance(recientes[0], HumanMessage):
        recientes.pop(0)

    return recientes if recientes else [messages[-1]]


async def notificar_venta_discord(nombre: str, telefono: str, servicio: str, monto: float, tenant_phone: str = "negocio"):
    """Envía un Embed enriquecido al canal de Discord vía Webhook."""
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ Advertencia: DISCORD_WEBHOOK_URL no configurado.")
        return

    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "username": f"Bot de Ventas ({tenant_phone})",
        "embeds": [
            {
                "title": "🎉 ¡Nueva Venta Registrada!",
                "description": f"Se ha cerrado un pedido desde el bot de WhatsApp para el negocio **{tenant_phone}**.",
                "color": 3066993,  # Verde #2ECC71
                "fields": [
                    {"name": "🏢 Tenant / Negocio", "value": tenant_phone, "inline": True},
                    {"name": "👤 Cliente", "value": nombre, "inline": True},
                    {"name": "📱 Teléfono / JID", "value": telefono, "inline": True},
                    {"name": "💼 Servicio Contratado", "value": servicio, "inline": False},
                    {"name": "💵 Monto Acordado", "value": f"${monto:.2f} USD", "inline": True},
                    {"name": "⏳ Estado", "value": "Pendiente Confirmación Anticipo (50%)", "inline": True},
                ],
                "footer": {
                    "text": f"Registrado el {ahora} | WhatsApp AI Gateway (Supabase + Vercel)"
                }
            }
        ]
    }

    try:
        res = await http_client.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5.0)
        res.raise_for_status()
        print(f"✅ Alerta de venta enviada a Discord para {nombre} (Tenant {tenant_phone}).")
    except Exception as e:
        print(f"❌ Error al enviar notificación a Discord: {e}")


# --- Funciones de Memoria RAG Multi-Tenant de Clientes ---

async def recuperar_memoria_cliente(tenant_phone: str, telefono: str, consulta_actual: str) -> str:
    """Busca antecedentes del cliente en Supabase PGVector filtrando estrictamente por tenant y teléfono."""
    try:
        # Usar asyncio.to_thread con similarity_search para evitar dependencia de _async_engine en PGVector
        resultados = await asyncio.to_thread(
            vectorstore_clientes.similarity_search,
            query=f"query: {consulta_actual}",
            k=2,
            filter={"tenant_phone": tenant_phone, "telefono": telefono}
        )
        if not resultados:
            return "Sin registros previos de este cliente."
        
        recuerdos = [doc.page_content.replace("passage: ", "") for doc in resultados]
        return "\n".join(f"- {r}" for r in recuerdos)
    except Exception as e:
        print(f"Error al recuperar memoria de cliente: {e}")
        return "Sin registros previos."

llm_resumen = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    request_timeout=12,
    max_retries=1
)


def extraer_y_guardar_hechos_cliente(tenant_phone: str, telefono: str, nombre: str, texto_cliente: str, respuesta_bot: str):
    """Ejecuta la extracción de hechos del cliente y guarda en Supabase PGVector."""
    prompt = f"""Eres un analista de datos comerciales. Analiza esta interacción reciente de WhatsApp y extrae información CLAVE sobre el cliente en
1 frase concisa (rubro del negocio, nombre del cliente, preferencias visuales, requerimientos especiales, presupuesto mencionado).
Si el mensaje solo contiene saludos, despedidas o preguntas genéricas sin datos sobre el perfil del cliente, responde únicamente la palabra:
DESCARTAR.

Cliente ({nombre}): {texto_cliente}
Bot: {respuesta_bot}

Hecho clave extraído:"""

    try:
        res_message = llm_resumen.invoke([HumanMessage(content=prompt)])
        raw_content = res_message.content

        if isinstance(raw_content, list):
            resultado = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in raw_content]).strip()
        else:
            resultado = str(raw_content).strip()

        if "DESCARTAR" not in resultado and len(resultado) > 10:
            doc = Document(
                page_content=f"passage: {resultado}",
                metadata={
                    "tenant_phone": tenant_phone,
                    "telefono": telefono,
                    "nombre": nombre,
                    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
            )
            vectorstore_clientes.add_documents([doc])
            print(f"🧠 Memoria guardada en Supabase PGVector para {telefono} (Tenant {tenant_phone}): {resultado}")
    except Exception as e:
        print(f"Error al sintetizar o guardar memoria de cliente: {e}")


# --- Herramientas del Agente Multi-Tenant ---

@tool
async def consultar_servicios_y_politicas(consulta: str, tenant_phone: str = "default_tenant") -> str:
    """Consulta la base de conocimientos RAG sobre precios, servicios y políticas del negocio específico."""
    try:
        # Usar asyncio.to_thread para invocación segura de PGVector similarity_search
        docs = await asyncio.to_thread(
            vectorstore_knowledge.similarity_search,
            query=consulta.strip(),
            k=3,
            filter={"tenant_phone": tenant_phone}
        )
        if not docs:
            return "No se encontró información específica en los documentos del negocio."
            
        contenido = "\n\n---\n\n".join([d.page_content for d in docs])
        return contenido[:2000]
    except Exception as e:
        print(f"Error en consulta RAG multi-tenant: {e}")
        return "No se pudo recuperar la información en este momento."

@tool
async def registrar_venta_cerrada(nombre_cliente: str, telefono: str, servicio: str, monto: str, tenant_phone: str = "default_tenant") -> str:
    """Registra una venta cerrada en Supabase PostgreSQL para el tenant actual y envía alerta a Discord."""
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    monto_numerico = limpiar_monto(monto)
    
    try:
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO ventas (tenant_phone, fecha, nombre_cliente, telefono, servicio, monto, estado)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (tenant_phone, fecha_actual, nombre_cliente, telefono, servicio, monto_numerico, "Pendiente Confirmación Anticipo")
                )
                await conn.commit()
    except Exception as e:
        print(f"Error al escribir venta en Supabase: {e}")

    await notificar_venta_discord(
        nombre=nombre_cliente,
        telefono=telefono,
        servicio=servicio,
        monto=monto_numerico,
        tenant_phone=tenant_phone
    )

    return "Venta registrada con éxito. Se ha guardado en la base de datos de Supabase y alertado al equipo."


tools = [consultar_servicios_y_politicas, registrar_venta_cerrada]
tools_by_name = {t.name: t for t in tools}

# --- Inicialización del Modelo Gemini ---
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    request_timeout=10,
    max_retries=1
)

llm_gemini_fallback = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    request_timeout=10,
    max_retries=1
)

gemini_con_tools = llm_gemini.bind_tools(tools)
fallback_con_tools = llm_gemini_fallback.bind_tools(tools)

router_llm = gemini_con_tools.with_fallbacks(
    [fallback_con_tools],
    exceptions_to_handle=(Exception,)
)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tenant_phone: str

SYSTEM_PROMPT = """Eres el asesor comercial de la agencia para el negocio '{tenant_phone}'. Tu objetivo es asesorar a clientes potenciales sobre nuestros servicios digitales y cerrar ventas.

REGLAS DE OPERACIÓN:
1. Solo puedes ofrecer información y tarifas obtenidas a través de `consultar_servicios_y_politicas`. Pasa siempre tu tenant_phone '{tenant_phone}'. Nunca inventes precios ni descuentos.
2. Solo invoca `registrar_venta_cerrada` si el cliente confirma explícitamente la compra y tienes su nombre. Pasa tenant_phone '{tenant_phone}'. Recuerda siempre el anticipo del 50%.
3. Si en el contexto del mensaje se incluyen "Antecedentes del cliente", úsalos para personalizar tu trato amablemente sin ser invasivo.
4. Respuestas concisas, comerciales y directas, óptimas para WhatsApp.
"""

async def call_model(state: AgentState):
    tenant = state.get("tenant_phone", "default_tenant")
    mensajes_recientes = sanitizar_mensajes_para_gemini(state["messages"])
    system_msg = SystemMessage(content=SYSTEM_PROMPT.format(tenant_phone=tenant))
    
    messages = [system_msg] + mensajes_recientes
    response = await router_llm.ainvoke(messages)
    return {"messages": [response]}


async def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    tenant = state.get("tenant_phone", "default_tenant")
    if not getattr(last_message, "tool_calls", None):
        return {"messages": []}
        
    results = []
    for tool_call in last_message.tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        tool_args = dict(tool_call["args"])
        if "tenant_phone" in tool_fn.args and "tenant_phone" not in tool_args:
            tool_args["tenant_phone"] = tenant
            
        output = await tool_fn.ainvoke(tool_args)
        results.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
        
    return {"messages": results}

def route_after_model(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None) and len(last_message.tool_calls) > 0:
        return "tools"
    return "__end__"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_model, {"tools": "tools", "__end__": END})
workflow.add_edge("tools", "agent")


# --- Reporte Diario Discord (Multi-Tenant / All) ---

async def enviar_reporte_diario_discord(tenant_phone: Optional[str] = None):
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ DISCORD_WEBHOOK_URL no configurado para el reporte diario.")
        return

    hoy_str = datetime.now().strftime("%Y-%m-%d")
    archivo_csv = f"reporte_ventas_{tenant_phone or 'global'}_{hoy_str}.csv"

    try:
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                if tenant_phone:
                    await cur.execute(
                        "SELECT id, tenant_phone, fecha, nombre_cliente, telefono, servicio, monto, estado FROM ventas WHERE tenant_phone = %s AND fecha::text LIKE %s ORDER BY id ASC",
                        (tenant_phone, f"{hoy_str}%")
                    )
                else:
                    await cur.execute(
                        "SELECT id, tenant_phone, fecha, nombre_cliente, telefono, servicio, monto, estado FROM ventas WHERE fecha::text LIKE %s ORDER BY id ASC",
                        (f"{hoy_str}%",)
                    )
                filas = await cur.fetchall()
                columnas = [desc[0] for desc in cur.description]

        total_ventas = len(filas)
        if total_ventas == 0:
            payload = {
                "username": "Cierre Diario de Ventas",
                "embeds": [{
                    "title": f"📊 Cierre Diario ({tenant_phone or 'Global'}) — {hoy_str}",
                    "description": "Hoy no se registraron nuevas ventas confirmadas en Supabase.",
                    "color": 9807270
                }]
            }
            await http_client.post(DISCORD_WEBHOOK_URL, json=payload)
            return

        with open(archivo_csv, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(columnas)
            for fila in filas:
                writer.writerow(list(fila))

        total_recaudado = sum(limpiar_monto(fila[6]) for fila in filas)

        resumen_items = "\n".join([
            f"• **{fila[3]}** ({fila[1]}): {fila[5]} (${limpiar_monto(fila[6]):.2f})"
            for fila in filas[:10]
        ])
        if total_ventas > 10:
            resumen_items += f"\n*... y {total_ventas - 10} más en el archivo adjunto.*"

        embed = {
            "title": f"📊 Cierre de Ventas Diario ({tenant_phone or 'Global'}) — {hoy_str}",
            "description": f"Resumen consolidado:\n\n{resumen_items}",
            "color": 3066993,
            "fields": [
                {"name": "💼 Total Contratos", "value": str(total_ventas), "inline": True},
                {"name": "💵 Monto Proyectado", "value": f"${total_recaudado:.2f} USD", "inline": True},
                {"name": "🏦 Anticipos (50%)", "value": f"${(total_recaudado * 0.5):.2f} USD", "inline": True}
            ],
            "footer": {"text": "WhatsApp AI Gateway (Supabase) | Adjunto: CSV oficial del día"}
        }

        with open(archivo_csv, "rb") as f:
            files = {"file": (archivo_csv, f.read(), "text/csv")}
            data = {"payload_json": json.dumps({"username": "Cierre Diario de Ventas", "embeds": [embed]})}
            res = await http_client.post(DISCORD_WEBHOOK_URL, data=data, files=files)
            if res.status_code in [200, 204]:
                print(f"✅ Reporte diario enviado a Discord ({total_ventas} ventas).")

        if os.path.exists(archivo_csv):
            os.remove(archivo_csv)

    except Exception as e:
        print(f"❌ Error generando reporte diario: {e}")


# --- Endpoint para Vercel Cron ---

@app.get("/api/cron/daily-report")
async def vercel_cron_daily_report(request: Request):
    """Endpoint invocado por Vercel Cron Jobs para emitir el reporte diario a las 20:00."""
    print("⏰ Ejecutando Vercel Cron Job: Reporte Diario de Ventas...")
    await enviar_reporte_diario_discord()
    return JSONResponse(content={"status": "success", "message": "Reporte diario procesado"})


# --- Auxiliares Meta Cloud API ---

LIMITES_META = {
    "texto": 3000,
    "imagen": 5 * 1024 * 1024,
    "audio": 3 * 1024 * 1024
}

def limpiar_numero(numero: str) -> str:
    if not numero:
        return ""
    return numero.replace("@s.whatsapp.net", "").replace("+", "").replace(" ", "").replace("@lid", "")

async def descargar_media_meta_con_limite(media_id: str):
    if not WHATSAPP_TOKEN or not media_id:
        return None, None, "error_config"

    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    try:
        res_info = await http_client.get(f"{GRAPH_API_URL}/{media_id}", headers=headers, timeout=10.0)
        if res_info.status_code != 200:
            return None, None, "error_metadata"
        
        info = res_info.json()
        file_size = info.get("file_size", 0)
        mime = info.get("mime_type", "")
        media_url = info.get("url")

        if "image" in mime and file_size > LIMITES_META["imagen"]:
            return None, mime, "imagen_muy_grande"
        if "audio" in mime and file_size > LIMITES_META["audio"]:
            return None, mime, "audio_muy_largo"

        if not media_url:
            return None, mime, "url_invalida"

        res_bin = await http_client.get(media_url, headers=headers, timeout=20.0)
        if res_bin.status_code != 200:
            return None, mime, "error_descarga"
        
        b64_data = base64.b64encode(res_bin.content).decode("utf-8")
        return b64_data, mime, "ok"
    except Exception as e:
        print(f"❌ Excepción descargando media: {e}")
        return None, None, "error_excepcion"


async def enviar_mensaje_whatsapp(to: str, mensaje: str):
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_WHATSAPP_NUMBER:
        print("⚠️ Credenciales de Twilio (SID, Token o Número) no configuradas en el entorno.")
        return

    # Formatear el destinatario asegurando el prefijo 'whatsapp:'
    destinatario = to if to.startswith("whatsapp:") else f"whatsapp:{limpiar_numero(to)}"

    try:
        # Petición a la API de Twilio (se usa asyncio.to_thread porque la librería oficial es síncrona)
        message = await asyncio.to_thread(
            client.messages.create,
            body=mensaje,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=destinatario
        )
        print(f"📤 Mensaje enviado vía Twilio a {destinatario}. SID: {message.sid}")
    except Exception as e:
        print(f"❌ Error enviando WhatsApp vía Twilio: {e}")

import base64
import httpx

async def descargar_media_twilio_con_limite(media_url: str):
    """
    Descarga archivos multimedia alojados en los servidores de Twilio
    usando Basic Auth (TWILIO_ACCOUNT_SID y TWILIO_AUTH_TOKEN).
    """
    auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Petición HEAD o GET para comprobar tamaño
            response = await client.get(media_url, auth=auth, timeout=15.0)
            response.raise_for_status()
            
            content_length = len(response.content)
            mime_type = response.headers.get("content-type", "")
            
            # Validaciones de tamaño
            if "image" in mime_type and content_length > 5 * 1024 * 1024:
                return None, mime_type, "imagen_muy_grande"
            if "audio" in mime_type and content_length > 3 * 1024 * 1024:
                return None, mime_type, "audio_muy_largo"
                
            b64_data = base64.b64encode(response.content).decode("utf-8")
            return b64_data, mime_type, "ok"
            
    except Exception as e:
        print(f"❌ Error descargando media de Twilio: {e}")
        return None, None, "error"


async def procesar_mensaje_ia(
    tenant_phone: str,
    remote_jid: str, 
    nombre_remitente: str, 
    texto: str,
    message_type: str = "text",
    media_url_or_b64: Optional[str] = None, # Ahora Twilio pasa una URL (MediaUrl0) o base64
    mime_type: Optional[str] = None
):
    telefono = limpiar_numero(remote_jid)
    thread_id = f"{tenant_phone}_{telefono}"
    config = {"configurable": {"thread_id": thread_id, "tenant_phone": tenant_phone}}
    
    # Límite de texto genérico (por ejemplo, 4000 caracteres)
    LIMITES_TEXTO = 4000
    if message_type == "text" and len(texto) > LIMITES_TEXTO:
        await enviar_mensaje_whatsapp(
            remote_jid,
            "⚠️ Tu mensaje es demasiado largo. Por favor envíame tu consulta resumida en un par de líneas."
        )
        return

    base64_final = None
    mime_final = mime_type

    if message_type in ["image", "audio", "video", "document"] and media_url_or_b64:
        # Si recibimos una URL de Twilio (empieza con http/https), la descargamos usando Auth de Twilio
        if media_url_or_b64.startswith("http"):
            b64_data, mime_detected, estado = await descargar_media_twilio_con_limite(media_url_or_b64)
            if estado == "imagen_muy_grande":
                await enviar_mensaje_whatsapp(remote_jid, "⚠️ La imagen pesa más de 5 MB.")
                return
            elif estado == "audio_muy_largo":
                await enviar_mensaje_whatsapp(remote_jid, "⚠️ El audio es muy largo (máx 3 MB).")
                return
            elif estado == "ok":
                base64_final = b64_data
                mime_final = mime_detected or mime_type
        else:
            base64_final = media_url_or_b64

    # Recuperar memoria RAG multi-tenant del cliente
    consulta_memoria = texto or ("imagen adjunta" if message_type == "image" else "audio de voz")
    antecedentes = await recuperar_memoria_cliente(tenant_phone, telefono, consulta_memoria)
    
    prompt_contexto = (
        f"[Antecedentes del cliente en sistema: {antecedentes}]\n"
        f"[Cliente: {nombre_remitente}, Tel: {telefono}, Tenant: {tenant_phone}]"
    )
    
    content_blocks = []
    
    if message_type == "image" and base64_final:
        clean_mime = mime_final.split(";")[0] if mime_final else "image/jpeg"
        prompt_texto = f"{prompt_contexto}\n[El cliente envió una imagen con el comentario: '{texto}']\nAnaliza la imagen."
        content_blocks.append({"type": "text", "text": prompt_texto})
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{clean_mime};base64,{base64_final}"}
        })
    elif message_type == "audio" and base64_final:
        clean_mime = mime_final.split(";")[0] if mime_final else "audio/ogg"
        prompt_texto = f"{prompt_contexto}\n[El cliente envió una nota de voz]. Escucha atentamente y responde."
        content_blocks.append({"type": "text", "text": prompt_texto})
        content_blocks.append({
            "type": "media",
            "mime_type": clean_mime,
            "data": base64_final
        })
    else:
        prompt_texto = f"{prompt_contexto}: {texto}"
        content_blocks.append({"type": "text", "text": prompt_texto})
    
    async with gemini_semaphore:
        output = await graph.ainvoke(
            {"messages": [HumanMessage(content=content_blocks)], "tenant_phone": tenant_phone},
            config=config
        )
    
    raw_content = output["messages"][-1].content
    if isinstance(raw_content, list):
        respuesta_final = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in raw_content])
    else:
        respuesta_final = str(raw_content)
        
    await enviar_mensaje_whatsapp(remote_jid, respuesta_final)
    
    texto_resumen = texto or (f"[{message_type.upper()} enviado por cliente]" if message_type != "text" else "")
    asyncio.create_task(
        asyncio.to_thread(
            extraer_y_guardar_hechos_cliente,
            tenant_phone, telefono, nombre_remitente, texto_resumen, respuesta_final
        )
    )

async def obtener_resumen_ventas_hoy(tenant_phone: str) -> str:
    """Consulta las ventas del día en Supabase para un tenant específico."""
    hoy_str = datetime.now().strftime("%Y-%m-%d")
    try:
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT nombre_cliente, servicio, monto FROM ventas WHERE tenant_phone = %s AND fecha::text LIKE %s ORDER BY id ASC",
                    (tenant_phone, f"{hoy_str}%")
                )
                filas = await cur.fetchall()

        if not filas:
            return f"📊 *Reporte del Día ({hoy_str}) - Tenant {tenant_phone}*\n\nNo se han registrado ventas el día de hoy."

        total_ventas = len(filas)
        total_recaudado = sum(limpiar_monto(f[2]) for f in filas)
        anticipos = total_recaudado * 0.5

        lineas = [f"📊 *REPORTE DIARIO DE VENTAS ({hoy_str}) - Tenant {tenant_phone}*\n"]
        for idx, f in enumerate(filas, 1):
            monto = limpiar_monto(f[2])
            lineas.append(f"{idx}. *{f[0]}* — {f[1]} (${monto:.2f})")

        lineas.append("\n" + "—" * 20)
        lineas.append(f"💼 *Total contratos:* {total_ventas}")
        lineas.append(f"💵 *Monto proyectado:* ${total_recaudado:.2f} USD")
        lineas.append(f"🏦 *Anticipos requeridos (50%):* ${anticipos:.2f} USD")

        return "\n".join(lineas)
    except Exception as e:
        return f"❌ Error generando resumen: {e}"


@app.get("/webhook")
async def verificar_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("✅ Webhook verificado exitosamente por Meta Cloud API.")
        return PlainTextResponse(content=challenge)
    
    return Response(content="Error de verificación", status_code=403)


@app.post("/webhook/twilio")
async def recibir_webhook_twilio(
    background_tasks: BackgroundTasks,
    From: str = Form(""),          # Número del cliente (ej: 'whatsapp:+593991034932')
    Body: str = Form(""),          # Texto del mensaje
    ProfileName: str = Form("Cliente"), # Nombre en WhatsApp
    NumMedia: int = Form(0),       # Cantidad de archivos multimedia
    MediaUrl0: str = Form(None),   # URL de la imagen/audio si existe
    MediaContentType0: str = Form(None) # MIME type
):
    try:
        # 1. Limpiar número del cliente (quitar 'whatsapp:' si viene presente)
        sender_phone = limpiar_numero(From.replace("whatsapp:", ""))
        remote_jid = sender_phone
        tenant_phone = "default_tenant"  # Puedes ajustarlo según tu lógica

        print(f"🔥 MENSAJE RECIBIDO DE {sender_phone} ({ProfileName}): '{Body}'")

        # 2. Manejo de archivos multimedia (imágenes, audios, etc.)
        msg_type = "text"
        media_id = None
        mime = MediaContentType0

        if NumMedia > 0:
          mime_lower = (MediaContentType0 or "").lower()
          if "image" in mime_lower:
            msg_type = "image"
          elif "audio" in mime_lower:
            msg_type = "audio"
          else:
            msg_type = "document"
        media_id = MediaUrl0

        # 3. Validar si es Admin
        es_admin = bool(ADMIN_PHONE and limpiar_numero(ADMIN_PHONE) in sender_phone)

        if es_admin and Body.lower() in ["/reporte", "/ventas", "!reporte"]:
            print(f"👑 Comando admin desde {sender_phone}: {Body}")
            reporte_texto = await obtener_resumen_ventas_hoy(tenant_phone)
            
            # Asegúrate de usar tu función de envío adaptada a Twilio
            await enviar_mensaje_whatsapp(From, reporte_texto)
            return Response(content="<Response></Response>", media_type="text/xml")

        # 4. Enviar a BackgroundTask para que tu agente LangGraph/Gemini responda
        if remote_jid and (Body or media_id):
            print(f"🚀 Enviando a BackgroundTask para {remote_jid}: {Body}")
            background_tasks.add_task(
                procesar_mensaje_ia, 
                tenant_phone, 
                From,  # Pasamos el JID completo 'whatsapp:+593...' para responder por Twilio
                ProfileName, 
                Body, 
                msg_type, 
                media_id, 
                mime
            )

    except Exception as e:
        print(f"❌ Error procesando webhook de Twilio: {e}")

    # Twilio requiere que respondas con TwiML (XML vacío de HTTP 200)
    return Response(content="<Response></Response>", media_type="text/xml")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
