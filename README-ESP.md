# RAG Personalizado para Atención al Cliente

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Licencia-MIT-green" alt="Licencia MIT">
  <img src="https://img.shields.io/badge/Docker-Listo-2496ED?logo=docker" alt="Docker Listo">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Inglés-blue" alt="English"></a>
</p>

> *"Tu agente de soporte técnico con IA disponible 24/7 — respondiendo preguntas de clientes SOLO desde tu documentación, eliminando alucinaciones y generando confianza."*

---

## El Problema

El soporte técnico tradicional enfrenta tres desafíos críticos:

1. **Preguntas Repetitivas** — Los equipos de soporte dedican más del 60% de su tiempo a responder las mismas preguntas una y otra vez.

2. **Costos Elevados** — El soporte humano 24/7 es caro y difícil de escalar en momentos de alta demanda.

3. **Alucinaciones de IA** — Los chatbots genéricos inventan respuestas peligrosas que dañan la confianza de la marca y pueden generar problemas legales.

---

## Cómo Funciona

| Paso | Descripción |
|------|-------------|
| **1. Ingestión** | Los PDFs se cargan y dividen en fragmentos superpuestos (1000 caracteres / 200 de superposición) para preservar el contexto |
| **2. Vectorización** | Cada fragmento se convierte en un vector de embedding usando el modelo de embedding del mismo proveedor LLM |
| **3. Búsqueda Semántica** | Las consultas de los usuarios se comparan con la base de datos vectorial para recuperar el contexto más relevante |
| **4. Generación Restringida** | El LLM (temperatura=0.0) genera respuestas SOLO a partir del contexto recuperado — no puede alucinar |

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SISTEMA RAG DE SOPORTE AL CLIENTE                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
   │  Archivos   │            │   Docker    │            │   .env      │
   │  PDF        │            │ Container   │            │ (API Keys)  │
   │(docs/manuals)            │             │            │             │
   └─────────────┘            └─────────────┘            └─────────────┘
          │                           │                           │
          ▼                           │                           │
   ┌─────────────┐                    │                           │
   │  PyPDF      │                    │                           │
   │  Loader     │                    │                           │
   └─────────────┘                    │                           │
          │                           │                           │
          ▼                           │                           │
   ┌─────────────┐                    │                           │
   │  Text       │                    │                           │
   │  Splitter   │                    │                           │
   │ (1000/200)  │                    │                           │
   └─────────────┘                    │                           │
          │                           │                           │
          ▼                           │                           │
   ┌─────────────┐                    │                           │
   │  ChromaDB   │◄───────────────────┘                           │
   │  Vector     │                                                │
   │  Store      │                                                │
   └─────────────┘                                                │
          │                                                       │
          ▼                                                       │
   ┌──────────────────────────────────────────────────────────────┐
   │                      LLM FACTORY                             │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
   │  │   Gemini    │  │   OpenAI    │  │  MiniMax    │           │
   │  │   3 Flash   │  │ GPT-4o Mini │  │    M2.5     │           │
   │  └─────────────┘  └─────────────┘  └─────────────┘           │
   │                         │                                    │
   │                    get_llm()                                 │
   │                    get_embeddings()                          │
   └──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                    RAG ENGINE                               │
   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
   │  │  Retrieval  │───►│   Prompt    │───►│     LLM     │      │
   │  │  (ChromaDB) │    │  Template   │    │ (temp=0.0)  │      │
   │  └─────────────┘    └─────────────┘    └─────────────┘      │
   │                                                   │         │
   │                              ┌────────────────────┘         │
   │                              ▼                              │
   │                     ┌─────────────────┐                     │
   │                     │ "No lo sé"      │                     │
   │                     │ si no hay       │                     │
   │                     │ contexto        │                     │
   │                     └─────────────────┘                     │
   └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                      UI DE STREAMLIT                        │
   │  ┌─────────────────────────────────────────────────────┐    │
   │  │  💬 Interfaz de Chat  │  📄 Fuentes  │  ⚙️ Config   │    │
   │  └─────────────────────────────────────────────────────┘    │
   └─────────────────────────────────────────────────────────────┘
```

---

## Tecnologías

| Tecnología | Propósito |
|------------|-----------|
| **Python 3.11+** | Lenguaje principal |
| **LangChain** | Orquestación RAG y abstracción de LLMs |
| **ChromaDB** | Almacenamiento vectorial local |
| **Streamlit** | Interfaz web / Chat |
| **Docker** | Despliegue en contenedores |

---

## Características Clave

- **🤖 Agnóstico Multi-LLM** — Cambia entre Gemini 3, OpenAI GPT-4o y MiniMax M2.5 cambiando una variable de entorno
- **🛡️ Cero Alucinaciones** — Temperatura configurada en 0.0; el sistema solo responde desde los documentos ingeridos o admite que no sabe
- **💾 Vector Store Local** — ChromaDB persiste los embeddings localmente; no requiere servicios externos en la nube
- **📄 Soporte PDF** — Carga documentación técnica, manuales y guías en formato PDF
- **🔄 Superposición de Contexto** — Fragmentos de 1000 caracteres con 200 de superposición preservan el contexto en los límites
- **🐳 Listo para Docker** — Containerización completa con docker-compose para despliegue fácil

---

## Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd RAG-system
```

### 2. Configurar variables de entorno

Copia el archivo de entorno de ejemplo y agrega tus API keys:

```bash
cp .env.example .env
```

Edita `.env` con tu proveedor LLM preferido:

```env
# Configuración del LLM
ACTIVE_LLM=gemini          # Opciones: gemini, openai, minimax

# API Keys - Reemplaza con tus claves reales
GEMINI_API_KEY=tu_clave_aqui
OPENAI_API_KEY=
MINIMAX_API_KEY=
MINIMAX_GROUP_ID=

# Rutas
CHROMA_PERSIST_DIR=./vectorstore
DATA_PATH=./data/pdfs

# Configuración RAG
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TEMPERATURE=0.0
MAX_TOKENS=2000
```

### 3. Coloca tus documentos PDF

Agrega tu documentación técnica, manuales o guías de soporte en:

```
data/pdfs/
```

---

## Despliegue con Docker

### Paso 1: Ejecutar el pipeline de ingestión

Esto procesa tus PDFs y crea el almacén vectorial:

```bash
docker-compose --profile ingest run rag-ingest
```

### Paso 2: Iniciar la aplicación

```bash
docker-compose up
```

### Paso 3: Acceder a la interfaz

Abre tu navegador y navega a:

```
http://localhost:8501
```

---

## Estructura del Proyecto

```
RAG-system/
├── app/
│   ├── __init__.py
│   └── main.py              # Interfaz de chat Streamlit
├── src/
│   ├── __init__.py
│   ├── config.py            # Cargador de variables de entorno
│   │   ├── llm_factory.py   # Fábrica LLM/Embeddings (agnóstica)
│   │   ├── document_loader.py  # Cargador PyPDF
│   │   ├── text_splitter.py     # Chunking (1000/200 superposición)
│   │   ├── ingest.py        # Script de ingestión (CLI)
│   │   ├── chain_builder.py    # Constructor de cadena RetrievalQA
│   │   └── rag_engine.py    # Motor RAG principal
├── data/
│   └── pdfs/                # Documentos PDF fuente
├── vectorstore/             # Almacenamiento persistente ChromaDB
├── tests/
│   ├── test_llm_factory.py
│   └── test_rag_engine.py
├── .env.example             # Plantilla de entorno
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Aviso Legal

> ⚠️ **Este es un proyecto de portafolio.** Este sistema está diseñado para demostración y propósitos educativos. Antes del despliegue en producción, considera:
> - Implementar autenticación y autorización adecuadas
> - Agregar limitación de velocidad y monitoreo
> - Proteger las API keys y variables de entorno
> - Escalar el vector store para grandes colecciones de documentos

---

<p align="center">
  <strong>Construido con ❤️ usando LangChain + ChromaDB + Streamlit</strong>
</p>
