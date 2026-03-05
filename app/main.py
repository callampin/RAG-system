import streamlit as st
from streamlit_chat import message
from typing import List, Dict, Any
import time

from src.rag_engine import RAGEngine
from src.config import config


st.set_page_config(
    page_title="RAG Soporte Técnico 24/7",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


class RAGChatApp:
    def __init__(self):
        self.engine = RAGEngine()
        self._init_session_state()

    def _init_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "engine_initialized" not in st.session_state:
            st.session_state.engine_initialized = False
        if "error" not in st.session_state:
            st.session_state.error = None

    def init_engine(self):
        try:
            self.engine.load_vectorstore()
            self.engine.initialize_qa_chain(k=4)
            st.session_state.engine_initialized = True
            st.session_state.error = None
        except FileNotFoundError as e:
            st.session_state.error = str(e)
            st.session_state.engine_initialized = False
        except Exception as e:
            st.session_state.error = f"Error initializing engine: {str(e)}"
            st.session_state.engine_initialized = False

    def render_sidebar(self):
        with st.sidebar:
            st.title("⚙️ Configuración")
            
            st.divider()
            
            st.subheader("Modelo LLM")
            st.info(f"**Activo:** {config.ACTIVE_LLM.upper()}")
            
            st.subheader("Estadísticas")
            if st.session_state.engine_initialized:
                stats = self.engine.get_stats()
                st.metric("Documentos indexados", stats["total_documents"])
                st.caption(f"Colección: {stats['collection_name']}")
            else:
                st.metric("Documentos indexados", "—")
            
            st.divider()
            
            if st.button("🔄 Reiniciar Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            if st.session_state.get("error"):
                st.error(st.session_state.error)
                st.caption("Ejecuta la ingestión primero: python -m src.ingest")

    def render_chat(self):
        st.header("🎧 Soporte Técnico 24/7")
        st.markdown("""
        Hola, soy tu asistente de soporte técnico. 
        Puedo responder preguntas sobre la documentación cargada.
        """)

        if not st.session_state.engine_initialized:
            st.warning("⚠️ El sistema no está inicializado. Ejecuta la ingestión primero.")
            return

        for i, msg in enumerate(st.session_state.messages):
            is_user = msg["role"] == "user"
            message(
                msg["content"],
                is_user=is_user,
                key=f"msg_{i}",
                avatar_style="avataaars" if is_user else "bottts",
            )

        if prompt := st.chat_input("Escribe tu pregunta..."):
            self._handle_user_message(prompt)

    def _handle_user_message(self, prompt: str):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar_style="avataaars"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar_style="bottts"):
            with st.spinner("🔍 Buscando información..."):
                try:
                    result = self.engine.query(prompt)
                    
                    response_placeholder = st.empty()
                    response_text = result["answer"]
                    response_placeholder.markdown(response_text)
                    
                    with st.expander("📄 Fuentes consultadas"):
                        for i, source in enumerate(result["sources"]):
                            st.caption(f"**Fuente {i+1}**: {source['source']} (pág. {source['page']})")
                            st.markdown(f"_{source['content']}_")
                            st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                except Exception as e:
                    error_msg = f"Error al procesar la pregunta: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    def run(self):
        self.render_sidebar()
        self.render_chat()


def main():
    app = RAGChatApp()
    app.init_engine()
    app.run()


if __name__ == "__main__":
    main()
