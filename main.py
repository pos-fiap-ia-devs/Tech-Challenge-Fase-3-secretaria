import shutil
import os
import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from src.engine.grafo_clinico import montar_grafo
from src.rag.busca_medquad import inicializar_rag
from src.db import buscar_historico

st.set_page_config(page_title="ConsultasMedica", layout="centered", page_icon="🩺")

st.markdown("""
<style>
[data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 8px; }
.status-online  { color: #2e7d32; font-weight: 700; font-size: 0.95rem; }
.status-offline { color: #c62828; font-weight: 700; font-size: 0.95rem; }
section[data-testid="stSidebar"] * { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages"      not in st.session_state: st.session_state.messages      = []
if "engines_ready" not in st.session_state: st.session_state.engines_ready = False
if "processing"    not in st.session_state: st.session_state.processing    = False
if "last_prompt"   not in st.session_state: st.session_state.last_prompt   = None


@st.cache_resource
def carregar_motores_ia():
    llm        = OllamaLLM(model="llama3.1:8b", temperature=0)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    retriever, is_vector = inicializar_rag(embeddings)
    return llm, retriever, is_vector


@st.cache_resource
def obter_grafo_compilado(_llm, _buscar_contexto):
    return montar_grafo(_llm, _buscar_contexto)


def inicializar_sistema():
    with st.status("Loading AI models...", expanded=True) as status:
        llm, retriever, is_vector = carregar_motores_ia()

        def buscar_contexto(query):
            if is_vector and retriever:
                try:
                    docs = retriever.invoke(query)
                    if docs:
                        return "\n\n---\n\n".join(doc.page_content for doc in docs)
                except Exception:
                    pass
            return ""

        st.session_state.graph_app = obter_grafo_compilado(llm, buscar_contexto)
        st.session_state.engines_ready = True
        status.update(label="Ready", state="complete", expanded=False)


def recarregar_sistema():
    carregar_motores_ia.clear()
    obter_grafo_compilado.clear()
    _faiss_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "faiss_index")
    if os.path.exists(_faiss_dir):
        shutil.rmtree(_faiss_dir)
    st.session_state.engines_ready = False
    st.session_state.messages      = []
    st.session_state.processing    = False
    st.session_state.last_prompt   = None


_RISCO_COLOR = {"VERMELHO": "🔴", "AMARELO": "🟡", "VERDE": "🟢"}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🩺 ConsultasMedica")
    st.write("Assistente Clínico · Saúde da Mulher")
    st.divider()

    if st.session_state.engines_ready:
        st.markdown('<p class="status-online">● Online</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-offline">● Carregando...</p>', unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Recarregar", use_container_width=True,
                     help="Recarrega modelos de IA e reconstrói índice RAG"):
            recarregar_sistema()
            st.rerun()
    with col2:
        if st.button("🗑️ Limpar", use_container_width=True,
                     help="Limpa histórico da conversa"):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    with st.expander("🗄️ Audit Log (SQLite)", expanded=True):
        try:
            atendimentos = buscar_historico(limit=10)
            if atendimentos:
                for a in atendimentos:
                    cor = _RISCO_COLOR.get(a["risco"], "⚪")
                    hora = a["data_hora"][11:16]
                    label = f"{cor} {hora} — {a['relato'][:30]}..."
                    with st.expander(label, expanded=False):
                        st.markdown(f"**Pergunta:** {a['relato']}")
                        st.markdown("**Resposta:**")
                        st.markdown(a.get("resposta", "—"))
            else:
                st.write("Nenhum registro ainda.")
        except Exception as e:
            st.write(f"BD indisponível: {e}")


# --- MAIN ---
st.markdown("## 🩺 Consultas Medica")
st.markdown("**Assistente Clínico · Saúde da Mulher**")
st.caption("Faça uma pergunta clínica sobre saúde da mulher.")
st.divider()

if not st.session_state.engines_ready:
    inicializar_sistema()
    st.rerun()
else:
    # Histórico
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask a question...", disabled=st.session_state.processing):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.last_prompt = prompt
        st.session_state.processing  = True
        st.rerun()

    # Resposta
    if st.session_state.processing and st.session_state.last_prompt:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = st.session_state.graph_app.invoke(
                        {"relato": st.session_state.last_prompt}
                    )
                    resposta = res["resposta_final"]
                    if res.get("protocolo_seguranca"):
                        st.error("Safety protocol activated. Please seek immediate support.")
                    st.markdown(resposta)
                    st.session_state.messages.append({"role": "assistant", "content": resposta})
                except Exception as e:
                    st.error(f"Error: {e}")
        st.session_state.processing  = False
        st.session_state.last_prompt = None
        st.rerun()
