import io
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


@st.cache_resource(show_spinner=False)
def build_vector_store(file_bytes: bytes, filename: str, api_key: str, chunk_size: int, chunk_overlap: int):
    pdf_reader = PdfReader(io.BytesIO(file_bytes))

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    docs = []
    indexed_pages = 0

    for page_num, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue

        indexed_pages += 1

        page_docs = splitter.create_documents(
            [page_text],
            metadatas=[{
                "page": page_num,
                "source": filename
            }]
        )
        docs.extend(page_docs)

    if not docs:
        raise ValueError("Impossible d'extraire un texte exploitable depuis ce PDF.")

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store, indexed_pages, len(docs)


def build_context(docs_and_scores):
    blocks = []
    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "Document")
        blocks.append(
            f"[Passage {i} | source: {source} | page: {page} | distance: {score:.4f}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def format_sources(docs_and_scores, excerpt_max_chars=800):
    formatted = []
    for doc, score in docs_and_scores:
        text = doc.page_content.strip()
        excerpt = text[:excerpt_max_chars] + ("..." if len(text) > excerpt_max_chars else "")
        formatted.append({
            "page": doc.metadata.get("page", "?"),
            "source": doc.metadata.get("source", "Document"),
            "distance": float(score),
            "excerpt": excerpt
        })
    return formatted


def main():
    st.set_page_config(
        page_title="Mon Assistant Personnel IA",
        page_icon="🤖",
        layout="wide"
    )

    st.markdown("""
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 0.98rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    .source-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Mon Assistant Personnel IA")
    st.markdown(
        "<div class='subtitle'>Charge un PDF, pose une question, puis consulte les passages réellement utilisés pour produire la réponse.</div>",
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.subheader("Paramètres")

        uploaded_file = st.file_uploader("Ajouter votre PDF", type="pdf")

        chunk_size = st.slider("Taille des passages", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chevauchement des passages", 0, 400, 200, 50)
        top_k = st.slider("Nombre maximal de passages récupérés", 1, 6, 4, 1)
        max_allowed_distance = st.slider("Distance maximale acceptée", 0.05, 1.00, 0.20, 0.05)

        st.markdown("---")
        st.caption("Plus la distance est faible, plus un passage est proche de la question.")
        st.caption("Le document PDF est utilisé dès qu'au moins un passage récupéré est sous le seuil de distance maximale acceptée.")
        st.caption("Seuls les passages sous le seuil sont utilisés dans la réponse générée.")

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("Ajoute ta clé OpenAI dans .streamlit/secrets.toml avec la variable OPENAI_API_KEY.")
        return

    if uploaded_file is None:
        st.info("Commence par importer un document PDF dans la barre latérale.")
        return

    file_bytes = uploaded_file.getvalue()

    try:
        with st.spinner("Indexation du document..."):
            vector_store, indexed_pages, chunk_count = build_vector_store(
                file_bytes=file_bytes,
                filename=uploaded_file.name,
                api_key=api_key,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    except Exception as e:
        st.error(f"Erreur lors du traitement du PDF : {e}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pages indexées", indexed_pages)
    with col2:
        st.metric("Passages créés", chunk_count)
    with col3:
        st.metric("Passages récupérés", top_k)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                st.caption(msg["mode"])
                if msg.get("sources"):
                    with st.expander("Passages sources utilisés"):
                        for i, src in enumerate(msg["sources"], start=1):
                            st.markdown(
                                f"""
<div class="source-box">
<b>Passage {i}</b><br>
Source : {src["source"]}<br>
Page : {src["page"]}<br>
Distance : {src["distance"]:.4f}
</div>
""",
                                unsafe_allow_html=True
                            )
                            st.write(src["excerpt"])

    question = st.chat_input("Posez votre question sur le document...")

    if not question:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Génération de la réponse..."):
            llm = ChatOpenAI(
                api_key=api_key,
                temperature=0,
                model="gpt-4o-mini"
            )

            docs_and_scores = vector_store.similarity_search_with_score(question, k=top_k)

            relevant_docs_and_scores = [
                (doc, score)
                for doc, score in docs_and_scores
                if score <= max_allowed_distance
            ]

            if relevant_docs_and_scores:
                best_distance = min(score for _, score in relevant_docs_and_scores)
                context = build_context(relevant_docs_and_scores)

                prompt = f"""
Tu es un assistant utile et rigoureux.

Consignes :
- Réponds d'abord à partir du contexte extrait du document.
- Si le contexte est partiel mais utile, appuie-toi dessus et complète prudemment avec tes connaissances générales.
- Si tu complètes avec tes connaissances générales, indique-le brièvement.
- Réponds en français, de manière claire et concise.

Question :
{question}

Contexte :
{context}
"""
                answer = llm.invoke(prompt).content
                mode = (
                    f"Réponse appuyée sur le document · "
                    f"{len(relevant_docs_and_scores)} passage(s) retenu(s) sur {len(docs_and_scores)} · "
                    f"meilleure distance : {best_distance:.4f}"
                )
                sources = format_sources(relevant_docs_and_scores)
            else:
                prompt = f"""
Tu es un assistant utile et rigoureux.
Réponds en français, de manière claire et concise.

Question :
{question}
"""
                answer = llm.invoke(prompt).content
                mode = "Réponse basée sur les connaissances générales du modèle"
                sources = []

        st.write(answer)
        st.caption(mode)

        if sources:
            with st.expander("Passages sources utilisés"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(
                        f"""
<div class="source-box">
<b>Passage {i}</b><br>
Source : {src["source"]}<br>
Page : {src["page"]}<br>
Distance : {src["distance"]:.4f}
</div>
""",
                        unsafe_allow_html=True
                    )
                    st.write(src["excerpt"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "mode": mode,
        "sources": sources
    })


if __name__ == "__main__":
    main()