
import os
import shutil
import tempfile
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

try:
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType
    DOCLING_AVAILABLE = True
except Exception:
    DOCLING_AVAILABLE = False


from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_cohere import CohereRerank


from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever


from serpapi import GoogleSearch


from langgraph.prebuilt import create_react_agent


try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

try:
    import google.genai as genai_clientlib
except Exception:
    genai_clientlib = None


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")    
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")    
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not (GOOGLE_API_KEY or GEMINI_API_KEY):
    raise ValueError("EX: .env içinde GOOGLE_API_KEY veya GEMINI_API_KEY ekleyiniz (Gemini 2.5 için).")
if not COHERE_API_KEY:
    raise ValueError("EX: .env içinde COHERE_API_KEY ekleyiniz (Cohere rerank).")
if not SERPAPI_API_KEY:
    raise ValueError("EX: .env içinde SERPAPI_API_KEY ekleyiniz (SerpAPI).")


USE_GOOGLE_KEY = GOOGLE_API_KEY if GOOGLE_API_KEY else GEMINI_API_KEY

if genai_clientlib is not None:
    try:
        genai_clientlib.configure(api_key=USE_GOOGLE_KEY)
        genai_client = genai_clientlib
    except Exception:
        genai_client = None
else:
    genai_client = None


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-2.5-flash"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
VECTORSTORE_PERSIST_DIR = "chroma_db"


vectordb = None
qa_chain = None


reranker_global = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-multilingual-v3.0", top_n=3)


if ChatGoogleGenerativeAI is None:
    raise ImportError("langchain_google_genai not installed; please install and retry.")
llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, api_key=USE_GOOGLE_KEY)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md"}

def ensure_local_path(uploaded):
    """Return local filesystem path for Gradio uploaded file (or string path)."""
    if uploaded is None:
        return None
    if isinstance(uploaded, str) and os.path.exists(uploaded):
        return uploaded
    path = getattr(uploaded, "name", None) or getattr(uploaded, "filepath", None)
    if path and os.path.exists(path):
        return path

    try:
        data = None
        if hasattr(uploaded, "file"):
            uploaded.file.seek(0)
            data = uploaded.file.read()
        elif hasattr(uploaded, "read"):
            uploaded.seek(0)
            data = uploaded.read()
        if data is None:
            return None
        suffix = Path(getattr(uploaded, "name", "upload")).suffix or ""
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(tmp, "wb") as f:
            if isinstance(data, str):
                f.write(data.encode("utf-8"))
            else:
                f.write(data)
        return tmp
    except Exception as e:
        logging.exception("ensure_local_path hata")
        return None

def load_file_to_documents(local_path):
    """Return list of langchain Documents for a given local file path."""
    if not local_path or not os.path.exists(local_path):
        logging.error(f"File does not exist: {local_path}")
        return []
    
    ext = Path(local_path).suffix.lower()
    docs = []
    
    logging.info(f"Loading file: {local_path}, extension: {ext}, size: {os.path.getsize(local_path)} bytes")
    
    try:
        if ext in TEXT_EXTENSIONS:
            logging.info("Loading as text file")
            # Farklı encoding'ler deneme
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    loader = TextLoader(local_path, encoding=encoding)
                    docs = loader.load()
                    logging.info(f"Successfully loaded with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    logging.warning(f"Failed to load with encoding: {encoding}")
                    continue
                    
        elif ext == ".pdf":
            logging.info(f"Loading PDF file, docling available: {DOCLING_AVAILABLE}")
            if DOCLING_AVAILABLE:
                try:
                    loader = DoclingLoader(file_path=[local_path], export_type=ExportType.DOC_CHUNKS)
                    docs = loader.load()
                    logging.info(f"Loaded PDF with docling: {len(docs)} chunks")
                except Exception as e:
                    logging.error(f"Docling failed, falling back to PyPDF: {e}")
                    loader = PyPDFLoader(local_path)
                    docs = loader.load()
                    logging.info(f"Loaded PDF with PyPDF: {len(docs)} pages")
            else:
                loader = PyPDFLoader(local_path)
                docs = loader.load()
                logging.info(f"Loaded PDF with PyPDF: {len(docs)} pages")
                
        elif ext in IMAGE_EXTENSIONS:
            logging.info(f"Loading image file, docling available: {DOCLING_AVAILABLE}")
            if DOCLING_AVAILABLE:
                try:
                    loader = DoclingLoader(file_path=[local_path], export_type=ExportType.DOC_CHUNKS)
                    docs = loader.load()
                    logging.info(f"Loaded image with docling: {len(docs)} chunks")
                except Exception as e:
                    logging.error(f"Image OCR with docling failed: {e}")
                    docs = []
            else:
                logging.warning("Image OCR (docling) not available; image content will not be extracted.")
                docs = []
                
        else:
            # Genel metin dosyası olarak yükleme deneme
            logging.info("Loading as generic text file")
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(local_path, "r", encoding=encoding, errors="ignore") as f:
                        txt = f.read()
                    if txt.strip():  # Boş dosya kontrolü
                        docs = [{'page_content': txt, 'metadata': {'source': local_path}}]
                        logging.info(f"Successfully loaded as text with encoding: {encoding}")
                        break
                except Exception as e:
                    logging.warning(f"Failed to load as text with encoding {encoding}: {e}")
                    continue
    
    except Exception as e:
        logging.exception(f"load_file_to_documents error for {local_path}: {e}")
        docs = []

    # Document normalization
    normalized = []
    from langchain.schema import Document
    
    for d in docs:
        try:
            if isinstance(d, Document):
                normalized.append(d)
            elif isinstance(d, dict):
                text = d.get("page_content") or d.get("text") or d.get("content") or ""
                if text and text.strip():  # Boş content kontrolü
                    meta = d.get("metadata") or {}
                    normalized.append(Document(page_content=text.strip(), metadata=meta))
            else:
                text = str(d).strip()
                if text:
                    normalized.append(Document(page_content=text, metadata={"source": local_path}))
        except Exception as e:
            logging.error(f"Error normalizing document: {e}")
            continue
    
    logging.info(f"Normalized {len(normalized)} documents from {local_path}")
    
    # İçerik kontrolü
    for i, doc in enumerate(normalized[:3]):  # İlk 3 dökümanın preview'ını göster
        content_preview = doc.page_content[:100].replace('\n', ' ')
        logging.info(f"Doc {i+1} preview: {content_preview}...")
    
    return normalized


def index_uploaded_files(uploaded_files):
    global vectordb, qa_chain
    if not uploaded_files:
        return "Lütfen en az 1 dosya yükleyiniz."
    if not isinstance(uploaded_files, (list, tuple)):
        uploaded_files = [uploaded_files]


    docs_all = []
    temp_dirs = []
    for u in uploaded_files:
        local = ensure_local_path(u)
        if not local:
            continue
      
        if str(local).startswith(tempfile.gettempdir()):
            temp_dirs.append(os.path.dirname(local))
        docs = load_file_to_documents(local)
        docs_all.extend(docs)

    if not docs_all:
        return "Hiç belge içeriği çıkarılamadı. Desteklenen: PDF, TXT, MD, DOCX, PPTX, PNG, JPG, TIFF."

    try:
        
        if os.path.exists(VECTORSTORE_PERSIST_DIR):
            vectordb = Chroma(persist_directory=VECTORSTORE_PERSIST_DIR, embedding_function=embeddings)
            vectordb.add_documents(docs_all)
        else:
            vectordb = Chroma.from_documents(documents=docs_all, embedding=embeddings, persist_directory=VECTORSTORE_PERSIST_DIR)

        base_retriever = vectordb.as_retriever(search_kwargs={"k": 10})
        reranker_local = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-multilingual-v3.0", top_n=3)
        compression_retriever = ContextualCompressionRetriever(base_compressor=reranker_local, base_retriever=base_retriever)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=compression_retriever,
            chain_type="stuff",
            return_source_documents=True,
        )

     
        for d in temp_dirs:
            try:
                shutil.rmtree(d)
            except Exception:
                pass

        return f" Indexleme tamamlandı — {len(docs_all)} chunk eklendi."
    except Exception as e:
        logging.exception("index_uploaded_files hata")
        return f"Chroma indexleme hatası: {e}"


def serpapi_search(query: str, count: int = 3):
    try:
        params = {"q": query, "api_key": SERPAPI_API_KEY, "engine": "google", "num": count}
        search = GoogleSearch(params)
        results = search.get_dict()
        hits = []
        for item in results.get("organic_results", [])[:count]:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "") or ""
            hits.append(f"{title} - {link}\n{snippet}")
        return hits
    except Exception as e:
        logging.exception("serpapi_search hata")
        return []


def score_with_model(answer_text: str) -> int:
    
    if genai_client:
        try:
            prompt = f"Below is an answer. Rate it 0-10 for correctness and clarity. Only return an integer:\n\n{answer_text}\n\nScore:"
            resp = genai_client.client.models.generate(model=LLM_MODEL_NAME, input=prompt) if hasattr(genai_client, "client") else genai_client.models.generate(model=LLM_MODEL_NAME, input=prompt)
            txt = getattr(resp, "output_text", None) or getattr(resp, "text", None) or str(resp)
            digits = "".join(ch for ch in txt if ch.isdigit())
            if digits:
                n = int(digits[:2])
                return max(0, min(10, n))
        except Exception:
            logging.exception("score_with_model (genai) hata")
 
    return min(10, max(0, len(answer_text) // 150))


DEFAULT_SCORE_THRESHOLD = 5

def answer_question(question: str, use_web_if_missing: bool = True, score_threshold: int = DEFAULT_SCORE_THRESHOLD):
    global qa_chain
    if not question or not question.strip():
        return " Lütfen bir soru yazınız."

    if qa_chain is None:
        web_hits = serpapi_search(question, count=3)
        web_text = "\n\n".join(web_hits) if web_hits else ""
        if not web_text:
            return " Ne belgede ne de web'de uygun sonuç bulunamadı."
       
        prompt = f"Soru: {question}\n\nWeb sonuçları:\n{web_text}\n\nCevap kısa ve net olsun:"
        resp = llm.generate(prompt) if hasattr(llm, "generate") else llm.predict(prompt)
        answer_text = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
        score = score_with_model(answer_text)
        if score >= score_threshold:
            return f"Web fallback (puan {score}/10):\n\n{answer_text}"
        return f" Bulunan web yanıtı puanı düşük: {score}/10 — güvenilir cevap bulunamadı."

  
    try:
        result = qa_chain({"query": question})
        answer = result.get("result", "").strip()
        source_docs = result.get("source_documents", [])

        if answer and len(answer) > 30:
            score = score_with_model(answer)
            if score >= score_threshold:
             
                sources_text = ""
                for i, d in enumerate(source_docs[:3], start=1):
                    src = d.metadata.get("source", "Bilinmeyen")
                    sources_text += f"{i}. {os.path.basename(str(src))}\n"
                return f" Belge cevabı (puan {score}/10):\n\n{answer}\n\nKaynaklar:\n{sources_text}"
            else:
              
                if use_web_if_missing:
                    web_hits = serpapi_search(question, count=3)
                    web_text = "\n\n".join(web_hits) if web_hits else ""
                    if web_text:
                        prompt = f"Soru: {question}\n\nWeb sonuçları:\n{web_text}\n\nCevap kısa ve net olsun:"
                        resp = llm.generate(prompt) if hasattr(llm, "generate") else llm.predict(prompt)
                        web_answer = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
                        web_score = score_with_model(web_answer)
                        if web_score >= score_threshold:
                            return f"Web fallback (puan {web_score}/10):\n\n{web_answer}"
                    return f" Belgeden elde edilen cevap puanı düşük ({score}/10) ve güvenilir web fallback bulunamadı."
                return f" Belgeden elde edilen cevap puanı düşük ({score}/10)."
        else:
           
            if use_web_if_missing:
                web_hits = serpapi_search(question, count=3)
                web_text = "\n\n".join(web_hits) if web_hits else ""
                if not web_text:
                    return " Ne belgede ne de web'de uygun sonuç bulunamadı."
                prompt = f"Soru: {question}\n\nWeb sonuçları:\n{web_text}\n\nCevap kısa ve net olsun:"
                resp = llm.generate(prompt) if hasattr(llm, "generate") else llm.predict(prompt)
                web_answer = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
                web_score = score_with_model(web_answer)
                if web_score >= score_threshold:
                    return f" Web fallback (puan {web_score}/10):\n\n{web_answer}"
                return f" Web'den bulunan cevap puanı düşük ({web_score}/10)."
            return " Belgelerde yeterli cevap bulunamadı."

    except Exception as e:
        logging.exception("answer_question hata")
        return f"Hata: {e}"




try:
    from langchain.tools import StructuredTool
except Exception:
    try:
        from langchain_core.tools.structured import StructuredTool
    except Exception:
        StructuredTool = None

def qa_tool(query: str) -> str:
    """Answer a question using indexed documents (RetrievalQA). Returns a short answer string."""
    if qa_chain is None:
        return "Henüz döküman indexlenmedi."
    try:
    
        res = qa_chain({"query": query})
        return res.get("result", "") or "Belge tabanlı cevap bulunamadı."
    except Exception as e:
        logging.exception("qa_tool hata")
        return f"QA tool hata: {e}"


def serpapi_tool(query: str) -> str:
    """Perform a web search via SerpAPI and return top hits as a joined string."""
    try:
        hits = serpapi_search(query, count=3)
        return "\n\n".join(hits) if hits else "SerpAPI: sonuç bulunamadı."
    except Exception as e:
        logging.exception("serpapi_tool hata")
        return f"SerpAPI tool hata: {e}"



_agent_instance = None

def get_or_create_agent():
    """Create or return a cached LangGraph react agent. Tools are wrapped as StructuredTool if available."""
    global _agent_instance
    if _agent_instance is not None:
        return _agent_instance

    
    tools_for_agent = None
    if StructuredTool is not None:
        try:
            tools_for_agent = [
                StructuredTool.from_function(func=qa_tool,
                                             name="qa_tool",
                                             description="Answer questions from indexed documents."),
                StructuredTool.from_function(func=serpapi_tool,
                                             name="serpapi_tool",
                                             description="Search the web using SerpAPI and return top results."),
            ]
            logging.info("Using StructuredTool.from_function() for agent tools.")
        except Exception as e:
            logging.exception("StructuredTool.from_function failed, falling back to raw functions: %s", e)
            tools_for_agent = None

   
    if tools_for_agent is None:
        logging.warning("StructuredTool not available or failed — passing raw functions as tools (may fail depending on langgraph version).")
        tools_for_agent = [qa_tool, serpapi_tool]


    try:
        _agent_instance = create_react_agent(
            model=llm,
            tools=tools_for_agent,
            prompt=(
                "You are an assistant. Prefer document answers using qa_tool; "
                "if qa_tool doesn't have the answer, use serpapi_tool to search the web and answer."
            ),
        )
        logging.info("LangGraph react agent created.")
    except Exception as e:
        logging.exception("create_react_agent failed: %s", e)

        raise

    return _agent_instance


def ask_with_agent(question: str) -> str:
    """Ask the LangGraph agent a question and return a cleaned string answer."""
    if not question or not question.strip():
        return "⚠️ Lütfen bir soru yazın."

    try:
        agent = get_or_create_agent()
    except Exception as e:
        logging.exception("get_or_create_agent hata")
        return f"Agent oluşturulamadı: {e}"

    try:
    
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})

        if isinstance(result, dict):
           
            for key in ("messages", "response", "result", "output", "text"):
                if key in result:
                    msgs = result[key]
                    break
            else:
             
                return str(result)

           
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                if isinstance(last, dict):
                    return last.get("content") or last.get("text") or str(last)
                return str(last)

          
            if isinstance(msgs, dict):
                return msgs.get("content") or msgs.get("text") or str(msgs)
            return str(msgs)

        
        if isinstance(result, list) and result:
            last = result[-1]
            return str(last)
        return str(result)

    except Exception as e:
        logging.exception("ask_with_agent hata")
        return f"Agent hatası: {e}"



with gr.Blocks(title=" Doküman Q&A + Web (SerpAPI) +  LangGraph Agent") as demo:
    gr.Markdown("# Dokümanlara Soru Sorun + Web Fallback + Agent")
    gr.Markdown("Gemini 2.5 Flash + Docling (opsiyonel OCR) + Chroma + Cohere Rerank + SerpAPI + LangGraph Agent")

    with gr.Row():
        file_input = gr.File(label=" Dosyaları Seçin", file_count="multiple",
                             file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".txt", ".md", ".docx", ".pptx"])
        index_btn = gr.Button(" Indexle", variant="primary")

    status = gr.Textbox(label=" Durum", interactive=False, lines=4)

    gr.Markdown("---")

    with gr.Row():
        question = gr.Textbox(label=" Sorunuzu yazın", placeholder="Belgeler veya web hakkında soru...", lines=2)
        ask_btn = gr.Button(" Sorun (Belge + Web fallback)")
        ask_agent_btn = gr.Button(" Ask (Agent) — LangGraph")

    answer_box = gr.Textbox(label=" Cevap", lines=15, interactive=False)
    score_thresh = gr.Slider(minimum=0, maximum=10, value=5, step=1, label="Cevap için minimum puan eşiği (0-10)")

   
    index_btn.click(index_uploaded_files, inputs=[file_input], outputs=[status])
    ask_btn.click(lambda q, u, s: answer_question(q, use_web_if_missing=True, score_threshold=int(s)), inputs=[question, file_input, score_thresh], outputs=[answer_box])
    ask_agent_btn.click(ask_with_agent, inputs=[question], outputs=[answer_box])
    question.submit(lambda q, s: answer_question(q, use_web_if_missing=True, score_threshold=int(s)), inputs=[question, score_thresh], outputs=[answer_box])

if __name__ == "__main__":
    print(" Arayüz başlatılıyor — http://127.0.0.1:7860")
    demo.launch(share=False)
