Chat-Docs-Agent



🇹🇷 Türkçe Açıklama
Proje Hakkında

Chat-Docs-Agent, belgelerinizden (PDF, Word, PowerPoint, TXT vb.) içerik çıkarıp bu belgelerle soru-cevap yapabilen ve gerektiğinde web üzerinden ek bilgi araması yapabilen bir yapay zeka destekli asistan projesidir.

Bu proje sayesinde:

Kendi PDF veya dokümanlarınızı yükleyebilir,

Belgeler üzerinde akıllı arama yapabilir,

SerpAPI entegrasyonu sayesinde internette ek araştırmalar yapılabilir,

Kullanıcı dostu bir Gradio arayüzü üzerinden kolayca kullanılabilir.

 Kullanılan Teknolojiler

LangChain → Belge işleme, zincir oluşturma ve LLM entegrasyonu için

LangGraph → Agent (düşünen ve araç kullanan yapay zeka) için

Gradio → Basit ve kullanıcı dostu web arayüzü için

Docling → PDF ve diğer dokümanlardan içerik çıkarma için

Sentence-Transformers (MiniLM) → Belge vektörleştirme ve semantik arama için

ChromaDB → Belgeleri indeksleyip hızlı arama yapabilmek için

SerpAPI → Google aramaları üzerinden web’den ek bilgi çekmek için

 Özellikler

✅ PDF, TXT, DOCX, PPTX, PNG, JPG gibi formatlardan içerik çıkarma


✅ Belgeler üzerinde doğal dil ile arama yapma


✅ Web’den gerçek zamanlı bilgi araması yapabilme


✅ Gradio ile kolay kullanım sağlayan arayüz


✅ OpenAI / Google Gemini gibi LLM entegrasyonlarına uygun yapı



 Çalıştırma aşaması

 
Bağımlılıkları yükleyiniz
pip install -r requirements.txt

Uygulamayı başlatınız
python app.py


Uygulama çalıştıktan sonra arayüz → http://127.0.0.1:7860 adresinde açılacaktır.

🇬🇧 English Description
 About the Project

Chat-Docs-Agent is an AI-powered assistant that extracts content from your documents (PDF, Word, PowerPoint, TXT, etc.) and allows you to ask questions based on them. If the document does not contain the answer, the assistant can perform web searches to find additional information.

With this project you can:

Upload your own documents,

Perform intelligent searches within them,

Use SerpAPI to fetch complementary information from the web,

Interact easily through a Gradio-based web interface.

 Technologies Used

LangChain → For document processing, chains, and LLM integration

LangGraph → For building intelligent agents

Gradio → For a simple and user-friendly web interface

Docling → For extracting content from PDFs and other documents

Sentence-Transformers (MiniLM) → For document embeddings and semantic search

ChromaDB → For indexing documents and enabling fast retrieval

SerpAPI → For fetching real-time information from Google Search

 Features

✅ Supports PDF, TXT, DOCX, PPTX, PNG, JPG extraction


✅ Natural language Q&A over your documents


✅ Real-time web search integration


✅ User-friendly interface powered by Gradio


✅ Ready for OpenAI / Google Gemini LLM integrations



🔹 How to Run
Install dependencies
pip install -r requirements.txt

Start the app
python app.py


Once started, the interface will be available at → http://127.0.0.1:7860
