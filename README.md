# 🤖 Generative AI with LangChain

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)](CONTRIBUTING.md)

> 🚀 **A comprehensive collection of LangChain examples, tutorials, and implementations for building powerful AI applications**

## 📖 Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ Repository Structure](#️-repository-structure)
- [🚀 Quick Start](#-quick-start)
- [📚 Learning Modules](#-learning-modules)
- [🛠️ Installation](#️-installation)
- [💡 Examples](#-examples)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Overview

This repository is your complete guide to **LangChain** - a powerful framework for developing applications with large language models (LLMs). Whether you're a beginner or an advanced developer, you'll find practical examples, comprehensive tutorials, and real-world implementations.

### What You'll Learn:
- 🧠 **LLM Integration** - Connect with various language models
- ⛓️ **Chain Building** - Create complex AI workflows
- 📄 **Document Processing** - Load and process various document types
- 🎯 **Output Parsing** - Structure AI responses effectively
- 🔧 **Tools & Agents** - Build autonomous AI systems
- 🗄️ **Vector Stores** - Implement semantic search
- 🤖 **RAG Systems** - Retrieval Augmented Generation

## 🏗️ Repository Structure

```
📁 langchain/
├── 📁 langchain-models/           # LLM, Chat, and Embedding Models
│   ├── 🤖 1.LLMs/                # Basic language models
│   ├── 💬 2.ChatModels/           # Conversational AI models
│   ├── 🔤 3.EmbeddingModels/      # Text embedding models
│   └── 🖼️ image_models/          # Image generation models
├── ⛓️ langchain_chains/           # Chain Implementations
│   ├── simple_chains.py          # Basic chain examples
│   ├── sequential_chain.py       # Sequential processing
│   ├── parallel_chain.py         # Parallel execution
│   └── conditional_chain.py      # Conditional logic
├── 📄 langchain_document_loader/  # Document Processing
│   ├── pdf_loader.py             # PDF document handling
│   ├── csv_loader.py             # CSV data processing
│   ├── text_loader.py            # Text file loading
│   └── Web_Base_Loader.py        # Web content scraping
├── 🎯 langchain_output_parser/    # Response Formatting
│   ├── pydantic_output_parser.py # Structured outputs
│   ├── jsonoutputparser.py       # JSON formatting
│   └── structured_output_parser.py
├── 💬 langchain_prompts/          # Prompt Engineering
│   ├── static_prompt.py          # Fixed prompts
│   ├── dynamic_prompt.py         # Dynamic content
│   ├── chatbot.py                # Conversational flows
│   └── chat_prompt_template.py   # Template management
├── 🔧 langchain_runnables/        # Advanced Workflows
│   ├── runnable_sequence.py      # Sequential execution
│   ├── runnable_parallel.py      # Parallel processing
│   ├── Runnable_Branch.py        # Conditional branching
│   └── Runnable_Lambda.py        # Custom functions
├── 📊 langchain_structured_output/ # Data Structures
│   ├── pydantic_demo.py          # Type-safe outputs
│   ├── typedict_demo.py          # Dictionary types
│   └── with_structured_output_*.py
├── ✂️ langchain_text_splitters/   # Text Processing
│   ├── length_based.py           # Character/token splitting
│   ├── semantic_meaning_based.py # Semantic chunking
│   └── Document_Structured_based.py
├── 🛠️ langchain_tools/            # AI Tools & Agents
├── 🔍 Langchain-Retrivers/        # Information Retrieval
├── 🗄️ vector_stores/              # Vector Databases
│   ├── langchain_chroma_db.ipynb # ChromaDB integration
│   └── langchain_chroma_FAISS.ipynb # FAISS implementation
├── 🧠 RAG/                        # Retrieval Augmented Generation
│   ├── RAG.ipynb                 # Complete RAG pipeline
│   └── README.md                 # RAG concepts explained
└── 🎥 Youtube_rag_chatbot/        # Real-world Application
    ├── app.py                    # Streamlit application
    ├── rag_using_langchain.ipynb # Implementation notebook
    └── requirements.txt          # Dependencies
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Generative-AI.git
cd Generative-AI
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv langchain_env
langchain_env\Scripts\activate  # Windows
# source langchain_env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r langchain/langchain-models/requirements.txt
```

### 3. Run Your First Example
```bash
cd langchain/langchain_chains
python simple_chains.py
```

## 📚 Learning Modules

### 🤖 **Module 1: Language Models**
Start your journey by understanding different types of models and how to integrate them.

<details>
<summary>📖 What you'll learn</summary>

- **LLMs**: Basic language model integration
- **Chat Models**: Conversational AI with various providers (OpenAI, Anthropic, Google, Ollama)
- **Embedding Models**: Vector representations for semantic search
- **Model Initialization**: Best practices for model setup

**🎯 Key Files:**
- `langchain-models/1.LLMs/llm_demo.py`
- `langchain-models/2.ChatModels/chatmodel_openai.py`
- `langchain-models/3.EmbeddingModels/`

</details>

### ⛓️ **Module 2: Chains**
Learn to build complex AI workflows by chaining components together.

<details>
<summary>📖 What you'll learn</summary>

- **Simple Chains**: Basic input-output processing
- **Sequential Chains**: Step-by-step processing pipelines
- **Parallel Chains**: Concurrent execution for efficiency
- **Conditional Chains**: Logic-based routing

**🎯 Key Files:**
- `langchain_chains/simple_chains.py`
- `langchain_chains/sequential_chain.py`
- `langchain_chains/parallel_chain.py`

</details>

### 📄 **Module 3: Document Processing**
Master the art of loading and processing various document types.

<details>
<summary>📖 What you'll learn</summary>

- **PDF Processing**: Extract text from PDF documents
- **CSV Handling**: Work with structured data
- **Web Scraping**: Load content from websites
- **Directory Processing**: Batch process multiple files

**🎯 Key Files:**
- `langchain_document_loader/pdf_loader.py`
- `langchain_document_loader/csv_loader.py`
- `langchain_document_loader/Web_Base_Loader.py`

</details>

### 🎯 **Module 4: Output Parsing**
Structure and format AI responses for your applications.

<details>
<summary>📖 What you'll learn</summary>

- **Pydantic Parsing**: Type-safe structured outputs
- **JSON Formatting**: Well-formatted JSON responses
- **String Parsing**: Custom string manipulation
- **Structured Outputs**: Complex data structure handling

**🎯 Key Files:**
- `langchain_output_parser/pydantic_output_parser.py`
- `langchain_output_parser/jsonoutputparser.py`

</details>

### 💬 **Module 5: Prompt Engineering**
Craft effective prompts for optimal AI performance.

<details>
<summary>📖 What you'll learn</summary>

- **Template Design**: Reusable prompt templates
- **Dynamic Prompts**: Context-aware prompt generation
- **Chat History**: Maintaining conversation context
- **Message Management**: Handling complex conversations

**🎯 Key Files:**
- `langchain_prompts/static_prompt.py`
- `langchain_prompts/dynamic_prompt.py`
- `langchain_prompts/chatbot.py`

</details>

### 🔧 **Module 6: Advanced Runnables**
Build sophisticated AI workflows with LangChain's runnable interface.

<details>
<summary>📖 What you'll learn</summary>

- **Runnable Sequences**: Chain components efficiently
- **Parallel Execution**: Run multiple processes simultaneously
- **Branching Logic**: Conditional workflow routing
- **Custom Functions**: Integrate custom Python functions

**🎯 Key Files:**
- `langchain_runnables/runnable_sequence.py`
- `langchain_runnables/runnable_parallel.py`
- `langchain_runnables/Runnable_Branch.py`

</details>

### 🧠 **Module 7: RAG (Retrieval Augmented Generation)**
Build powerful question-answering systems that can access external knowledge.

<details>
<summary>📖 What you'll learn</summary>

- **RAG Fundamentals**: Understanding retrieval-augmented generation
- **Vector Stores**: ChromaDB and FAISS integration
- **Text Splitting**: Optimal chunking strategies
- **Semantic Search**: Finding relevant information

**🎯 Key Files:**
- `RAG/RAG.ipynb`
- `vector_stores/langchain_chroma_db.ipynb`
- `langchain_text_splitters/semantic_meaning_based.py`

</details>

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Core Dependencies
```bash
pip install langchain
pip install langchain-openai
pip install langchain-anthropic
pip install langchain-google-genai
pip install chromadb
pip install faiss-cpu
pip install streamlit
```

### API Keys Setup
Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

## 💡 Examples

### 🚀 Quick Start Examples

#### Simple Chat with OpenAI
```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Initialize the model
llm = ChatOpenAI(temperature=0.7)

# Send a message
response = llm.invoke([HumanMessage(content="Hello, how are you?")])
print(response.content)
```

#### Basic RAG Pipeline
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load and split documents
loader = TextLoader("your_document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# Query the knowledge base
query = "What is this document about?"
results = vectorstore.similarity_search(query)
```

### 🎯 Interactive Demos

| Demo | Description | Launch |
|------|-------------|---------|
| 🎥 **YouTube RAG Chatbot** | Chat with YouTube video content | [`app.py`](Youtube_rag_chatbot/app.py) |
| 📊 **Document Q&A** | Ask questions about your documents | [`RAG.ipynb`](RAG/RAG.ipynb) |
| 🤖 **Multi-Model Chat** | Compare different AI models | [`groq_+_gemini_2_5_pro.ipynb`](langchain-models/2.ChatModels/groq_+_gemini_2_5_pro.ipynb) |

## 🎓 Learning Path

### 👶 **Beginner Track** (Start Here!)
1. 📖 Read [RAG/README.md](RAG/README.md) - Understand LLM fundamentals
2. 🤖 Explore `langchain-models/` - Learn model integration
3. ⛓️ Try `langchain_chains/simple_chains.py` - Basic chains
4. 💬 Practice with `langchain_prompts/static_prompt.py` - Prompt basics

### 🏃 **Intermediate Track**
1. 📄 Work through `langchain_document_loader/` - Document processing
2. 🎯 Master `langchain_output_parser/` - Response formatting
3. ✂️ Study `langchain_text_splitters/` - Text chunking strategies
4. 🔧 Build with `langchain_runnables/` - Advanced workflows

### 🚀 **Advanced Track**
1. 🗄️ Implement `vector_stores/` - Vector databases
2. 🧠 Complete `RAG/RAG.ipynb` - Full RAG pipeline
3. 🎥 Deploy `Youtube_rag_chatbot/` - Production application
4. 🛠️ Explore `langchain_tools/` - AI agents and tools

## 🛠️ Tools & Technologies

| Category | Tools Used |
|----------|------------|
| **Language Models** | OpenAI GPT, Anthropic Claude, Google Gemini, Ollama (Llama3, DeepSeek) |
| **Vector Stores** | ChromaDB, FAISS |
| **Document Types** | PDF, CSV, TXT, Web pages |
| **Frameworks** | LangChain, Streamlit |
| **Embeddings** | OpenAI Embeddings, HuggingFace |

## 🎯 Use Cases

### 🏢 **Business Applications**
- 📋 **Document Q&A**: Query internal documents and knowledge bases
- 🎧 **Customer Support**: Automated help desk with RAG
- 📊 **Data Analysis**: Natural language queries on CSV/database data
- 📝 **Content Generation**: Automated report and content creation

### 🎓 **Educational Projects**
- 📚 **Study Assistant**: Chat with textbooks and lecture notes
- 🧑‍💻 **Code Explainer**: Understand complex codebases
- 🌍 **Language Learning**: Interactive conversation practice
- 📖 **Research Helper**: Summarize and analyze research papers

### 🔬 **Research & Development**
- 🧪 **Experiment with Models**: Compare different LLM capabilities
- 🔍 **Retrieval Systems**: Test various chunking and embedding strategies
- ⚡ **Performance Optimization**: Benchmark different vector stores
- 🎛️ **Prompt Engineering**: Develop optimal prompting techniques

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 📝 **Ways to Contribute**
- 🐛 **Bug Reports**: Found an issue? Let us know!
- ✨ **Feature Requests**: Have an idea? We'd love to hear it!
- 📖 **Documentation**: Help improve our guides and examples
- 💻 **Code**: Submit bug fixes or new features
- 🎓 **Tutorials**: Share your learning journey

### 🚀 **Getting Started**
1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔄 Open a Pull Request

### 📋 **Contribution Guidelines**
- ✅ Follow Python PEP 8 style guidelines
- 📚 Add docstrings to new functions and classes
- 🧪 Include tests for new features
- 📖 Update documentation for any changes
- 🏷️ Use clear, descriptive commit messages

## 📚 Additional Resources

### 📖 **Documentation**
- [LangChain Official Docs](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### 🎥 **Tutorials & Guides**
- [LangChain Crash Course](https://python.langchain.com/docs/get_started)
- [RAG Implementation Guide](RAG/README.md)
- [Vector Store Comparison](vector_stores/)

### 🌟 **Community**
- [LangChain Discord](https://discord.gg/langchain)
- [GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/langchain)

## 📊 Project Stats

```
📁 Total Files: 50+
🐍 Python Scripts: 40+
📓 Jupyter Notebooks: 10+
📖 Documentation: 5+
🎯 Examples: 30+
```

## 🏷️ Tags

`langchain` `llm` `ai` `machine-learning` `nlp` `rag` `vector-database` `openai` `anthropic` `chatbot` `document-processing` `embeddings` `python` `generative-ai`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🦜 [LangChain Team](https://www.langchain.com/) for the amazing framework
- 🤖 [OpenAI](https://openai.com/) for GPT models
- 🧠 [Anthropic](https://www.anthropic.com/) for Claude models
- 🔍 [Chroma](https://www.trychroma.com/) for vector database solutions
- 🌟 All contributors who helped build this repository

---

<div align="center">

### 🚀 Ready to build amazing AI applications?

**[⭐ Star this repo](../../stargazers)** • **[🍴 Fork it](../../fork)** • **[📝 Contribute](../../issues)**

Made with ❤️ by the Adarsha Rimal

</div>
