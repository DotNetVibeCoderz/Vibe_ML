Name: LocalGen

Deskripsi:
Solusi Local AI Inference Engine baru berbasis .NET yang menggabungkan kekuatan Ollama, LocalAI, dan LM Studio, berikut adalah fiturnya:

---

 🔹 Core Engine Features
- Multi-model support: dukungan untuk LLaMA, Mistral, Gemma, Qwen, DeepSeek, HuggingFace models (format GGUF, safetensors).  
- Cross-platform runtime: build dengan .NET 10, Background Service + ASP.Net Core Min API, Avalonia untuk Admin Control dan Playground berbasis desktop multi-platform, CLI dengan Spectre Console.  
- OpenAI-compatible API: drop-in replacement agar aplikasi yang sudah pakai OpenAI API bisa langsung switch.  
- Semantic Kernel integration: native support untuk orchestration, planner, dan skill functions.  
- Vector database RAG: Microsoft.Extensions.VectorData dengan backend SQLite, Qdrant, Chroma, Azure AI Search.  

---

Fitur Admin Control
- Web Service Status: Start, Stop, Status, Logs
- Playground Chat: Menggunakan library semantic kernel, Select Model, Model Config (temperature, system prompt, dsb), Built-in Kernel Function yang bisa di checklist - defaultnya on semua (Math, Internet Search, Download, Web Scrap, Code Execution, Time and Date, File System), Skills (gallery - can search, download, use), MCP (gallery - can search and use), Dapat membuatkan aplikasi/contoh code dengan SDK yang dimiliki, berikan berbagai contoh prompt/query, terdapat session chat yang bisa di add/remove/clear
- Skills disini tidak hanya bisa baca instruksi, tapi menggunakan template/asset dan eksekusi script 
- Model Gallery - search, model info, download, serve from Hugging Face, Foundry Local, etc
- Engine - default: LlamaSharp (can select: CPU/GPU), Onnx Runtime, Foundry Local, other (give recommendation)
- About
- UI UX yang keren dan modern dengan theme dark/light dibuat dengan skill frontend-design 
---
 🔹 Developer-Oriented Features
- CLI tools: command-line untuk pull, run, benchmark model.  
- Modelfile configs: konfigurasi prompt, temperature, context length, embedding.  
- SDK .NET: library untuk console, web, desktop, IoT.  
- Plugin system: extensible untuk custom kernels, RAG connectors, dan agent frameworks.  
- Monitoring dashboard: real-time metrics (GPU usage, latency, throughput, token usage, etc).  

---

 🔹 User-Oriented Features
- Blazor UI: chat interface modern dengan dukungan light/dark theme.  
- Model browser: visual explorer untuk memilih model, quantization, ukuran.  
- Interactive charts: analitik penggunaan model, token, dan performa.  
- Multi-GPU support: tensor parallelism untuk model besar.  
- Offline mode: inference tanpa internet, cocok untuk edge deployment.  

---

 🔹 Advanced Features
- Agent orchestration: integrasi dengan workflow engine Agent Framework (multi-agent, task delegation).  
- File & image input: upload PDF, gambar, atau dokumen untuk RAG.  
- Web scraping: kernel function untuk ambil data dari URL.  
- Web Search: kernel function search internet dengan Tavily
- Math & date functions: built-in kernel untuk kalkulasi dan kalender.  
- Code Execution: memungkinkan running berbagai code bahasa pemrograman dan script seperti bash/powershell/python, termasuk menginstall sdk/runtime/dependensi yang diperlukan
- Deployment options: bisa jalan di laptop, server, atau container (Docker/Kubernetes).  

---

 🔹 Dokumentasi & Ekosistem
- README.md: panduan instalasi, konfigurasi, dan contoh kode dalam Bahasa inggris dan Bahasa Indonesia.  
- Dokumentasi lengkap di folder docs termasuk cara instalasi, penggunaan aplikasi, SDK, project template, code sample
- Sample apps: gallery dengan Blazor, WPF, Avalonia, dan console.  
- VSCode extensions: integrasi untuk debugging dan model management.  
- Template projects: scaffolding untuk web, desktop, dan API.  
- Tambahkan info di aplikasi dan dokumentasi: Dibuat oleh Gravicode Studios dipimpin oleh Kang Fadhil
- PLAN.md untuk roadmap pengembangan, Progress.md untuk tracking development