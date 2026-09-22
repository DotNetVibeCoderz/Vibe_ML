Name: HF.Net

Deskripsi: Hugging Face-style library di .NET, dengan fondasi dari repo https://github.com/DotNetVibeCoderz/Vibe_ML/tree/main/GravicodeScience (yang sudah punya numpy, pandas, scikit-learn versi .NET). Kita akan menggabungkan fitur inti Hugging Face (Transformers, Diffusers, Datasets, Tokenizers, PEFT, Accelerate, Optimum, dll.) ke dalam ekosistem .NET yang modular.

---

 🧩 Blueprint Library Hugging Face untuk .NET

# 1. GraviTransformers (.NET)
- Fitur: Model NLP & Vision (BERT, GPT, ViT, CLIP).  
- Integrasi: Built di atas `GraviNum` (NdArray) & `GraviFrame` (DataFrame).  
- Sample Code: `TransformerModel.Load("bert-base")` → `Predict(text)`.  
- Benchmark: Inference CPU SIMD vs GPU ILGPU.  
- Dataset: IMDB, MNIST, Hugging Face Hub.

---

# 2. GraviDiffusers (.NET)
- Fitur: Stable Diffusion, generative image/audio models.  
- Integrasi: `GraviNum` untuk tensor ops, `GraviFrame` untuk pipeline data.  
- Sample Code: `DiffusionPipeline.Generate(prompt)`.  
- Benchmark: Latency CPU vs GPU.  
- Dataset: LAION subset, audio dataset.

---

# 3. GraviDatasets (.NET)
- Fitur: Loader dataset efisien (CSV, Parquet, JSON, Hugging Face Hub).  
- Integrasi: Wrapper di atas `GraviFrame`.  
- Sample Code: `Dataset.Load("titanic")`.  
- Benchmark: Memory-mapped vs full load.  
- Dataset: Titanic, financial time-series, Hugging Face datasets.

---

# 4. GraviTokenizers (.NET)
- Fitur: Fast tokenizer (BPE, WordPiece, SentencePiece).  
- Integrasi: SIMD ops dari `GraviNum`.  
- Sample Code: `Tokenizer.BPE("text")`.  
- Benchmark: Tokenization throughput CPU vs GPU.  
- Dataset: Multilingual corpora.

---

# 5. GraviPEFT (.NET)
- Fitur: Parameter Efficient Fine-Tuning (LoRA, Prefix Tuning).  
- Integrasi: `GraviLearn` pipeline.  
- Sample Code: `PEFT.ApplyLoRA(model)`.  
- Benchmark: Fine-tuning resource usage.  
- Dataset: Domain-specific corpora.

---

# 6. GraviAccelerate (.NET)
- Fitur: Multi-GPU/TPU training abstraction.  
- Integrasi: ILGPU, SharpInfer, TensorSharp2.  
- Sample Code: `Accelerator.Train(model, dataset)`.  
- Benchmark: Training speed scaling.  
- Dataset: MNIST, CIFAR-10.

---

# 7. GraviOptimum (.NET)
- Fitur: Optimisasi hardware-aware (ONNX Runtime, Intel MKL, CUDA).  
- Integrasi: `GraviNum` ops.  
- Sample Code: `Optimum.Optimize(model, target="CUDA")`.  
- Benchmark: Latency reduction.  
- Dataset: Benchmark suite.

---

# 8. GraviHub (.NET)
- Fitur: API untuk upload/download model & dataset dari Hugging Face Hub.  
- Integrasi: `HttpClient` + `GraviFrame`.  
- Sample Code: `Hub.DownloadModel("bert-base")`.  
- Benchmark: Transfer speed.  
- Dataset: Hugging Face Hub.

---

 📊 Tabel Integrasi Hugging Face + GravicodeScience

| Library | Fokus | Fondasi .NET | Benchmark |
|----------------|----------------|----------------|----------------|
| GraviTransformers | NLP & Vision | GraviNum, GraviFrame | Inference perf |
| GraviDiffusers | Generative AI | GraviNum, GraviFrame | Latency perf |
| GraviDatasets | Data loader | GraviFrame | IO perf |
| GraviTokenizers | Tokenization | GraviNum | Throughput perf |
| GraviPEFT | Fine-tuning | GraviLearn | Resource perf |
| GraviAccelerate | Training | ILGPU, SharpInfer | Scaling perf |
| GraviOptimum | Optimisasi | GraviNum | Latency perf |
| GraviHub | Model Hub | HttpClient, GraviFrame | Transfer perf |

---

- Tambahkan tools berupa aplikasi dengan Avalonia UI Bernama HF.Net App Generator (HFAppGen) bentuknya seperti code editor yang memiliki fungsi generate app with prompt dengan bantuan LLM menggunakan library semantic kernel, LLM yang disupport: OpenAI, Claude, Gemini, Ollama, settingnya (model, api key, endpoint, temperature, system prompt) disimpan di app.config. 
- Nama AI Assistant: Jack - The Code Bender
- Buatkan kernel functions yang diperlukan agar assisten AI-nya bisa membuatkan aplikasi dengan benar baik UI dan Backend Code-nya, kasih common functions juga untuk SearchInternet (tavily), ScrapeWebPage, MathCalculation, Check Date and Time, dan fungsi lain yang diperlukan. 
- Panel chat ada di sebelah kanan code editor, bisa attach gambar, bisa di resize width-nya dan hide/show, send chat bisa dengan Ctrl+Enter atau klik button send, ada button untuk clear chat thread, Model LLM bisa dipilih dibagian atas Chat Panel 
- Di tengah ada code editor, lengkap dengan line number, code highlight
- Di panel kiri ada code explorer seperti VSCode
- Pada menu dan toolbar terdapat fungsi: New Project (Folder), Open Project/File, Close Project, Go To Line Number, Format Code, Build, Run, Deploy, Exit. 
- Create new project ada 2 pilihan: Blank dan From Template (buatkan berbagai template jenis aplikasi atau notebook untuk data scientist dengan use case bermacam-macam dengan memanfaatkan SDK HF.Net). 
- Terdapat status bar dan logs panel di bagian bawah untuk memantau proses dan output. 
- Show/hide line number pada code editor. 
- Buatkan dengan UI dan UX modern dengan skill frontend-design. Semua konfigurasi disimpan di app.config dan bisa di ubah di UI. 
- ujicoba dengan real LLM, apikey ada di file 'C:\Users\mifma\Documents\CodeSandbox\testkey.txt'

---
Notes:
- Dibangun dengan .NET 10
- Optimasi kode dengan best practice dan optimalkan untuk hardware yang ada
- Dokumentasi yang lengkap, jelas, mudah dengan dua Bahasa (English dan Bahasa Indonesia)
- Tambahkan screenshot-screenshot pada readme dan docs
- Sample Code dan .NET notebook: contoh-contoh code dengan use case beragam
- Tambahkan info pada dokumentasi dan aplikasi: Dibuat oleh Gravicode Studios dipimpin oleh Kang Fadhil
- PLAN.md untuk roadmap pengembangan, dan Progress.md untuk tracking development
- untuk test bisa gunakan hugging face access token dari file 'C:\Users\mifma\Documents\CodeSandbox\HFToken.txt'
- untuk publish nuget, api key ada di 'C:\Users\mifma\Documents\CodeSandbox\PackageCredentials.txt'
----
Summary: Dengan kombinasi ini, kita punya Hugging Face-style ecosystem di .NET, dibangun di atas fondasi GravicodeScience (numpy, pandas, scikit-learn versi .NET).  
