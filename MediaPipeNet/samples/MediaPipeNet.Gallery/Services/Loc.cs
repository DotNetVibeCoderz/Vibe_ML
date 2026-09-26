namespace MediaPipeNet.Gallery.Services;

/// <summary>Minimal English / Bahasa Indonesia string table for the Gallery UI.</summary>
public static class Loc
{
    private static readonly Dictionary<string, (string En, string Id)> Strings = new()
    {
        ["nav.home"] = ("Overview", "Ikhtisar"),
        ["nav.detect"] = ("DETECT", "DETEKSI"),
        ["nav.landmarks"] = ("LANDMARKS", "LANDMARK"),
        ["nav.understand"] = ("SEGMENT & CLASSIFY", "SEGMENTASI & KLASIFIKASI"),
        ["nav.pipelines"] = ("PIPELINES", "PIPELINE"),
        ["nav.app"] = ("APP", "APLIKASI"),
        ["nav.live"] = ("Live camera", "Kamera langsung"),
        ["nav.graph"] = ("Graph API", "Graph API"),
        ["nav.benchmark"] = ("Benchmark", "Benchmark"),
        ["nav.models"] = ("Models", "Model"),
        ["nav.settings"] = ("Settings", "Pengaturan"),
        ["nav.about"] = ("About", "Tentang"),

        ["home.eyebrow"] = ("MEDIAPIPE.NET GALLERY", "GALERI MEDIAPIPE.NET"),
        ["home.title"] = ("See what the model sees.", "Lihat apa yang dilihat model."),
        ["home.lede"] = (
            "Google MediaPipe's perception models, running natively in .NET on ONNX Runtime. Pick a task, drop in an image or open your camera, tune the options, then copy the C# that produced the result.",
            "Model persepsi Google MediaPipe, berjalan native di .NET di atas ONNX Runtime. Pilih task, masukkan gambar atau buka kamera, atur opsinya, lalu salin kode C# yang menghasilkan hasil tersebut."),
        ["home.tasks"] = ("TASKS", "TASK"),
        ["home.more"] = ("TOOLS", "ALAT"),

        ["task.samples"] = ("Samples", "Contoh"),
        ["task.open"] = ("Open image…", "Buka gambar…"),
        ["task.run"] = ("Run again", "Jalankan ulang"),
        ["task.options"] = ("OPTIONS", "OPSI"),
        ["task.results"] = ("RESULTS", "HASIL"),
        ["task.preview"] = ("Preview", "Pratinjau"),
        ["task.code"] = ("C# code", "Kode C#"),
        ["task.json"] = ("JSON", "JSON"),
        ["task.copy"] = ("Copy", "Salin"),
        ["task.copied"] = ("Copied", "Tersalin"),
        ["task.running"] = ("Running…", "Memproses…"),
        ["task.nothing"] = ("Nothing found in this image. Try another sample or lower the confidence.", "Tidak ada yang terdeteksi. Coba contoh lain atau turunkan confidence."),
        ["task.error"] = ("The task failed: ", "Task gagal: "),

        ["live.title"] = ("Live camera", "Kamera langsung"),
        ["live.lede"] = ("Runs the selected task on your webcam in video mode: tracking between frames, and frames that arrive while the model is busy are skipped so latency stays low.",
            "Menjalankan task terpilih pada webcam dalam mode video: tracking antar-frame, dan frame yang datang saat model sibuk dilewati agar latensi tetap rendah."),
        ["live.start"] = ("Start camera", "Mulai kamera"),
        ["live.stop"] = ("Stop", "Berhenti"),
        ["live.task"] = ("Task", "Task"),
        ["live.camera"] = ("Camera", "Kamera"),
        ["live.mirror"] = ("Mirror (selfie view)", "Cermin (tampilan selfie)"),
        ["live.nocamera"] = ("No camera found. Connect a webcam, then press Start camera.", "Kamera tidak ditemukan. Sambungkan webcam, lalu tekan Mulai kamera."),
        ["live.idle"] = ("Camera is off. Press Start camera.", "Kamera mati. Tekan Mulai kamera."),

        ["graph.title"] = ("Graph API", "Graph API"),
        ["graph.lede"] = ("Compose your own pipeline from calculator nodes. This graph fans one image out to a face detector and a hand landmarker running in parallel, joins their results per timestamp, and pixelates faces with a custom node.",
            "Susun pipeline sendiri dari node kalkulator. Graph ini membagi satu gambar ke face detector dan hand landmarker yang berjalan paralel, menggabungkan hasilnya per timestamp, lalu memburamkan wajah dengan node kustom."),
        ["graph.run"] = ("Run graph", "Jalankan graph"),
        ["graph.pbtxt"] = ("Graph config (.pbtxt)", "Konfigurasi graph (.pbtxt)"),
        ["graph.runpbtxt"] = ("Run config", "Jalankan konfigurasi"),

        ["bench.title"] = ("Benchmark", "Benchmark"),
        ["bench.lede"] = ("Measures end-to-end latency of every task on this machine (pre-processing, inference and post-processing) at 640×480.",
            "Mengukur latensi end-to-end setiap task di mesin ini (pre-processing, inferensi, dan post-processing) pada 640×480."),
        ["bench.run"] = ("Run benchmark", "Jalankan benchmark"),
        ["bench.iterations"] = ("Iterations", "Iterasi"),
        ["bench.target"] = ("NFR-1 target: face detection under 50 ms", "Target NFR-1: deteksi wajah di bawah 50 ms"),

        ["models.title"] = ("Models", "Model"),
        ["models.lede"] = ("Every model was converted from the official MediaPipe TFLite release and is checked against its SHA-256 before use.",
            "Setiap model dikonversi dari rilis TFLite resmi MediaPipe dan dicek SHA-256-nya sebelum dipakai."),
        ["models.verify"] = ("Verify checksums", "Verifikasi checksum"),
        ["models.download"] = ("Download missing", "Unduh yang belum ada"),
        ["models.local"] = ("available", "tersedia"),
        ["models.missing"] = ("missing", "belum ada"),

        ["settings.title"] = ("Settings", "Pengaturan"),
        ["settings.provider"] = ("Execution provider", "Execution provider"),
        ["settings.provider.help"] = ("Auto tries CUDA, then DirectML, CoreML and finally CPU.", "Auto mencoba CUDA, lalu DirectML, CoreML, dan terakhir CPU."),
        ["settings.threads"] = ("CPU threads per model (0 = automatic)", "Thread CPU per model (0 = otomatis)"),
        ["settings.language"] = ("Language", "Bahasa"),
        ["settings.theme"] = ("Theme", "Tema"),
        ["settings.modeldir"] = ("Extra model folder (optional)", "Folder model tambahan (opsional)"),
        ["settings.points"] = ("Draw landmark points", "Gambar titik landmark"),
        ["settings.save"] = ("Save settings", "Simpan pengaturan"),
        ["settings.saved"] = ("Settings saved. Tasks reload with the new configuration.", "Pengaturan disimpan. Task dimuat ulang dengan konfigurasi baru."),
        ["settings.runtime"] = ("RUNTIME", "RUNTIME"),

        ["about.title"] = ("About", "Tentang"),
        ["footer.credit"] = ("Created by Gravicode Studios · led by Kang Fadhil", "Dibuat oleh Gravicode Studios · dipimpin Kang Fadhil"),
    };

    /// <summary>Current language code: "en" or "id".</summary>
    public static string Language => AppSettings.Current.Language;

    public static bool IsIndonesian => Language == "id";

    /// <summary>Looks up a UI string.</summary>
    public static string T(string key) =>
        Strings.TryGetValue(key, out var s) ? (IsIndonesian ? s.Id : s.En) : key;

    /// <summary>Picks between an English and an Indonesian text.</summary>
    public static string Pick(string en, string id) => IsIndonesian ? id : en;
}
