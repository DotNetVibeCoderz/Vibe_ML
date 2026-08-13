# GraviGraph

*[English](../GraviGraph.md)* · Graph ML untuk .NET — struktur, algoritma, embedding node, dan GNN.

## Struktur graf

Representasi utamanya adalah adjacency list, karena pekerjaan graf hampir selalu bersifat lokal di
sekitar tetangga: PageRank, BFS, dan message passing semuanya mengiterasi tetangga sebuah node,
yang berbiaya `O(derajat)` di sini dan `O(node)` pada matriks dense. Jaringan nyata bersifat jarang
— Cora punya 2.708 node dan 5.429 edge, sehingga matriks dense-nya akan 99,9% nol.

```csharp
using Gravicode.Science.GraviGraph;

var g = new Graph(directed: false);
g.AddNode("paper-1", label: 0);
g.AddEdge(0, 1, weight: 2.5);

g.NodeCount;  g.EdgeCount;  g.Density;
g.Neighbors(0);  g.Predecessors(0);
g.Degree(0);  g.InDegree(0);  g.OutDegree(0);  g.WeightedDegree(0);
g.HasEdge(0, 1);  g.Edges();  g.Degrees();

g.NodeFeatures = matrix;      // (node, fitur)
g.NodeLabels;  g.NodeNames;  g.Classes;
```

### Bentuk matriks

```csharp
g.ToSparseAdjacency();                                            // CSR
g.ToSparseAdjacency(addSelfLoops: true, symmetricNormalize: true); // matriks propagasi GCN
g.ToDenseAdjacency();                                             // dijaga — melempar di atas ~50 juta entri
g.Laplacian();                                                    // D - A
```

### Transformasi dan generator

```csharp
g.Subgraph([1, 5, 9]);        // terinduksi, dinomori ulang dari nol
g.AsUndirected();             // setiap edge bisa ditelusuri dua arah

Graph.Random(nodes: 100, p: 0.05, seed: 42);
Graph.ScaleFree(nodes: 1000, edgesPerNode: 3);   // preferential attachment — derajat power-law
Graph.Communities(communities: 3, sizePerCommunity: 50, internalP: 0.4, externalP: 0.01);
Graph.Cycle(10);  Graph.Complete(10);
```

### Memuat dan menyimpan

```csharp
var cora = Graph.Load("datasets/cora_graph.json");
g.Save("graph.json", name: "milik-saya", description: "...");
Graph.LoadEdgeList("edges.txt", directed: false);
```

Fitur node disimpan sebagai **indeks elemen tak-nol**. Fitur Cora berdimensi 1.433 tetapi rata-rata
hanya sekitar delapan belas yang tak-nol, sehingga bentuk sparse-nya kira-kira delapan puluh kali
lebih kecil dan proporsional lebih cepat dimuat.

## Algoritma

```csharp
using Gravicode.Science.GraviGraph.Algorithms;

GraphAlgorithms.BreadthFirstSearch(g, start);
GraphAlgorithms.DepthFirstSearch(g, start);
GraphAlgorithms.HopDistances(g, start);              // -1 bila tak terjangkau

GraphAlgorithms.ShortestPaths(g, start);             // Dijkstra, binary heap
GraphAlgorithms.ShortestPath(g, start, end);

GraphAlgorithms.PageRank(g, damping: 0.85);
GraphAlgorithms.PersonalizedPageRank(g, seeds: [42]);

GraphAlgorithms.DegreeCentrality(g);
GraphAlgorithms.ClosenessCentrality(g);
GraphAlgorithms.BetweennessCentrality(g);            // Brandes, O(VE)
GraphAlgorithms.EigenvectorCentrality(g);

GraphAlgorithms.ConnectedComponents(g);              // lemah pada graf berarah
GraphAlgorithms.StronglyConnectedComponents(g);      // Kosaraju
GraphAlgorithms.TriangleCounts(g);
GraphAlgorithms.ClusteringCoefficients(g);
GraphAlgorithms.TopologicalSort(dag);                // kosong bila ada siklus
GraphAlgorithms.LabelPropagation(g);
GraphAlgorithms.Modularity(g, communities);
```

### Dua hal yang perlu diketahui

**PageRank meredistribusi massa node menggantung.** Node tanpa edge keluar akan membocorkan
probabilitas keluar sistem dan membuat total skor tidak lagi satu. Detail itulah pembeda antara
implementasi yang benar dan yang skornya diam-diam menyusut.

**Keterhubungan pada graf berarah berarti keterhubungan *lemah*.** `ConnectedComponents` menelusuri
edge ke dua arah. Menelusuri hanya edge keluar akan memecah Cora menjadi lebih dari 1.500 serpihan
semata karena sitasi menunjuk satu arah; dengan dua arah, komponen terbesarnya berisi **2.485 dari
2.708 node**, sesuai angka yang dipublikasikan. Gunakan `StronglyConnectedComponents` bila yang
Anda maksud memang keterjangkauan timbal balik.

`BetweennessCentrality` tetap `O(VE)` bahkan dengan algoritma Brandes — batas algoritmis, bukan
batas implementasi. Batasi pada subgraf untuk jaringan besar.

## Embedding node

```csharp
using Gravicode.Science.GraviGraph.Embeddings;

var deepWalk = new DeepWalk(dimensions: 128, walksPerNode: 10, walkLength: 80, epochs: 5)
    .Train(graph.AsUndirected());

var node2vec = new Node2Vec(dimensions: 128, p: 1.0, q: 0.5, walksPerNode: 10)
    .Train(graph.AsUndirected());

embeddings.Similarity(a, b);
embeddings.MostSimilar(node, top: 10);
embeddings.LinkScore(source, target);      // prediksi tautan
embeddings.Save("nodes.vec");

RandomWalks.Uniform(g, walksPerNode: 10, walkLength: 80);
RandomWalks.Biased(g, p: 1.0, q: 0.5, walksPerNode: 10, walkLength: 80);
```

Gagasan DeepWalk adalah bahwa urutan node dari random walk memiliki statistik power-law yang sama
dengan bahasa alami, sehingga model embedding kata bisa langsung diterapkan. node2vec menambahkan
bias orde kedua:

| | Efek |
|---|---|
| `q < 1` | Walk mendorong keluar — depth-first — dan embedding menangkap **komunitas** |
| `q > 1` | Walk tetap lokal — breadth-first — dan embedding menangkap **peran struktural** |
| `p` | Seberapa mudah walk kembali ke langkah sebelumnya |

**Gunakan `AsUndirected()` dulu pada graf sitasi atau pengikut.** Menelusuri hanya searah sitasi
membuat sebagian besar walk terhenti setelah satu atau dua langkah.

## Graph neural network

```csharp
using Gravicode.Science.GraviGraph.Neural;

var gcn = new GraphConvolutionalNetwork(hiddenSize: 16, learningRate: 0.01, epochs: 200,
        dropout: 0.5, weightDecay: 5e-4, seed: 42)
    .Train(graph, trainMask, validationMask);

gcn.Predict();  gcn.PredictProbabilities();  gcn.NodeEmbeddings();
gcn.Score(graph, testMask);
gcn.History;    // loss dan akurasi per epoch

new GraphSage(hiddenSize: 16, epochs: 200).Train(graph, trainMask);
new GraphAttentionNetwork(hiddenSize: 8, heads: 4, epochs: 200).Train(graph, trainMask);
```

Ketiganya **dilatih penuh**, dan ketiganya mengambil gradiennya dari
[tape autodiff](GraviNum.md#diferensiasi-otomatis): setiap lapisan ditulis maju, dan backward
pass-nya diturunkan dari situ. Parameter attention GAT benar-benar dipelajari, termasuk melalui
softmax atas edge yang masuk ke tiap node — penurunan tersulit di library ini, dan kini tidak
diturunkan dengan tangan sama sekali.

### Menulis lapisan sendiri

Tape inilah yang membuat arsitektur keempat menjadi murah: tulis forward pass, gradiennya menyusul.

```csharp
using Gravicode.Science.GraviGraph.Neural;
using Gravicode.Science.GraviNum.Autodiff;

var propagation = graph.ToSparseAdjacency(addSelfLoops: true, symmetricNormalize: true);

var w = Tensor.Parameter(GnnMath.Glorot(features, classes, rng));
var adam = new TapeAdam(w, weightDecay: 5e-4);

for (var epoch = 0; epoch < 200; epoch++)
{
    var logits = GnnTape.Convolve(propagation, Tensor.Constant(x), w, bias);
    var loss = TensorOps.SoftmaxCrossEntropy(logits, graph.NodeLabels, trainMask);

    loss.Backward();      // tanpa penurunan rumus di mana pun
    adam.Step(0.01);
}
```

`GnnTape` menyediakan `Convolve`, `SageLayer`, `AttentionLayer`, `SegmentSoftmax`, `Dropout`,
`MeanAggregator`, `EdgeList`, dan `Descend`. Di bawahnya ada operasi tape berbentuk graf:
`SparseMatMul` untuk message passing, `Gather` dan `SegmentSum` untuk kerja tingkat edge, plus
`ConcatColumns`, `LeakyRelu`, dan `SoftmaxCrossEntropy` bermasker. Periksa setiap gradien baru
dengan `GradientCheck` sebelum mempercayainya.

`Gather` dan `SegmentSum` saling adjoint — mengumpulkan ke depan berarti menjumlahkan ke belakang —
dan itulah sebabnya attention tidak butuh kernel khusus. `SegmentSoftmax` *disusun* dari keduanya,
bukan ditulis sebagai operasi tersendiri, sehingga ia mewarisi gradien yang sudah terverifikasi
alih-alih menuntut Jacobian softmax diturunkan ulang.

> **Satu gradien turunan tangan ternyata salah sepanjang usia GCN.** Backward pass-nya melewatkan
> mask dropout pada lapisan tersembunyi, sehingga ketika dropout aktif gradiennya meleset sekitar
> **40%** — dan modelnya tetap terlatih sampai 71% pada Cora yang terlihat masuk akal, dan justru
> itulah sebabnya tidak ada yang menyadarinya. Tape menemukannya seketika, karena forward pass
> tidak punya tempat untuk menyembunyikan suku yang hilang. Akurasi uji dengan gradien yang benar
> adalah 69,3%; angka lama yang lebih tinggi berasal dari gradien rusak yang kebetulan berperan
> sebagai regularisasi aneh.

### Perbedaannya

| | Pembobotan tetangga | Sifat |
|---|---|---|
| **GCN** | Tetap: `1/sqrt(d_i d_j)` — hanya struktur | Transduktif |
| **GraphSAGE** | Dipelajari terpisah untuk diri sendiri dan rata-rata tetangga | **Induktif** |
| **GAT** | Dipelajari dari isi: `softmax(LeakyReLU(a·[Wh_i ‖ Wh_j]))` | Transduktif |

GraphSAGE menyimpan `[h_diri ; mean(h_tetangga)]` pada dua paruh masukan lapisan yang terpisah,
sehingga ia dapat menimbang "apa saya" terhadap "apa di sekitar saya" secara independen — dan
karena agregatornya didefinisikan per node alih-alih atas matriks adjacency ternormalisasi yang
tetap, bobot yang sama berlaku untuk node yang tak pernah dilihat saat pelatihan:

```csharp
var predictions = sage.PredictInductive(unseenGraph, unseenFeatures);
```

Dua lapisan adalah kedalaman lazim. Setiap lapisan mencampurkan satu hop tambahan, dan di luar
sekitar tiga hop representasi tiap node menyatu ke rata-rata graf — masalah over-smoothing.

Pelatihan GCN dan GAT bersifat transduktif: seluruh graf dilihat setiap epoch, tetapi loss hanya
dihitung pada node berlabel di `trainMask`. Pada Cora dengan 140 paper berlabel (20 per kelas) dan
60 epoch, terhadap baseline mayoritas 30,2%:

| | Akurasi uji |
|---|---:|
| GCN | 69,3% |
| GraphSAGE | 70,3% |
| **GAT** | **72,1%** |

Angka GCN yang dipublikasikan ~81%. Sisa jurangnya adalah tiadanya penjadwalan learning rate, tanpa
early stopping, dan hanya 60 epoch — bukan gradiennya, yang kini diperiksa terhadap beda hingga.
Attention mengungguli normalisasi derajat tetap adalah urutan yang memang diharapkan, dan melihatnya
justru *setelah* pindah ke tape lebih meyakinkan daripada sebelumnya.

## Kesalahan yang sering terjadi

| Gejala | Penyebab |
|---|---|
| Ribuan "komponen" pada graf sitasi | Wajar untuk keterhubungan kuat — gunakan `ConnectedComponents` untuk yang lemah |
| Random walk langsung berhenti | Grafnya berarah; panggil `AsUndirected()` |
| `ToDenseAdjacency` melempar exception | Grafnya terlalu besar — gunakan `ToSparseAdjacency` |
| Pelatihan GNN tidak berbuat apa-apa | Graf tidak punya `NodeFeatures`; berikan secara eksplisit |

## Graf heterogen

Sebagian besar graf nyata tidak homogen. Graf rekomendasi punya pengguna dan item; graf sitasi punya
makalah, penulis, dan tempat terbit. Meratakannya menjadi satu himpunan simpul menghilangkan hal yang
membuatnya informatif: bahwa "pengguna 3 membeli item 7" dan "item 7 ada di kategori 2" adalah jenis
bukti berbeda dan tidak boleh dirata-ratakan bersama.

```csharp
var graph = new HeterogeneousGraph();
graph.AddEdge("user", "watched", "film", 0, 1);
graph.AddEdge("user", "rated", "film", 1, 2, weight: 5.0);
graph.SetFeatures("user", userFeatures);       // tiap tipe boleh punya lebar berbeda
graph.SetFeatures("film", filmFeatures);
graph.AddReverseEdges(new EdgeType("user", "watched", "film"));
```

**Indeks simpul bersifat lokal terhadap tipenya** — pengguna 0 dan film 0 adalah simpul berbeda — dan
itulah yang memungkinkan tiap tipe punya dimensi fiturnya sendiri. Dan tipe sisi adalah *tripelnya*,
bukan nama relasinya: `(user, rates, film)` dan `(critic, rates, film)` adalah relasi berbeda yang
kebetulan berbagi kata kerja, dan model yang menyatukan keduanya mempelajari satu himpunan bobot
untuk dua perilaku.

Penyampaian pesan hanya bergerak searah sisi, sehingga graf bipartit dengan sisi hanya dari pengguna
ke film tidak memberi film cara untuk menginformasikan pengguna. `AddReverseEdges` adalah cara
informasi mengalir dua arah, dan ia menambahkan relasi *terpisah* dengan namanya sendiri karena
"pengguna menilai film" dan "film dinilai pengguna" layak mendapat bobot berbeda.

`RelationalConvolution` adalah R-GCN: satu matriks bobot per relasi, dijumlahkan di tujuan.

```csharp
var layer = new RelationalConvolution(graph, inputSizes, outputSize: 64);
var next = layer.Forward(graph, representations);
```

Normalisasi menurut derajat-masuk **per relasi**, bukan secara keseluruhan, dilakukan dengan sengaja.
Simpul dengan seribu sisi `viewed` dan tiga sisi `bought` kalau tidak akan membuat pembeliannya
tenggelam sama sekali — padahal pembelian itulah sinyal yang informatif. Bobot self-loop per tipe
simpul menjaga fitur simpul itu sendiri tetap hidup melewati satu lapisan; tanpanya representasi
simpul terpencil persis nol dan ia menjadi tak terbedakan dari setiap simpul terpencil lainnya.

## Fitur sisi

Bobot adalah kasus satu dimensi. Begitu ada lebih dari satu angka yang perlu dikatakan tentang sebuah
sisi — skor penilaian, nilai transaksi, sebuah timestamp — melipatnya menjadi skalar membuang
sisanya.

```csharp
graph.SetEdgeFeatures(new EdgeType("user", "rated", "film"), scoresAndTimes);
```

## Graf temporal

Jaringan transaksi, log pesan, dan catatan sitasi semuanya adalah barisan peristiwa, dan meruntuhkan
semuanya menjadi satu matriks ketetanggaan menghancurkan urutannya. Itu lebih penting daripada
tampaknya: pada graf statis, sisi `a→b` dan sisi `b→c` menyiratkan adanya jalur dari `a` ke `c`,
tetapi jika `b→c` terjadi *sebelum* `a→b`, tidak ada yang bisa menempuh jalan itu. Informasi, uang,
dan penyakit semuanya menuruti urutan tersebut, dan analisis statis secara sistematis melebih-lebihkan
apa yang terjangkau.

```csharp
var graph = TemporalGraph.LoadCsv("events.csv");

graph.TemporallyReachable(source, maxGap: 3600);   // menghormati urutan sisi
graph.Snapshot(from, to);                          // tampilan statis satu jendela
graph.SnapshotUpTo(cutoff);                        // yang boleh dilihat model pada saat itu
graph.Windows(count);
graph.TemporalEfficiency();                        // seberapa besar tampilan statis melebih-lebihkan
graph.TimeDecayedFeatures(features, asOf, halfLife: 30);
```

`TemporallyReachable` dihitung dengan satu lintasan atas sisi yang terurut waktu. Karena diproses
menurut urutan waktu, setiap sisi yang bisa memperpanjang jalur sudah memiliki waktu kedatangan
paling awal sumbernya yang final — dan itulah yang membuat satu lintasan mencukupi, sementara graf
statis memerlukan penelusuran. `maxGap` membatasi berapa lama sebuah jalur boleh menunggu di antara
sisi berurutan.

`SnapshotUpTo` adalah pemotongan yang mencegah prediktor tautan dilatih pada data ujinya sendiri.
`TimeDecayedFeatures` adalah penyematan temporal termurah yang berguna: interaksi terkini seharusnya
mengatakan lebih banyak tentang sebuah simpul daripada interaksi setahun lalu, dan agregasi statis
menimbangnya sama — itulah sebabnya model yang dilatih pada graf yang diruntuhkan terus
merekomendasikan apa yang pernah disukai seseorang, dahulu sekali.

## Klasifikasi graf

Klasifikasi simpul punya satu representasi per simpul dan tidak memerlukan readout. Klasifikasi graf
— apakah molekul ini beracun, apakah program ini berbahaya — memerlukan satu vektor per graf, dan
graf memiliki jumlah simpul yang berbeda-beda.

```csharp
GraphPooling.Pool(nodeFeatures, PoolingKind.MeanMax);
GraphPooling.AttentionPool(nodeFeatures, gate);

var classifier = new GraphClassifier(inputSize: 2, hiddenSize: 32, layers: 2).Fit(graphs, labels);
classifier.Predict(graph);
classifier.Accuracy(testGraphs, testLabels);
```

**Readout tidak boleh bergantung pada urutan simpul.** Simpul graf tidak punya penomoran kanonik,
sehingga readout yang peka terhadap permutasi membuat keluaran model bergantung pada bagaimana
berkasnya kebetulan ditulis. Setiap fungsi pooling di sini adalah agregat simetris justru karena
alasan itu, dan itulah sebabnya menyambung vektor simpul — cara paling jelas mendapatkan ukuran tetap
— bukan pilihan.

Pilihan di antaranya adalah keputusan pemodelan yang nyata. **Mean** invarian terhadap ukuran graf,
yang tepat ketika molekul besar dan kecil harus dinilai berdasarkan komposisi; **sum** tidak, yang
tepat ketika ukuran itu sendiri informatif. **Max** menanyakan apakah sebuah fitur muncul sama
sekali, yang mendeteksi satu substruktur tak biasa yang akan diencerkan oleh rata-rata.
`AttentionPool` mempelajari simpul mana yang perlu didengarkan, dan bobotnya dapat dibaca setelahnya
— ia menyatakan bagian graf mana yang mendorong prediksi.

Penyampaian pesan di sini memakai proyeksi acak tetap dengan hanya pengklasifikasi akhir yang
dilatih. Itu arsitektur yang nyata, bukan jalan pintas: ia adalah padanan graf dari model fitur acak,
dilatih dalam bentuk tertutup, dan merupakan baseline yang sungguh kuat — GNN terlatih yang tidak
mampu mengalahkannya tidak sedang mempelajari apa pun yang belum diberikan strukturnya. Versi yang
terlatih penuh tempatnya di pita autodiff berdampingan dengan `GnnTape`.

Setiap putaran penyampaian pesan melebarkan medan reseptif sebuah simpul sejauh satu lompatan; di
atas tiga atau empat, representasinya cenderung saling menyatu, dan itu adalah over-smoothing yang
tampak sebagai akurasi yang menurun seiring kedalaman.

## Pengambilan sampel ketetanggaan

Penyampaian pesan batch penuh menghitung representasi setiap simpul di setiap lapisan, sehingga satu
langkah memerlukan seluruh graf. Itu baik untuk Cora dan mustahil untuk jejaring sosial.

```csharp
foreach (var batch in NeighborSampler.Batches(trainNodes, batchSize: 512, rng))
{
    var block = NeighborSampler.Sample(graph, batch, fanOut: [10, 5], rng);
    var features = NeighborSampler.GatherFeatures(block, allFeatures);
    var output = NeighborSampler.Aggregate(block, features, weights);
}
```

Masalah yang sebenarnya dipecahkannya bukan memori melainkan **ledakan ketetanggaan**. GNN dua
lapisan pada graf berderajat rata-rata 100 menyentuh sepuluh ribu simpul per target; tiga lapisan
menyentuh sejuta. Membatasi fan-out per lompatan — sumbangan GraphSAGE — membuat biaya per target
terbatas dan tidak bergantung pada ukuran graf.

**Pengambilan sampel mengubah estimatornya, bukan sekadar kecepatannya.** Agregat tiap simpul kini
merupakan estimasi stokastik dari agregat ketetanggaan penuh, tak bias untuk agregator rata-rata dan
lebih berderau untuk fan-out kecil. Sampel yang sangat kecil membuat pelatihan tidak stabil, bukan
sekadar menjadi hampiran.

Blok dibangun ke luar dari target lalu dibalik, karena lapisan yang harus dijalankan lebih dulu
adalah yang terjauh darinya. `Aggregate` memisahkan kontribusi simpul itu sendiri dari rata-rata
ketetanggaan alih-alih memasukkannya ke dalam rata-rata — itulah yang membuat model bisa membedakan
sebuah simpul dari sekelilingnya, dan itulah perbedaan antara SAGE dan GCN biasa. **Karena itu satu
lapisan SAGE menerima dua kali lebar fiturnya**, sebab diri dan ketetanggaan disambung sebelum
diproyeksikan.

Pengacakan lebih penting di sini daripada pada mini-batching biasa. Id simpul pada graf nyata jarang
sembarang — sering mengikuti urutan penelusuran, sehingga id berurutan adalah tetangga — dan batch
tanpa pengacakan lalu menjadi satu wilayah padat, bukan sampel dari grafnya.

---

## Visualisasi

Dihasilkan oleh `samples/GraviGraph.Console`. `notebooks/GraviGraph.Notebook.ipynb` menambahkan
grafik seberapa cepat ketetanggaan tersampel membesar seiring fan-out — argumen untuk membatasinya.

![Jaringan dengan ukuran simpul menurut sentralitas](../screenshots/gravigraph_network.png)

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
