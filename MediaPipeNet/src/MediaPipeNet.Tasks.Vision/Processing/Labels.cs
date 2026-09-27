namespace MediaPipeNet.Tasks.Vision.Processing;

/// <summary>Label maps embedded in the assembly.</summary>
public static class Labels
{
    private static readonly Lazy<IReadOnlyList<string>> s_coco = new(() => Load("coco_labels.txt"));
    private static readonly Lazy<IReadOnlyList<string>> s_imagenet = new(() => Load("imagenet_labels.txt"));
    private static readonly Lazy<IReadOnlyList<string>> s_gestures = new(() => Load("gesture_labels.txt"));
    private static readonly Lazy<IReadOnlyList<string>> s_selfieMulticlass = new(() => Load("selfie_multiclass_labels.txt"));
    private static readonly Lazy<IReadOnlyList<string>> s_hair = new(() => Load("hair_labels.txt"));
    private static readonly Lazy<IReadOnlyList<string>> s_pascalVoc = new(() => Load("deeplab_labels.txt"));

    /// <summary>The 90 COCO labels of EfficientDet-Lite (unused ids are "???").</summary>
    public static IReadOnlyList<string> Coco => s_coco.Value;

    /// <summary>The 1000 ImageNet labels of EfficientNet-Lite.</summary>
    public static IReadOnlyList<string> ImageNet => s_imagenet.Value;

    /// <summary>The 8 canned gestures: None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou.</summary>
    public static IReadOnlyList<string> Gestures => s_gestures.Value;

    /// <summary>The 6 selfie multiclass categories: background, hair, body-skin, face-skin, clothes, others.</summary>
    public static IReadOnlyList<string> SelfieMulticlass => s_selfieMulticlass.Value;

    /// <summary>The hair segmenter categories: background, hair.</summary>
    public static IReadOnlyList<string> Hair => s_hair.Value;

    /// <summary>The 21 PASCAL VOC categories of DeepLab v3 (background, aeroplane, ..., tv).</summary>
    public static IReadOnlyList<string> PascalVoc => s_pascalVoc.Value;

    private static string[] Load(string name)
    {
        using var stream = typeof(Labels).Assembly.GetManifestResourceStream("MediaPipeNet.Tasks.Vision.Resources." + name)
                           ?? throw new InvalidOperationException($"Missing embedded label file {name}.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd().Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0).ToArray();
    }
}
