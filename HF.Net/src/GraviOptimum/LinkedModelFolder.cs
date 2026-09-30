using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace Gravicode.HFNet.GraviOptimum;

/// <summary>
/// Gives ONNX Runtime a model folder it will accept when the real one is made of symbolic links.
/// </summary>
/// <remarks>
/// <para>
/// Python's Hugging Face cache stores each file once under <c>blobs/</c> and puts a symbolic link to
/// it in every snapshot folder. ONNX Runtime resolves a model's external data file (<c>weights.pb</c>,
/// <c>model.onnx_data</c>) to its real path and refuses it when that path leaves the model's folder -
/// which a link into <c>blobs/</c> always does - with "External data path ... escapes model directory".
/// </para>
/// <para>
/// So a folder with links in it is opened through a sibling folder of <b>hard</b> links to the same
/// files: a hard link is a second name for the file, with no target to resolve and no second copy of
/// the bytes. Where one cannot be made (another volume, a file system without them) the file is copied.
/// </para>
/// </remarks>
internal static class LinkedModelFolder
{
    /// <summary>The path to open: <paramref name="modelPath"/> itself, or its twin in a folder of hard links.</summary>
    internal static string Resolve(string modelPath)
    {
        var folder = Path.GetDirectoryName(Path.GetFullPath(modelPath))!;
        var files = Directory.GetFiles(folder);
        if (!files.Any(f => new FileInfo(f).LinkTarget is not null)) return modelPath;

        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(folder)))[..16];
        var view = Path.Combine(Path.GetTempPath(), "hfnet-onnx", hash);
        Directory.CreateDirectory(view);

        foreach (var file in files)
        {
            var info = new FileInfo(file);
            var source = info.LinkTarget is null ? info : (FileInfo)(info.ResolveLinkTarget(returnFinalTarget: true) ?? info);
            var target = new FileInfo(Path.Combine(view, info.Name));

            if (target.Exists && target.Length == source.Length && target.LastWriteTimeUtc >= source.LastWriteTimeUtc) continue;
            if (target.Exists) target.Delete();

            if (!HardLink(target.FullName, source.FullName)) File.Copy(source.FullName, target.FullName);
        }

        return Path.Combine(view, Path.GetFileName(modelPath));
    }

    private static bool HardLink(string link, string existing)
    {
        try
        {
            return OperatingSystem.IsWindows()
                ? CreateHardLinkW(link, existing, IntPtr.Zero)
                : link_unix(existing, link) == 0;
        }
        catch (Exception exception) when (exception is DllNotFoundException or EntryPointNotFoundException)
        {
            return false;
        }
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateHardLinkW(string lpFileName, string lpExistingFileName, IntPtr lpSecurityAttributes);

    [DllImport("libc", EntryPoint = "link", SetLastError = true)]
    private static extern int link_unix(string existing, string link);
}
