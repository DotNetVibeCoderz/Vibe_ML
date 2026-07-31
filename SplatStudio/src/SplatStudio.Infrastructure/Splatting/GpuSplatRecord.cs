using System.Runtime.InteropServices;

namespace SplatStudio.Infrastructure.Splatting;

/// <summary>
/// One splat in the exact on-disk layout of the .splat format (see
/// <see cref="SplatFileWriter"/>): 32 bytes, so the buffer the GPU fills can be
/// reinterpreted as file bytes without a second pass on the CPU.
///
/// This type must be <c>public</c> and top-level: ILGPU generates a
/// <c>ViewImplementation&lt;T&gt;</c> for every kernel buffer element type via
/// Reflection.Emit, and emitting that against an internal or nested type fails at
/// kernel-load time with "TypeLoadException: Access is denied".
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = SplatFileWriter.BytesPerPoint)]
public struct GpuSplatRecord
{
    public float PosX, PosY, PosZ;
    public float ScaleX, ScaleY, ScaleZ;
    public byte R, G, B, A;
    public byte RotX, RotY, RotZ, RotW;
}
