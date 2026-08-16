using System.Text;
using LocalGen.Core.Inference;

namespace LocalGen.Core.Prompting;

/// <summary>
/// Splits a token stream into user-visible text and completed tool calls.
/// </summary>
/// <remarks>
/// Tags arrive split across tokens, so text is held back whenever its tail could still grow
/// into <see cref="ToolCallProtocol.OpenTag"/>. Without that, the opening angle bracket of a
/// tool call would be streamed to the user a token before the rest of the tag revealed itself.
/// </remarks>
public sealed class ToolCallStreamFilter
{
    private readonly StringBuilder _pending = new();
    private readonly List<ToolCall> _calls = [];
    private bool _insideCall;

    /// <summary>Tool calls parsed so far.</summary>
    public IReadOnlyList<ToolCall> Calls => _calls;

    public bool HasCalls => _calls.Count > 0;

    /// <summary>
    /// Feeds newly generated text and returns the portion that is safe to show the user,
    /// which may be empty while a tag is being disambiguated.
    /// </summary>
    public string Push(string token)
    {
        _pending.Append(token);
        return Drain(flush: false);
    }

    /// <summary>
    /// Flushes everything still buffered at end of generation. An unterminated tool call is
    /// released as plain text rather than being swallowed.
    /// </summary>
    public string Flush() => Drain(flush: true);

    private string Drain(bool flush)
    {
        var output = new StringBuilder();

        while (true)
        {
            var buffer = _pending.ToString();

            if (_insideCall)
            {
                var close = buffer.IndexOf(ToolCallProtocol.CloseTag, StringComparison.Ordinal);
                if (close < 0)
                {
                    if (flush)
                    {
                        // Generation ended without a closing tag. Models do this often — they
                        // emit the call and then stop — so the payload is given a chance to
                        // parse on its own before being written off as text. Only genuinely
                        // truncated JSON falls through to the raw output.
                        if (ToolCallProtocol.TryParseUnterminated(buffer, _calls.Count, out var recovered))
                        {
                            _calls.Add(recovered);
                        }
                        else
                        {
                            output.Append(ToolCallProtocol.OpenTag).Append(buffer);
                        }

                        _pending.Clear();
                        _insideCall = false;
                    }

                    break;
                }

                var payload = buffer[..close];
                var (calls, _) = ToolCallProtocol.Parse(
                    ToolCallProtocol.OpenTag + payload + ToolCallProtocol.CloseTag);

                if (calls.Count > 0)
                {
                    _calls.AddRange(calls);
                }
                else
                {
                    output.Append(ToolCallProtocol.OpenTag).Append(payload).Append(ToolCallProtocol.CloseTag);
                }

                _pending.Remove(0, close + ToolCallProtocol.CloseTag.Length);
                _insideCall = false;
                continue;
            }

            var open = buffer.IndexOf(ToolCallProtocol.OpenTag, StringComparison.Ordinal);
            if (open >= 0)
            {
                output.Append(buffer, 0, open);
                _pending.Remove(0, open + ToolCallProtocol.OpenTag.Length);
                _insideCall = true;
                continue;
            }

            if (flush)
            {
                output.Append(buffer);
                _pending.Clear();
                break;
            }

            // Hold back any tail that could still become an opening tag.
            var safeLength = buffer.Length - PartialTagLength(buffer, ToolCallProtocol.OpenTag);
            if (safeLength > 0)
            {
                output.Append(buffer, 0, safeLength);
                _pending.Remove(0, safeLength);
            }

            break;
        }

        return output.ToString();
    }

    /// <summary>Length of the longest suffix of <paramref name="text"/> that starts <paramref name="tag"/>.</summary>
    private static int PartialTagLength(string text, string tag)
    {
        var max = Math.Min(text.Length, tag.Length - 1);

        for (var length = max; length > 0; length--)
        {
            if (string.CompareOrdinal(text, text.Length - length, tag, 0, length) == 0)
            {
                return length;
            }
        }

        return 0;
    }
}
