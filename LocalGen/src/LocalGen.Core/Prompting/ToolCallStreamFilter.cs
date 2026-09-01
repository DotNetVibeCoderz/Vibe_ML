using System.Text;
using LocalGen.Core.Inference;

namespace LocalGen.Core.Prompting;

/// <summary>
/// Splits a token stream into user-visible text and completed tool calls.
/// </summary>
/// <remarks>
/// Tags arrive split across tokens, so text is held back whenever its tail could still grow
/// into the dialect's opening tag. Without that, the opening angle bracket of a tool call would
/// be streamed to the user a token before the rest of the tag revealed itself.
/// </remarks>
public sealed class ToolCallStreamFilter
{
    private readonly StringBuilder _pending = new();
    private readonly List<ToolCall> _calls = [];
    private readonly ToolDialect _dialect;
    private bool _insideCall;

    // Markerless dialects only: whether the opening character has been seen and judged, and
    // whether that judgement was "prose", after which nothing more needs inspecting.
    private bool _decided;
    private bool _streamingProse;

    /// <param name="dialect">
    /// The model family's call convention. Defaults to the Hermes form, which is what an
    /// unrecognised model is asked to use.
    /// </param>
    public ToolCallStreamFilter(ToolDialect? dialect = null) => _dialect = dialect ?? ToolDialect.Hermes;

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
        if (!_dialect.HasOpenTag)
        {
            return DrainMarkerless(flush);
        }

        var output = new StringBuilder();

        while (true)
        {
            var buffer = _pending.ToString();

            if (_insideCall)
            {
                // A dialect with no closing marker ends its call at the end of the turn, so there
                // is nothing to search for — the payload is held until the flush.
                var close = _dialect.HasCloseTag
                    ? buffer.IndexOf(_dialect.CloseTag, StringComparison.Ordinal)
                    : -1;

                if (close < 0)
                {
                    if (flush)
                    {
                        // Generation ended without a closing tag. Models do this often — they
                        // emit the call and then stop — so the payload is given a chance to
                        // parse on its own before being written off as text. Only genuinely
                        // truncated JSON falls through to the raw output.
                        if (_dialect.TryParseUnterminated(buffer, _calls.Count, out var recovered))
                        {
                            _calls.AddRange(recovered);
                        }
                        else
                        {
                            output.Append(_dialect.OpenTag).Append(buffer);
                        }

                        _pending.Clear();
                        _insideCall = false;
                    }

                    break;
                }

                var payload = buffer[..close];
                var (calls, _) = _dialect.Parse(
                    _dialect.OpenTag + payload + _dialect.CloseTag);

                if (calls.Count > 0)
                {
                    _calls.AddRange(calls);
                }
                else
                {
                    output.Append(_dialect.OpenTag).Append(payload).Append(_dialect.CloseTag);
                }

                _pending.Remove(0, close + _dialect.CloseTag.Length);
                _insideCall = false;
                continue;
            }

            var open = buffer.IndexOf(_dialect.OpenTag, StringComparison.Ordinal);
            if (open >= 0)
            {
                output.Append(buffer, 0, open);
                _pending.Remove(0, open + _dialect.OpenTag.Length);
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
            var safeLength = buffer.Length - PartialTagLength(buffer, _dialect.OpenTag);
            if (safeLength > 0)
            {
                output.Append(buffer, 0, safeLength);
                _pending.Remove(0, safeLength);
            }

            break;
        }

        return output.ToString();
    }

    /// <summary>
    /// Streams a family that marks its calls with nothing at all, such as Llama 3.
    /// </summary>
    /// <remarks>
    /// With no opening tag there is nothing to hold text back on, so the decision is made from
    /// the first non-whitespace character instead: a reply that opens with a brace is a candidate
    /// call and is buffered whole until the end, and a reply that opens with anything else is
    /// prose and streams from then on without further inspection. Deciding once is what keeps a
    /// long answer from being buffered in its entirety on the chance it turns into JSON.
    /// </remarks>
    private string DrainMarkerless(bool flush)
    {
        var buffer = _pending.ToString();

        if (_streamingProse)
        {
            _pending.Clear();
            return buffer;
        }

        if (!_decided)
        {
            var leading = buffer.TrimStart();

            if (leading.Length == 0)
            {
                // Nothing but whitespace so far, and it is still anyone's guess. Held rather than
                // emitted, because releasing it would commit to prose before knowing.
                if (!flush)
                {
                    return string.Empty;
                }

                _pending.Clear();
                return buffer;
            }

            var opening = _dialect.PayloadIsArray ? '[' : '{';
            var prefix = _dialect.OptionalPrefix;

            // Still too short to tell the optional prefix from prose that merely starts the same
            // way. Waiting costs a few tokens of latency; guessing costs a whole reply.
            if (prefix.Length > 0 &&
                leading.Length < prefix.Length &&
                prefix.StartsWith(leading, StringComparison.Ordinal) &&
                !flush)
            {
                return string.Empty;
            }

            var couldBeCall = leading.StartsWith(opening) ||
                (prefix.Length > 0 && leading.StartsWith(prefix, StringComparison.Ordinal));

            _decided = true;

            if (!couldBeCall)
            {
                _streamingProse = true;
                _pending.Clear();
                return buffer;
            }

            _insideCall = true;
        }

        if (!flush)
        {
            return string.Empty;
        }

        _pending.Clear();
        _insideCall = false;

        var (calls, _) = _dialect.Parse(buffer);

        if (calls.Count > 0)
        {
            _calls.AddRange(calls);
            return string.Empty;
        }

        // It opened like a call but never became one. Better shown than swallowed.
        return buffer;
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
