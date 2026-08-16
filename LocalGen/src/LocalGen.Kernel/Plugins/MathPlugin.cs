using System.ComponentModel;
using System.Globalization;
using Microsoft.SemanticKernel;

namespace LocalGen.Kernel.Plugins;

/// <summary>
/// Arithmetic the model can call instead of guessing.
/// </summary>
/// <remarks>
/// Language models are unreliable at arithmetic, and a small local model especially so. Exposing
/// a real evaluator turns "compute 17.5% of 84,320" from a coin flip into a deterministic answer.
/// </remarks>
public sealed class MathPlugin
{
    [KernelFunction("evaluate")]
    [Description(
        "Evaluates a mathematical expression and returns the exact result. " +
        "Supports + - * / % ^, parentheses, and the functions sqrt, abs, min, max, pow, log, ln, " +
        "exp, sin, cos, tan, floor, ceil, round, as well as the constants pi and e.")]
    public string Evaluate(
        [Description("The expression, for example \"(17.5 / 100) * 84320\"")] string expression)
    {
        try
        {
            var value = ExpressionEvaluator.Evaluate(expression);

            // Integral results are reported without a trailing ".0", which reads better in chat.
            return value == Math.Floor(value) && Math.Abs(value) < 1e15
                ? ((long)value).ToString(CultureInfo.InvariantCulture)
                : value.ToString("G15", CultureInfo.InvariantCulture);
        }
        catch (FormatException ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction("percentage")]
    [Description("Calculates what percent one number is of another.")]
    public string Percentage(
        [Description("The part")] double part,
        [Description("The whole")] double whole) =>
        whole == 0
            ? "Error: the whole cannot be zero."
            : (part / whole * 100).ToString("G15", CultureInfo.InvariantCulture);

    [KernelFunction("statistics")]
    [Description("Computes count, sum, mean, median, min, max and standard deviation for a list of numbers.")]
    public string Statistics(
        [Description("Numbers separated by commas or spaces, e.g. \"4, 8, 15, 16, 23, 42\"")] string numbers)
    {
        var values = numbers
            .Split([',', ' ', ';', '\n', '\t'], StringSplitOptions.RemoveEmptyEntries)
            .Select(static token => double.TryParse(
                token.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
                ? value
                : (double?)null)
            .Where(static value => value.HasValue)
            .Select(static value => value!.Value)
            .ToArray();

        if (values.Length == 0)
        {
            return "Error: no numbers were found in the input.";
        }

        Array.Sort(values);

        var mean = values.Average();
        var median = values.Length % 2 == 1
            ? values[values.Length / 2]
            : (values[values.Length / 2 - 1] + values[values.Length / 2]) / 2;

        // Population standard deviation; a single value has no spread.
        var standardDeviation = values.Length > 1
            ? Math.Sqrt(values.Sum(v => Math.Pow(v - mean, 2)) / values.Length)
            : 0;

        return $"count={values.Length}, sum={values.Sum():G15}, mean={mean:G15}, " +
               $"median={median:G15}, min={values[0]:G15}, max={values[^1]:G15}, " +
               $"stddev={standardDeviation:G15}";
    }
}

/// <summary>
/// A recursive-descent evaluator for arithmetic expressions.
/// </summary>
/// <remarks>
/// Written by hand rather than delegating to <c>DataTable.Compute</c> or a scripting engine:
/// this input comes from a language model, so it must not be able to reach anything but numbers.
/// </remarks>
internal static class ExpressionEvaluator
{
    public static double Evaluate(string expression)
    {
        if (string.IsNullOrWhiteSpace(expression))
        {
            throw new FormatException("the expression is empty.");
        }

        var parser = new Parser(expression);
        var value = parser.ParseExpression();
        parser.ExpectEnd();
        return value;
    }

    private sealed class Parser(string text)
    {
        private readonly string _text = text;
        private int _position;

        /// <summary>expression := term (('+' | '-') term)*</summary>
        public double ParseExpression()
        {
            var left = ParseTerm();

            while (true)
            {
                SkipWhitespace();

                if (Match('+'))
                {
                    left += ParseTerm();
                }
                else if (Match('-'))
                {
                    left -= ParseTerm();
                }
                else
                {
                    return left;
                }
            }
        }

        /// <summary>term := factor (('*' | '/' | '%') factor)*</summary>
        private double ParseTerm()
        {
            var left = ParseFactor();

            while (true)
            {
                SkipWhitespace();

                if (Match('*'))
                {
                    left *= ParseFactor();
                }
                else if (Match('/'))
                {
                    var divisor = ParseFactor();
                    if (divisor == 0)
                    {
                        throw new FormatException("division by zero.");
                    }

                    left /= divisor;
                }
                else if (Match('%'))
                {
                    var divisor = ParseFactor();
                    if (divisor == 0)
                    {
                        throw new FormatException("modulo by zero.");
                    }

                    left %= divisor;
                }
                else
                {
                    return left;
                }
            }
        }

        /// <summary>factor := unary ('^' factor)? — right-associative, as exponentiation should be.</summary>
        private double ParseFactor()
        {
            var left = ParseUnary();

            SkipWhitespace();
            return Match('^') ? Math.Pow(left, ParseFactor()) : left;
        }

        private double ParseUnary()
        {
            SkipWhitespace();

            if (Match('-'))
            {
                return -ParseUnary();
            }

            if (Match('+'))
            {
                return ParseUnary();
            }

            return ParsePrimary();
        }

        private double ParsePrimary()
        {
            SkipWhitespace();

            if (_position >= _text.Length)
            {
                throw new FormatException("the expression ends unexpectedly.");
            }

            if (Match('('))
            {
                var value = ParseExpression();
                SkipWhitespace();

                if (!Match(')'))
                {
                    throw new FormatException("a closing parenthesis is missing.");
                }

                return value;
            }

            var current = _text[_position];

            if (char.IsAsciiDigit(current) || current == '.')
            {
                return ParseNumber();
            }

            if (char.IsAsciiLetter(current))
            {
                return ParseIdentifier();
            }

            throw new FormatException($"unexpected character '{current}' at position {_position}.");
        }

        private double ParseNumber()
        {
            var start = _position;

            while (_position < _text.Length &&
                   (char.IsAsciiDigit(_text[_position]) || _text[_position] == '.' ||
                    _text[_position] == '_' || _text[_position] == ','))
            {
                _position++;
            }

            // Thousands separators are stripped so "84,320" parses the way a person means it.
            var token = _text[start.._position].Replace("_", string.Empty).Replace(",", string.Empty);

            // Scientific notation: consume the exponent when one follows.
            if (_position < _text.Length && (_text[_position] is 'e' or 'E'))
            {
                var save = _position;
                _position++;

                if (_position < _text.Length && (_text[_position] is '+' or '-'))
                {
                    _position++;
                }

                if (_position < _text.Length && char.IsAsciiDigit(_text[_position]))
                {
                    while (_position < _text.Length && char.IsAsciiDigit(_text[_position]))
                    {
                        _position++;
                    }

                    token = _text[start.._position].Replace("_", string.Empty).Replace(",", string.Empty);
                }
                else
                {
                    _position = save;
                }
            }

            return double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
                ? value
                : throw new FormatException($"'{token}' is not a valid number.");
        }

        private double ParseIdentifier()
        {
            var start = _position;

            while (_position < _text.Length && char.IsAsciiLetterOrDigit(_text[_position]))
            {
                _position++;
            }

            var name = _text[start.._position].ToLowerInvariant();

            SkipWhitespace();

            if (!Match('('))
            {
                return name switch
                {
                    "pi" => Math.PI,
                    "e" => Math.E,
                    "tau" => Math.Tau,
                    _ => throw new FormatException($"unknown constant '{name}'.")
                };
            }

            var arguments = new List<double>();

            SkipWhitespace();
            if (!Match(')'))
            {
                do
                {
                    arguments.Add(ParseExpression());
                    SkipWhitespace();
                }
                while (Match(','));

                if (!Match(')'))
                {
                    throw new FormatException($"'{name}(' is missing its closing parenthesis.");
                }
            }

            return Apply(name, arguments);
        }

        private static double Apply(string name, List<double> arguments)
        {
            double One() => arguments.Count == 1
                ? arguments[0]
                : throw new FormatException($"{name} takes exactly one argument.");

            double Two(int index) => arguments.Count == 2
                ? arguments[index]
                : throw new FormatException($"{name} takes exactly two arguments.");

            return name switch
            {
                "sqrt" => Math.Sqrt(One()),
                "abs" => Math.Abs(One()),
                "floor" => Math.Floor(One()),
                "ceil" or "ceiling" => Math.Ceiling(One()),
                "round" => arguments.Count switch
                {
                    1 => Math.Round(arguments[0], MidpointRounding.AwayFromZero),
                    2 => Math.Round(arguments[0], (int)arguments[1], MidpointRounding.AwayFromZero),
                    _ => throw new FormatException("round takes one or two arguments.")
                },
                "exp" => Math.Exp(One()),
                "ln" => Math.Log(One()),
                "log" => arguments.Count switch
                {
                    1 => Math.Log10(arguments[0]),
                    2 => Math.Log(arguments[0], arguments[1]),
                    _ => throw new FormatException("log takes one or two arguments.")
                },
                "sin" => Math.Sin(One()),
                "cos" => Math.Cos(One()),
                "tan" => Math.Tan(One()),
                "asin" => Math.Asin(One()),
                "acos" => Math.Acos(One()),
                "atan" => Math.Atan(One()),
                "pow" => Math.Pow(Two(0), Two(1)),
                "min" => arguments.Count > 0 ? arguments.Min() : throw new FormatException("min needs arguments."),
                "max" => arguments.Count > 0 ? arguments.Max() : throw new FormatException("max needs arguments."),
                _ => throw new FormatException($"unknown function '{name}'.")
            };
        }

        public void ExpectEnd()
        {
            SkipWhitespace();

            if (_position < _text.Length)
            {
                throw new FormatException(
                    $"unexpected trailing input '{_text[_position..]}'.");
            }
        }

        private void SkipWhitespace()
        {
            while (_position < _text.Length && char.IsWhiteSpace(_text[_position]))
            {
                _position++;
            }
        }

        private bool Match(char expected)
        {
            if (_position < _text.Length && _text[_position] == expected)
            {
                _position++;
                return true;
            }

            return false;
        }
    }
}
