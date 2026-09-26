namespace MediaPipeNet;

/// <summary>Base exception for errors raised by MediaPipe.NET.</summary>
public class MediaPipeException : Exception
{
    /// <summary>Creates the exception.</summary>
    public MediaPipeException() { }

    /// <summary>Creates the exception with a message.</summary>
    public MediaPipeException(string message) : base(message) { }

    /// <summary>Creates the exception with a message and an inner exception.</summary>
    public MediaPipeException(string message, Exception innerException) : base(message, innerException) { }
}

/// <summary>Raised when a required model cannot be found, downloaded or verified.</summary>
public class ModelNotFoundException : MediaPipeException
{
    /// <summary>Creates the exception.</summary>
    public ModelNotFoundException() { }

    /// <summary>Creates the exception with a message.</summary>
    public ModelNotFoundException(string message) : base(message) { }

    /// <summary>Creates the exception with a message and an inner exception.</summary>
    public ModelNotFoundException(string message, Exception innerException) : base(message, innerException) { }
}
