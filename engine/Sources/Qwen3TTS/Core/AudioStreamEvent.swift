import Foundation

/// Events yielded by `Qwen3TTSModel.generateAudioStream`.
public enum AudioStreamEvent: Sendable {
    /// Incremental 24 kHz mono Float samples ready to play.
    case chunk([Float])
    /// Final generation statistics.
    case info(AudioGenerationInfo)
}
