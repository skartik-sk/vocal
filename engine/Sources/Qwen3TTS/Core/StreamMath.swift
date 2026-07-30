import Foundation

/// Pure helpers for chunked audio streaming. No MLX — fully unit-testable.
public enum StreamMath {
    /// Returns `next`, with its first `overlap` samples linearly blended into the
    /// tail of `prev` (linear ramp). `overlap` is clamped to the shorter array.
    /// Used to glue consecutive decoded chunks without clicks at the junction.
    public static func crossfade(prev: [Float], next: [Float], overlap: Int) -> [Float] {
        var out = next
        let r = max(0, min(overlap, prev.count, next.count))
        guard r > 0 else { return out }
        for i in 0..<r {
            let w = Float(i) / Float(r) // 0..<1 ramp
            out[i] = prev[prev.count - r + i] * (1 - w) + next[i] * w
        }
        return out
    }

    /// Sample-index range within a decoded window corresponding to codec tokens
    /// `[start..<end)`, where the window's first token is `windowStart`.
    public static func samples(
        forTokenRangeStart start: Int, end: Int,
        windowStart: Int, samplesPerToken: Int
    ) -> Range<Int> {
        let lo = (start - windowStart) * samplesPerToken
        let hi = (end - windowStart) * samplesPerToken
        return lo..<hi
    }
}
