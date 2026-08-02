//
//  BPE.swift — minimal BPE tokenizer for the multilingual Chatterbox
//  (loads tokenizer.json vocab + merges directly; no HF config needed).
//

import Foundation
import MLX

/// A minimal byte-level BPE tokenizer. Sufficient for the chatterbox
/// tokenizer.json (vocab + merges + added_tokens).
final class BPE: @unchecked Sendable {
    let vocab: [String: Int]
    let merges: [String: Int]          // "a b" -> rank
    let addedTokens: [String: Int]
    private let vocabByID: [Int: String]

    init(vocab: [String: Int], merges: [String: Int], addedTokens: [String: Int]) {
        self.vocab = vocab
        self.merges = merges
        self.addedTokens = addedTokens
        var byID = [Int: String]()
        for (k, v) in vocab { byID[v] = k }
        for (k, v) in addedTokens { byID[v] = k }
        self.vocabByID = byID
    }

    /// Load from tokenizer.json (Foundation JSON).
    static func load(_ data: Data) throws -> BPE {
        let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let model = obj["model"] as! [String: Any]
        let vocabObj = model["vocab"] as! [String: Any]
        var vocab = [String: Int]()
        for (k, v) in vocabObj { vocab[k] = (v as! NSNumber).intValue }
        var merges = [String: Int]()
        if let mergeList = model["merges"] as? [Any] {
            for (i, m) in mergeList.enumerated() {
                merges[(m as! String)] = i
            }
        }
        var added = [String: Int]()
        if let addedTokens = obj["added_tokens"] as? [[String: Any]] {
            for t in addedTokens {
                if let content = t["content"] as? String, let id = t["id"] as? NSNumber {
                    added[content] = id.intValue
                }
            }
        }
        return BPE(vocab: vocab, merges: merges, addedTokens: added)
    }

    /// Tokenize text to token IDs (single merge pass).
    func encode(text: String) -> [Int] {
        // 1. Match longest added tokens first (e.g. [SPACE], [START]).
        var chars = Array(text.unicodeScalars).map { String($0) }
        // Replace known added-token substrings (they can contain multi-char tokens).
        var out: [String] = []
        var i = 0
        let keys = addedTokens.keys.sorted { $0.count > $1.count }
        while i < chars.count {
            var matched: String? = nil
            for k in keys {
                let ks = Array(k.unicodeScalars).map { String($0) }
                if ks.count <= chars.count - i, Array(chars[i ..< (i + ks.count)]) == ks {
                    matched = k
                    break
                }
            }
            if let m = matched {
                out.append(m)
                i += Array(m.unicodeScalars).count
            } else {
                out.append(chars[i])
                i += 1
            }
        }

        // 2. Byte fallback for unknown chars (utf-8 bytes as single chars).
        var tokens = out.flatMap { piece -> [String] in
            if vocab[piece] != nil { return [piece] }
            if addedTokens[piece] != nil { return [piece] }
            // unknown: split into bytes with <0x..> style? The model's BPE is
            // char-level; unknown chars map through utf8 bytes.
            return Array(piece.utf8).map { String(UnicodeScalar($0)) }
        }

        // 3. Greedy merges.
        var changed = true
        while changed && tokens.count > 1 {
            changed = false
            var bestRank = Int.max
            var bestIdx = -1
            for idx in 0 ..< (tokens.count - 1) {
                let pair = tokens[idx] + " " + tokens[idx + 1]
                if let rank = merges[pair], rank < bestRank {
                    bestRank = rank
                    bestIdx = idx
                }
            }
            if bestIdx >= 0 {
                let merged = tokens[bestIdx] + tokens[bestIdx + 1]
                tokens.replaceSubrange(bestIdx ... (bestIdx + 1), with: [merged])
                changed = true
            }
        }

        // 4. Map to ids.
        return tokens.compactMap { token -> Int? in
            if let id = vocab[token] { return id }
            return addedTokens[token]
        }
    }
}
