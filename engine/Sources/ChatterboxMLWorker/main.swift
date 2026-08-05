//
//  main.swift — Chatterbox multilingual (Hindi) worker: reads text from
//  CHATTERBOX_ML_TEXT (or the first CLI arg), synthesizes speech, and writes
//  a 24 kHz WAV to CHATTERBOX_ML_OUT (default: /tmp/chatterbox_ml.wav).
//

import Foundation
import MLX
import Chatterbox

func writeWav(_ samples: [Float], to url: URL, sampleRate: Int = 24000) throws {
    var pcm = Data()
    for s in samples {
        var v = Int16(max(-1, min(1, s)) * 32767)
        pcm.append(Data(bytes: &v, count: 2))
    }
    var out = Data()
    func put(_ s: String) { out.append(s.data(using: .ascii)!) }
    func put32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.append(Data($0)) } }
    func put16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { out.append(Data($0)) } }
    put("RIFF"); put32(UInt32(36 + pcm.count)); put("WAVE")
    put("fmt "); put32(16); put16(1); put16(1); put32(UInt32(sampleRate))
    put32(UInt32(sampleRate * 2)); put16(2); put16(16)
    put("data"); put32(UInt32(pcm.count)); out.append(pcm)
    try out.write(to: url)
}

@main
struct ChatterboxMLWorkerMain {
    static func main() async throws {
        let env = ProcessInfo.processInfo.environment
        let modelPath = env["CHATTERBOX_ML_MODEL"] ?? "/tmp/chatterbox-4bit"
        let outPath = env["CHATTERBOX_ML_OUT"] ?? "/tmp/chatterbox_ml.wav"
        let text: String
        if CommandLine.arguments.count > 1 {
            text = CommandLine.arguments.dropFirst().joined(separator: " ")
        } else if let t = env["CHATTERBOX_ML_TEXT"] {
            text = t
        } else {
            text = "This is Swift Chatterbox speaking. नमस्ते दोस्तों, this is the real test."
        }
        let lang = env["CHATTERBOX_ML_LANG"] ?? "hi"
        let maxTokens = Int(env["CHATTERBOX_ML_MAX_TOKENS"] ?? "150") ?? 150

        let model = try await ChatterboxML.fromPretrained(modelPath)
        let wav = model.generate(text: text, language: lang, temperature: 0.8,
                                 maxSpeechTokens: maxTokens)
        let peak = wav.map { abs($0) }.max() ?? 0
        try writeWav(wav, to: URL(fileURLWithPath: outPath))
        print("[ChatterboxMLWorker] text-tokens=\(model.tokenCount(text: text, language: lang)) wav=\(wav.count) peak=\(peak)")
        print("[ChatterboxMLWorker] wrote \(outPath)")
    }
}
