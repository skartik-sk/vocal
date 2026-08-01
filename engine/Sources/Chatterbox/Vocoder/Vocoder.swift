//
//  Vocoder.swift — HiFTNet: mel-spectrogram → 24 kHz waveform.
//
//  Ported from chatterbox_turbo/models/s3gen/hifigan.py. Neural source filter + iSTFTNet.
//  Convs are float16 [O,K,I] (load directly). Snake.alpha ships as a raw [channels] param → the
//  loader renames `.alpha`→`.alpha.weight` and reshapes to [channels,1] so it is held by an
//  Embedding. The STFT (source branch) uses precomputed real DFT bases; the final iSTFT is done
//  in plain Swift Float (overlap-add) since it is the last op in the pipeline.
//

import Foundation
import MLX
import MLXNN

// Reuses Conv1dPT / ConvTranspose1dPT from Decoder.swift.

// MARK: - Snake activation

final class Snake: Module {
    @ModuleInfo(key: "alpha") var alpha: Embedding   // [channels, 1] (raw param, renamed .weight)
    let channels: Int
    init(_ channels: Int) {
        self.channels = channels
        _alpha.wrappedValue = Embedding(embeddingCount: channels, dimensions: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, C, T); alpha -> (1, C, 1)
        let a = alpha.weight.reshaped([1, channels, 1])
        let aClamped = MLX.where(a.abs() .< 1e-9, MLXArray(1e-4), a)
        return x + (1.0 / aClamped) * square(sin(x * a))
    }
}

// MARK: - HiFi-GAN residual block

final class HifiResBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1dPT]
    @ModuleInfo(key: "convs2") var convs2: [Conv1dPT]
    @ModuleInfo(key: "activations1") var activations1: [Snake]
    @ModuleInfo(key: "activations2") var activations2: [Snake]

    init(_ channels: Int, kernel k: Int, dilations: [Int]) {
        func pad(_ d: Int) -> Int { (k * d - d) / 2 }
        _convs1.wrappedValue = dilations.map { Conv1dPT(channels, channels, kernel: k, padding: pad($0), dilation: $0) }
        _convs2.wrappedValue = dilations.map { _ in Conv1dPT(channels, channels, kernel: k, padding: pad(1)) }
        _activations1.wrappedValue = dilations.map { _ in Snake(channels) }
        _activations2.wrappedValue = dilations.map { _ in Snake(channels) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for i in 0..<convs1.count {
            var xt = activations1[i](out)
            xt = convs1[i](xt)
            xt = activations2[i](xt)
            xt = convs2[i](xt)
            out = xt + out
        }
        return out
    }
}

// MARK: - Source module (sinusoidal excitation)

final class SineGen: Module {
    let samplingRate: Float
    let harmonicNum: Int
    let sineAmp: Float
    let noiseStd: Float
    let voicedThreshold: Float

    init(samplingRate: Int, harmonicNum: Int = 0, sineAmp: Float = 0.1,
         noiseStd: Float = 0.003, voicedThreshold: Float = 0) {
        self.samplingRate = Float(samplingRate)
        self.harmonicNum = harmonicNum
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.voicedThreshold = voicedThreshold
        super.init()
    }

    /// f0 (B, 1, T) → (sine_waves [B, H, T], uv [B,1,T])
    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray) {
        let B = f0.dim(0), T = f0.dim(2)
        let harmonics = MLXArray((1...(harmonicNum + 1)).map { Float($0) })
            .reshaped([1, harmonicNum + 1, 1])              // (1, H, 1)
        let Fmat = f0 * harmonics / samplingRate             // (B, H, T)
        let two = 2.0 * Float.pi
        var theta = two * Fmat.cumsum(axis: -1)
        theta = theta - floor(theta / two) * two             // mod 2π
        // voiced/unvoiced
        let uv = (f0 .> voicedThreshold).asType(.float32)    // (B,1,T)
        let sine = sineAmp * sin(theta)                      // (B,H,T)
        let noiseAmp = uv * noiseStd + (1.0 - uv) * (sineAmp / 3.0)
        let noise = noiseAmp * MLXRandom.normal(sine.shape)
        let sineWaves = sine * uv + noise
        _ = B; _ = T
        return (sineWaves, uv)
    }
}

final class SourceModule: Module {
    var lSinGen: SineGen
    @ModuleInfo(key: "l_linear") var lLinear: Linear
    let sineAmp: Float

    init(samplingRate: Int, harmonicNum: Int = 8, sineAmp: Float = 0.1,
         noiseStd: Float = 0.003, voicedThreshold: Float = 10) {
        self.lSinGen = SineGen(samplingRate: samplingRate, harmonicNum: harmonicNum,
                               sineAmp: sineAmp, noiseStd: noiseStd, voicedThreshold: voicedThreshold)
        _lLinear.wrappedValue = Linear(harmonicNum + 1, 1)
        self.sineAmp = sineAmp
        super.init()
    }

    /// f0 (B, T, 1) → (sine_merge [B,T,1], noise [B,T,1], uv [B,T,1])
    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray) {
        let (sineW, uv) = lSinGen(f0.transposed(0, 2, 1))    // sine (B,H,T), uv (B,1,T)
        let sineWavs = sineW.transposed(0, 2, 1)             // (B,T,H)
        let sineMerge = tanh(lLinear(sineWavs))              // (B,T,1)
        let noise = MLXRandom.normal(uv.transposed(0, 2, 1).shape) * (sineAmp / 3.0)
        return (sineMerge, noise)
    }
}

// MARK: - F0 predictor

private func eluAct(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    MLX.where(x .> 0, x, alpha * (exp(x) - 1.0))
}

final class F0Predictor: Module {
    @ModuleInfo(key: "condnet") var condnet: [Conv1dPT]
    @ModuleInfo(key: "classifier") var classifier: Linear

    init(inChannels: Int = 80, hiddenChannels: Int = 512, numLayers: Int = 5) {
        var net = [Conv1dPT]()
        for i in 0..<numLayers {
            let inCh = i == 0 ? inChannels : hiddenChannels
            net.append(Conv1dPT(inCh, hiddenChannels, kernel: 3, padding: 1))
        }
        _condnet.wrappedValue = net
        _classifier.wrappedValue = Linear(hiddenChannels, 1)
        super.init()
    }

    /// mel (B, 80, T) → f0 (B, T)
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = mel
        for conv in condnet { x = eluAct(conv(x)) }
        x = x.transposed(0, 2, 1)                            // (B, T, C)
        var f0 = classifier(x)[0..., 0..., 0..<1]           // (B, T, 1)
        f0 = abs(f0)
        return f0[0..., 0..., 0..<1].squeezed(axis: 2)      // (B, T)
    }
}

// MARK: - real DFT bases (n_fft = 16)

private let NFFT = 16
/// cos/sin tables for rfft: [n_fft, n_fft/2+1] with cos[n][k]=cos(2πkn/N), sin likewise.
private let rfftCos: MLXArray = {
    let N = NFFT, K = N / 2 + 1
    var flat = [Float](repeating: 0, count: N * K)
    for n in 0..<N {
        for k in 0..<K {
            flat[n * K + k] = Float(cos(2 * .pi * Double(k) * Double(n) / Double(N)))
        }
    }
    return MLXArray(flat).reshaped([N, K])
}()
private let rfftSin: MLXArray = {
    let N = NFFT, K = N / 2 + 1
    var flat = [Float](repeating: 0, count: N * K)
    for n in 0..<N {
        for k in 0..<K {
            flat[n * K + k] = Float(sin(2 * .pi * Double(k) * Double(n) / Double(N)))
        }
    }
    return MLXArray(flat).reshaped([N, K])
}()

/// hanning periodic window of length N.
private func hanningPeriodic(_ N: Int) -> [Float] {
    (0..<N).map { n in 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(N)) }
}

// MARK: - HiFTGenerator

final class HiFTGenerator: Module {
    let samplingRate: Int
    let hopLen = 4
    let nFft = 16
    let nbHarmonics = 8
    let audioLimit: Float = 0.99
    let numUpsamples = 3
    let numKernels = 3
    let f0UpsampleScale: Int

    var f0Predictor: F0Predictor
    var mSource: SourceModule
    @ModuleInfo(key: "conv_pre") var convPre: Conv1dPT
    @ModuleInfo(key: "ups") var ups: [ConvTranspose1dPT]
    @ModuleInfo(key: "source_downs") var sourceDowns: [Conv1dPT]
    @ModuleInfo(key: "source_resblocks") var sourceResblocks: [HifiResBlock]
    @ModuleInfo(key: "resblocks") var resblocks: [HifiResBlock]
    @ModuleInfo(key: "conv_post") var convPost: Conv1dPT
    let stftWindow: MLXArray

    init(samplingRate: Int) {
        self.samplingRate = samplingRate
        let upsampleRates = [8, 5, 3]
        let upsampleKernels = [16, 11, 7]
        let resblockKernels = [3, 7, 11]
        let resblockDilations: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        let sourceResKernels = [7, 7, 11]
        let sourceResDilations: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        let baseCh = 512
        self.f0Predictor = F0Predictor()
        self.f0UpsampleScale = upsampleRates.reduce(1, *) * 4   // 120 * 4 = 480
        self.mSource = SourceModule(samplingRate: samplingRate, harmonicNum: 8,
                                    sineAmp: 0.1, noiseStd: 0.003, voicedThreshold: 10)
        self.stftWindow = MLXArray(hanningPeriodic(16))

        _convPre.wrappedValue = Conv1dPT(80, baseCh, kernel: 7, padding: 3)
        var up = [ConvTranspose1dPT]()
        for i in 0..<upsampleRates.count {
            let k = upsampleKernels[i], u = upsampleRates[i]
            up.append(ConvTranspose1dPT(baseCh / (1 << i), baseCh / (1 << (i + 1)),
                                        kernel: k, stride: u, padding: (k - u) / 2))
        }
        _ups.wrappedValue = up

        let downsampleRates = [1] + Array(upsampleRates.reversed().dropLast())  // [1,3,5]
        var sd = [Conv1dPT](), sr = [HifiResBlock]()
        let dcRev = [15, 3, 1]   // reversed cumprod([1,3,5])
        for i in 0..<3 {
            let u = dcRev[i], outCh = baseCh / (1 << (i + 1))
            if u == 1 {
                sd.append(Conv1dPT(18, outCh, kernel: 1))
            } else {
                sd.append(Conv1dPT(18, outCh, kernel: u * 2, stride: u, padding: u / 2))
            }
            sr.append(HifiResBlock(outCh, kernel: sourceResKernels[i], dilations: sourceResDilations[i]))
        }
        _sourceDowns.wrappedValue = sd
        _sourceResblocks.wrappedValue = sr

        var rb = [HifiResBlock]()
        for i in 0..<upsampleRates.count {
            let resCh = baseCh / (1 << (i + 1))
            for j in 0..<resblockKernels.count {
                rb.append(HifiResBlock(resCh, kernel: resblockKernels[j], dilations: resblockDilations[j]))
            }
        }
        _resblocks.wrappedValue = rb

        let finalCh = baseCh / (1 << upsampleRates.count)
        _convPost.wrappedValue = Conv1dPT(finalCh, 18, kernel: 7, padding: 3)
        _ = resblockKernels; _ = downsampleRates
        super.init()
    }

    /// STFT of the source signal s (B, T) -> (real, imag) each [B, 9, n_frames]. Pure MLX.
    private func stft(_ s: MLXArray) -> (MLXArray, MLXArray) {
        let B = s.dim(0), T = s.dim(1)
        var nFrames = (T - nFft) / hopLen + 1
        if nFrames < 1 { nFrames = 1 }
        // frame start indices -> [n_frames, 16]
        var idxFlat = [Int32](repeating: 0, count: nFrames * nFft)
        for f in 0..<nFrames {
            for j in 0..<nFft { idxFlat[f * nFft + j] = Int32(f * hopLen + j) }
        }
        let idx = MLXArray(idxFlat).reshaped([nFrames, nFft])
        var frames = take(s, idx, axis: 1)                  // [B, n_frames, 16]
        frames = frames * stftWindow                         // window
        // rfft via matmul: real = frames @ cos, imag = frames @ (-sin); cos/sin are [16,9]
        let realF = matmul(frames, rfftCos)                 // [B, n_frames, 9]
        let imagF = matmul(frames, -rfftSin)
        let real = realF.transposed(0, 2, 1)                 // [B, 9, n_frames]
        let imag = imagF.transposed(0, 2, 1)
        _ = B
        return (real, imag)
    }

    /// decode mel (B,80,T) + source s (B,1,T_audio) -> magnitude, phase (each [B,9,T]).
    private func decode(_ x: MLXArray, _ s: MLXArray) -> (MLXArray, MLXArray) {
        let (sReal, sImag) = stft(s[0..., 0, 0...])          // s[:,0,:] -> (B, T)
        let sStft = concatenated([sReal, sImag], axis: 1)    // [B, 18, n_frames]

        var h = convPre(x)
        for i in 0..<numUpsamples {
            h = leakyRelu(h, negativeSlope: 0.1)
            h = ups[i](h)
            if i == numUpsamples - 1 { h = padded(h, widths: [[0, 0], [0, 0], [1, 0]]) }
            var si = sourceDowns[i](sStft)
            si = sourceResblocks[i](si)
            let minLen = min(h.dim(2), si.dim(2))
            h = h[0..., 0..., 0..<minLen] + si[0..., 0..., 0..<minLen]
            var xs: MLXArray? = nil
            for j in 0..<numKernels {
                let idx = i * numKernels + j
                xs = xs == nil ? resblocks[idx](h) : xs! + resblocks[idx](h)
            }
            h = xs! / Float(numKernels)
        }
        h = leakyRelu(h)
        h = convPost(h)
        let half = nFft / 2 + 1
        let magnitude = exp(h[0..., 0..<half, 0...])
        let phase = sin(h[0..., half..<h.dim(1), 0...])
        return (magnitude, phase)
    }

    /// Inverse STFT (overlap-add) in plain Float. magnitude/phase [9, T] -> [T_audio].
    private func istft(_ magnitude: MLXArray, _ phase: MLXArray) -> [Float] {
        let mag = magnitude.asArray(Float.self)              // [9*T]
        let pha = phase.asArray(Float.self)
        let T = magnitude.dim(1)
        let N = nFft, H = hopLen
        // precompute bases (k=1..7)
        func c(_ k: Int, _ n: Int) -> Float { cos(2 * .pi * Float(k) * Float(n) / Float(N)) }
        func s(_ k: Int, _ n: Int) -> Float { sin(2 * .pi * Float(k) * Float(n) / Float(N)) }
        let outputLen = (T - 1) * H + N
        var audio = [Float](repeating: 0, count: outputLen)
        var winSum = [Float](repeating: 0, count: outputLen)
        let win = hanningPeriodic(N)
        for f in 0..<T {
            // reconstruct 16 samples for frame f
            for n in 0..<N {
                var acc: Float = mag[f * 9 + 0]                       // r0 (k=0)
                acc += mag[f * 9 + 8] * (n % 2 == 0 ? 1.0 : -1.0)    // r8 * (-1)^n (Nyquist)
                for k in 1..<8 {
                    let rk = mag[f * 9 + k] * cos(pha[f * 9 + k])
                    let ik = mag[f * 9 + k] * sin(pha[f * 9 + k])
                    acc += 2.0 * (rk * c(k, n) - ik * s(k, n))
                }
                acc /= Float(N)
                let pos = f * H + n
                audio[pos] += acc * win[n]
                winSum[pos] += win[n] * win[n]
            }
        }
        for i in 0..<outputLen { audio[i] /= max(winSum[i], 1e-8) }
        // trim center padding (matches torch.istft center=True)
        let pad = N / 2
        let expected = (T - 1) * H
        return Array(audio[pad..<(pad + expected)])
    }

    /// mel (B, 80, T) -> audio [Float] (mono, batch 0).
    func generate(_ mel: MLXArray) -> [Float] {
        let f0 = f0Predictor(mel)                              // (B, T)
        let f0Up = repeated(f0.expandedDimensions(axis: -1), count: f0UpsampleScale, axis: 1)  // (B, T*480, 1)
        let (sMerge, _) = mSource(f0Up)                        // (B, T_audio, 1)
        let s = sMerge.transposed(0, 2, 1)                     // (B, 1, T_audio)
        let (mag, pha) = decode(mel, s)
        eval(mag); eval(pha)
        let wav = istft(mag[0], pha[0])
        return wav.map { Swift.min(Swift.max($0, -audioLimit), audioLimit) }
    }
}
