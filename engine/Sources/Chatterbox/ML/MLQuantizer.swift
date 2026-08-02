//
//  MLQuantizer.swift — load packed 4-bit weights directly into QuantizedLinear /
//  QuantizedEmbedding modules (chatterbox-4bit uses per-tensor group sizes).
//
//  MLXNN's quantize() packs float weights itself; here the checkpoint already
//  ships packed uint32 weights + scales + biases, so we build the quantized
//  modules straight from those arrays via the arrays-in init.
//

import Foundation
import MLX
import MLXNN

/// Walk `model`'s leaves and replace Linear/Embedding modules whose weight key
/// exists in `packed` with a QuantizedLinear / QuantizedEmbedding holding the
/// packed weight + scales + biases from the checkpoint.
///
/// - Parameters:
///   - packed: renamed key -> all tensors (weight, scales, biases, bias)
func applyQuantized(
    model: Module,
    tensors: [String: MLXArray],
    groupSizeMap: [String: Int],
    bitsMap: [String: Int]
) throws {
    let leaves = model.leafModules().flattened()
    var updates: [(String, Module)] = []
    for (path, module) in leaves {
        guard let g = groupSizeMap[path], let b = bitsMap[path] else { continue }
        guard let w = tensors[path + ".weight"], let sc = tensors[path + ".scales"] else {
            continue
        }
        // `.biases` = quantization biases (rows, nGroups); `.bias` = linear bias (rows).
        let quantBiases = tensors[path + ".biases"]
        let linearBias = tensors[path + ".bias"]
        if let linear = module as? Linear, !(module is QuantizedLinear) {
            let q = QuantizedLinear(
                weight: w, bias: linearBias, scales: sc, biases: quantBiases,
                groupSize: g, bits: b, mode: .affine)
            updates.append((path, q))
        }
        // Embeddings are left as plain Embedding; the loader dequantizes their packed
        // weights to float and pours them via update(parameters:).
    }
    model.update(modules: ModuleChildren.unflattened(updates))
}
