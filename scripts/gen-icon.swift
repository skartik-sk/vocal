import CoreGraphics
import ImageIO
import Foundation

// Renders the Vocal app icon at a given size: graphite rounded square with a
// subtle vertical gradient, a hairline inner border, and a 4-bar waveform
// (steel-blue center bar). Classic/professional counterpart to the in-app SVG.
//
// Usage:  swift scripts/gen-icon.swift src-tauri/icons
// Then:   iconutil -c icns src-tauri/icons/Vocal.iconset -o src-tauri/icons/icon.icns
func render(size: Int) -> CGImage? {
    let s = CGFloat(size)
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(
        data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
        space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    ctx.clear(CGRect(x: 0, y: 0, width: s, height: s))

    // macOS-style canvas: ~9% transparent margin, squircle body.
    let m = s * 0.09
    let body = CGRect(x: m, y: m, width: s - 2 * m, height: s - 2 * m)
    let radius = body.width * 0.225
    let path = CGPath(roundedRect: body, cornerWidth: radius, cornerHeight: radius, transform: nil)

    ctx.addPath(path)
    ctx.clip()

    // Vertical graphite gradient.
    let colors = [
        CGColor(srgbRed: 0.185, green: 0.200, blue: 0.227, alpha: 1), // #2F333A
        CGColor(srgbRed: 0.110, green: 0.118, blue: 0.133, alpha: 1), // #1C1E22
    ] as CFArray
    let grad = CGGradient(colorsSpace: cs, colors: colors, locations: [0, 1])!
    ctx.drawLinearGradient(grad, start: CGPoint(x: 0, y: s), end: CGPoint(x: 0, y: 0), options: [])

    // Hairline inner border.
    ctx.setLineWidth(max(1, s * 0.006))
    ctx.setStrokeColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.10))
    ctx.addPath(path)
    ctx.strokePath()

    // Waveform bars: 4 bars, centered; center bar in steel blue.
    let barW = s * 0.075
    let gap = s * 0.048
    let heights: [CGFloat] = [0.185, 0.340, 0.520, 0.280]
    let totalW = barW * 4 + gap * 3
    var x = (s - totalW) / 2
    for (i, hfrac) in heights.enumerated() {
        let h = s * hfrac
        let bar = CGRect(x: x, y: (s - h) / 2, width: barW, height: h)
        ctx.setFillColor(i == 2
            ? CGColor(srgbRed: 0.561, green: 0.690, blue: 0.863, alpha: 1) // #8FB0DC
            : CGColor(srgbRed: 0.906, green: 0.910, blue: 0.918, alpha: 1)) // #E7E8EA
        ctx.addPath(CGPath(roundedRect: bar, cornerWidth: barW / 2, cornerHeight: barW / 2, transform: nil))
        ctx.fillPath()
        x += barW + gap
    }

    return ctx.makeImage()
}

func writePNG(_ img: CGImage, to url: URL) throws {
    let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, img, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw NSError(domain: "icon", code: 1, userInfo: [NSLocalizedDescriptionKey: "finalize failed for \(url)"])
    }
}

let outDir = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "."
let fm = FileManager.default

// iconset variants (macOS)
let variants: [(String, Int)] = [
    ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32), ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512), ("icon_512x512@2x.png", 1024),
]

let setDir = URL(fileURLWithPath: outDir).appendingPathComponent("Vocal.iconset")
try? fm.removeItem(at: setDir)
try fm.createDirectory(at: setDir, withIntermediateDirectories: true)

for (name, px) in variants {
    guard let img = render(size: px) else { fatalError("render \(px) failed") }
    try writePNG(img, to: setDir.appendingPathComponent(name))
}

// Standalone pngs used by README + Tauri fallback.
for (name, px) in [("icon.png", 1024), ("icon32.png", 32)] {
    guard let img = render(size: px) else { fatalError("render \(px) failed") }
    try writePNG(img, to: URL(fileURLWithPath: outDir).appendingPathComponent(name))
}

print("✅ rendered \(variants.count) iconset variants + icon.png + icon32.png into \(outDir)")
print("ℹ️  finish with: iconutil -c icns \(outDir)/Vocal.iconset -o \(outDir)/icon.icns")
