import XCTest
@testable import Qwen3TTS

final class StreamMathTests: XCTestCase {
    func testCrossfadeBlendsEqualLengthArrays() {
        let prev: [Float] = [0, 0, 0, 0]
        let next: [Float] = [1, 1, 1, 1]
        let out = StreamMath.crossfade(prev: prev, next: next, overlap: 3)
        // first 3 samples ramp 0 -> ~1, 4th untouched
        XCTAssertEqual(out[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(out[1], 1.0 / 3.0, accuracy: 1e-6)
        XCTAssertEqual(out[2], 2.0 / 3.0, accuracy: 1e-6)
        XCTAssertEqual(out[3], 1.0, accuracy: 1e-6)
        XCTAssertEqual(out.count, 4)
    }

    func testCrossfadeClampsOverlapToShorterArray() {
        let out = StreamMath.crossfade(prev: [5, 5], next: [1, 1, 1, 1], overlap: 99)
        XCTAssertEqual(out.count, 4)
        // r clamped to 2; i=0 -> w=0 -> prev value (5)
        XCTAssertEqual(out[0], 5.0, accuracy: 1e-6)
        // i=1 -> w=0.5 -> blend (5*0.5 + 1*0.5 = 3)
        XCTAssertEqual(out[1], 3.0, accuracy: 1e-6)
        // beyond overlap: untouched next values
        XCTAssertEqual(out[2], 1.0, accuracy: 1e-6)
        XCTAssertEqual(out[3], 1.0, accuracy: 1e-6)
    }

    func testCrossfadeZeroOverlapReturnsNextUnchanged() {
        let out = StreamMath.crossfade(prev: [9, 9, 9], next: [1, 2, 3], overlap: 0)
        XCTAssertEqual(out, [1, 2, 3])
    }

    func testSamplesRangeForTokenSpanWithinWindow() {
        // tokens [2..<5) inside a window starting at token 1, 1920 samples/token
        let r = StreamMath.samples(forTokenRangeStart: 2, end: 5, windowStart: 1, samplesPerToken: 1920)
        XCTAssertEqual(r.lowerBound, (2 - 1) * 1920) // 1920
        XCTAssertEqual(r.upperBound, (5 - 1) * 1920) // 7680
    }
}
