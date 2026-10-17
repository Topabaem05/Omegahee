// Omegahee Metal harness.
//
// Loads omegahee.metallib, runs the four parity kernels against the binary
// vectors from metal/tools/gen_vectors.py, and measures the achievable memory
// bandwidth of this device. Writes metal/build/report.json.
//
// Build and run: bash metal/run.sh

import Foundation
import Metal

// ---------------------------------------------------------------- util

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

struct Manifest: Codable {
    struct Case: Codable {
        let file: String
        let sha256: String
    }
    struct Cases: Codable {
        let ksc_decode_dot: Case
        let awsrc_dot: Case
        let readkv_prefix_load: Case
        let lilicorr_score: Case
    }
    let block: Int
    let tolerances: [String: Double]
    let cases: Cases
}

final class Reader {
    let data: Data
    private var offset = 0
    init(_ data: Data) { self.data = data }

    func u32() -> UInt32 {
        let value = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
        offset += 4
        return UInt32(littleEndian: value)
    }
    func u16() -> UInt16 {
        let value = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self) }
        offset += 2
        return UInt16(littleEndian: value)
    }
    func f32() -> Float {
        Float(bitPattern: u32())
    }
    func floatArray(_ count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count { out[i] = f32() }
        return out
    }
    func uintArray(_ count: Int) -> [UInt32] {
        var out = [UInt32](repeating: 0, count: count)
        for i in 0..<count { out[i] = u32() }
        return out
    }
    func ushortArray(_ count: Int) -> [UInt16] {
        var out = [UInt16](repeating: 0, count: count)
        for i in 0..<count { out[i] = u16() }
        return out
    }
    func bytes(_ count: Int) -> [UInt8] {
        let slice = [UInt8](data[offset..<(offset + count)])
        offset += count
        return slice
    }
}

func makeBuffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
    let length = max(MemoryLayout<T>.stride * values.count, 1)
    guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
        fail("failed to allocate buffer")
    }
    values.withUnsafeBytes { raw in
        buffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
    }
    return buffer
}

func readFloats(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
    let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: ptr, count: count))
}

func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Double {
    guard a.count == b.count else { return .infinity }
    var worst = 0.0
    for i in 0..<a.count {
        worst = max(worst, abs(Double(a[i]) - Double(b[i])))
    }
    return worst
}

// ---------------------------------------------------------------- main

let root = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()   // harness/
    .deletingLastPathComponent()   // metal/
let buildDir = root.appendingPathComponent("build")
let vectorDir = buildDir.appendingPathComponent("vectors")

guard let manifestData = try? Data(contentsOf: vectorDir.appendingPathComponent("manifest.json")),
      let manifest = try? JSONDecoder().decode(Manifest.self, from: manifestData) else {
    fail("missing vectors; run: python metal/tools/gen_vectors.py")
}
guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
guard let queue = device.makeCommandQueue() else { fail("no command queue") }

let libraryURL = buildDir.appendingPathComponent("omegahee.metallib")
guard let library = try? device.makeLibrary(URL: libraryURL) else {
    fail("cannot load omegahee.metallib; run: bash metal/run.sh")
}

func pipeline(_ name: String) -> MTLComputePipelineState {
    guard let fn = library.makeFunction(name: name) else { fail("missing kernel: \(name)") }
    do { return try device.makeComputePipelineState(function: fn) }
    catch { fail("pipeline \(name): \(error)") }
}

func dispatch(_ name: String, _ buffers: [MTLBuffer], _ constants: [(Int, Float)],
              _ constantsU: [(Int, UInt32)], _ width: Int, _ height: Int = 1) -> MTLBuffer? {
    let state = pipeline(name)
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
    encoder.setComputePipelineState(state)
    for (i, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: i) }
    for (index, value) in constants {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<Float>.stride, index: index)
    }
    for (index, value) in constantsU {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<UInt32>.stride, index: index)
    }
    let w = state.threadExecutionWidth
    let h = max(state.maxTotalThreadsPerThreadgroup / w, 1)
    let tg = MTLSize(width: min(w, width), height: min(h, height), depth: 1)
    encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                            threadsPerThreadgroup: tg)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return buffers.last
}

/// Dispatch with an explicit threadgroup size, for kernels whose lanes must map
/// onto one row (the SIMD decode family: 32 threads per row).
func dispatchFixedGroup(_ name: String, _ buffers: [MTLBuffer], _ constants: [(Int, Float)],
                        _ constantsU: [(Int, UInt32)], _ width: Int,
                        threadsPerGroup: Int) -> MTLBuffer? {
    let state = pipeline(name)
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
    encoder.setComputePipelineState(state)
    for (i, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: i) }
    for (index, value) in constants {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<Float>.stride, index: index)
    }
    for (index, value) in constantsU {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<UInt32>.stride, index: index)
    }
    encoder.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return buffers.last
}

/// Dispatch a fixed number of 256-thread groups, each covering up to 8 rows.
func dispatchGroupsShared(_ name: String, _ buffers: [MTLBuffer], _ constants: [(Int, Float)],
                          _ constantsU: [(Int, UInt32)], groups: Int,
                          threadsPerGroup: Int, sharedBytes: Int) -> MTLBuffer? {
    let state = pipeline(name)
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
    encoder.setComputePipelineState(state)
    for (i, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: i) }
    for (index, value) in constants {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<Float>.stride, index: index)
    }
    for (index, value) in constantsU {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<UInt32>.stride, index: index)
    }
    encoder.setThreadgroupMemoryLength(sharedBytes, index: 0)
    encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return buffers.last
}

/// Dispatch with explicit threadgroup size plus threadgroup memory.
func dispatchSharedFixedGroup(_ name: String, _ buffers: [MTLBuffer], _ constants: [(Int, Float)],
                              _ constantsU: [(Int, UInt32)], _ width: Int,
                              threadsPerGroup: Int, sharedBytes: Int) -> MTLBuffer? {
    let state = pipeline(name)
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
    encoder.setComputePipelineState(state)
    for (i, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: i) }
    for (index, value) in constants {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<Float>.stride, index: index)
    }
    for (index, value) in constantsU {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<UInt32>.stride, index: index)
    }
    encoder.setThreadgroupMemoryLength(sharedBytes, index: 0)
    encoder.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return buffers.last
}

/// Dispatch with a threadgroup-memory pointer at index 8 for the KSC decode
/// family, which stages beta and x in 2 * OMEGA_BLOCK floats.
func dispatchShared(_ name: String, _ buffers: [MTLBuffer], _ constants: [(Int, Float)],
                    _ constantsU: [(Int, UInt32)], _ width: Int,
                    sharedBytes: Int) -> MTLBuffer? {
    let state = pipeline(name)
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else { return nil }
    encoder.setComputePipelineState(state)
    for (i, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: i) }
    for (index, value) in constants {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<Float>.stride, index: index)
    }
    for (index, value) in constantsU {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<UInt32>.stride, index: index)
    }
    encoder.setThreadgroupMemoryLength(sharedBytes, index: 0)
    let w = state.threadExecutionWidth
    let h = max(state.maxTotalThreadsPerThreadgroup / w, 1)
    encoder.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return buffers.last
}

var report: [String: Any] = [:]
var parity: [String: Any] = [:]
var pass = true

// ---- ksc_decode_dot
do {
    let caseInfo = manifest.cases.ksc_decode_dot
    let payload = try! Data(contentsOf: vectorDir.appendingPathComponent(caseInfo.file))
    let r = Reader(payload)
    let rows = Int(r.u32())
    let scale = r.f32()
    let s0 = r.ushortArray(rows)
    let codes = r.bytes(rows * manifest.block / 4)
    let alpha = r.floatArray(rows)
    let beta = r.floatArray(manifest.block)
    let x = r.floatArray(manifest.block)
    let expected = r.floatArray(rows)

    let outBuffer = device.makeBuffer(length: rows * 4, options: .storageModeShared)!
    var out = [MTLBuffer]()
    out.append(makeBuffer(device, s0))
    out.append(makeBuffer(device, codes))
    out.append(makeBuffer(device, alpha))
    out.append(makeBuffer(device, beta))
    out.append(makeBuffer(device, x))
    out.append(outBuffer)
    _ = dispatch("ksc_decode_dot", out, [(6, scale)], [(7, UInt32(rows))], rows)
    let got = readFloats(outBuffer, rows)
    let diff = maxAbsDiff(got, expected)
    let tol = manifest.tolerances["ksc_decode_dot"]!
    pass = pass && diff <= tol
    parity["ksc_decode_dot"] = ["max_abs_diff": diff, "tolerance": tol, "pass": diff <= tol]
}

// ---- ksc_decode_simd (same vectors, lane-parallel decode)
do {
    let caseInfo = manifest.cases.ksc_decode_dot
    let payload = try! Data(contentsOf: vectorDir.appendingPathComponent(caseInfo.file))
    let r = Reader(payload)
    let rows = Int(r.u32())
    let scale = r.f32()
    let s0 = r.ushortArray(rows)
    let codes = r.bytes(rows * manifest.block / 4)
    let alpha = r.floatArray(rows)
    let beta = r.floatArray(manifest.block)
    let x = r.floatArray(manifest.block)
    let expected = r.floatArray(rows)

    let outBuffer = device.makeBuffer(length: rows * 4, options: .storageModeShared)!
    let buffers = [makeBuffer(device, s0), makeBuffer(device, codes), makeBuffer(device, alpha),
                   makeBuffer(device, beta), makeBuffer(device, x), outBuffer]
    _ = dispatchSharedFixedGroup("ksc_decode_simd", buffers, [(6, scale)], [(7, UInt32(rows))],
                                 rows * 32, threadsPerGroup: 256,
                                 sharedBytes: 2 * manifest.block * MemoryLayout<Float>.stride)
    let got = readFloats(outBuffer, rows)
    let diff = maxAbsDiff(got, expected)
    // Lane-parallel accumulation changes the summation order, so the tolerance
    // is wider than the scalar kernel but still tight for float32 over 64 terms.
    let tol = 2e-4
    pass = pass && diff <= tol
    parity["ksc_decode_simd"] = ["max_abs_diff": diff, "tolerance": tol, "pass": diff <= tol]
}

// ---- awsrc_dot
do {
    let caseInfo = manifest.cases.awsrc_dot
    let payload = try! Data(contentsOf: vectorDir.appendingPathComponent(caseInfo.file))
    let r = Reader(payload)
    let rows = Int(r.u32())
    let mSeeds = Int(r.u32())
    let invBlock = r.f32()
    let seeds = r.uintArray(mSeeds)
    let gammas = r.floatArray(rows * mSeeds)
    let x = r.floatArray(manifest.block)
    let expected = r.floatArray(rows)

    let outBuffer = device.makeBuffer(length: rows * 4, options: .storageModeShared)!
    let buffers = [makeBuffer(device, seeds), makeBuffer(device, gammas), makeBuffer(device, x), outBuffer]
    _ = dispatch("awsrc_dot", buffers, [(6, invBlock)],
                 [(4, UInt32(rows)), (5, UInt32(mSeeds))], rows)
    let got = readFloats(outBuffer, rows)
    let diff = maxAbsDiff(got, expected)
    let tol = manifest.tolerances["awsrc_dot"]!
    pass = pass && diff <= tol
    parity["awsrc_dot"] = ["max_abs_diff": diff, "tolerance": tol, "pass": diff <= tol]
}

// ---- readkv_prefix_load
do {
    let caseInfo = manifest.cases.readkv_prefix_load
    let payload = try! Data(contentsOf: vectorDir.appendingPathComponent(caseInfo.file))
    let r = Reader(payload)
    let channels = Int(r.u32())
    let budgets = r.bytes(channels)
    let planes = r.uintArray(8)
    let expected = r.uintArray(channels)

    let outBuffer = device.makeBuffer(length: channels * 4, options: .storageModeShared)!
    let buffers = [makeBuffer(device, planes), makeBuffer(device, budgets), outBuffer]
    _ = dispatch("readkv_prefix_load", buffers, [], [(3, UInt32(channels))], channels)
    let ptr = outBuffer.contents().bindMemory(to: UInt32.self, capacity: channels)
    let got = Array(UnsafeBufferPointer(start: ptr, count: channels))
    let mismatches = zip(got, expected).reduce(0) { $0 + ($1.0 == $1.1 ? 0 : 1) }
    let tol = manifest.tolerances["readkv_prefix_load"]!
    pass = pass && Double(mismatches) <= tol
    parity["readkv_prefix_load"] = [
        "mismatched_entries": mismatches,
        "channels": channels,
        "pass": Double(mismatches) <= tol,
    ]
}

// ---- lilicorr_score
do {
    let caseInfo = manifest.cases.lilicorr_score
    let payload = try! Data(contentsOf: vectorDir.appendingPathComponent(caseInfo.file))
    let r = Reader(payload)
    let groups = Int(r.u32())
    let k = Int(r.u32())
    let dCorr = Int(r.u32())
    let pairs = Int(r.u32())
    let tau = r.f32()
    let uo = r.floatArray(groups * k * dCorr)
    let ui = r.floatArray(groups * k * dCorr)
    let expected = r.floatArray(pairs * k * k)

    let outBuffer = device.makeBuffer(length: pairs * k * k * 4, options: .storageModeShared)!
    let buffers = [makeBuffer(device, uo), makeBuffer(device, ui), outBuffer]
    _ = dispatch("lilicorr_score", buffers, [(6, tau)],
                 [(3, UInt32(pairs)), (4, UInt32(k)), (5, UInt32(dCorr))], pairs, k * k)
    let got = readFloats(outBuffer, pairs * k * k)
    let diff = maxAbsDiff(got, expected)
    let tol = manifest.tolerances["lilicorr_score"]!
    pass = pass && diff <= tol
    parity["lilicorr_score"] = ["max_abs_diff": diff, "tolerance": tol, "pass": diff <= tol]
}

// ---- stream_probe (achievable bandwidth)
var bandwidth: [String: Any] = [:]
// ---- ksc_throughput (real trellis decode rate)
var kscThroughput: [String: Any] = [:]
do {
    // 1 MiB of packed weights per pass: 64 Ki block rows. Larger than any SLC
    // slice for one pass, small enough to hold several passes in flight.
    let rows = 1 << 16
    let passes: UInt32 = 64
    var rng = SystemRandomNumberGenerator()
    var s0 = [UInt16](repeating: 0, count: rows)
    for i in 0..<rows { s0[i] = UInt16(truncatingIfNeeded: rng.next() as UInt64) }
    var codes = [UInt8](repeating: 0, count: rows * manifest.block / 4)
    for i in 0..<codes.count { codes[i] = UInt8(truncatingIfNeeded: rng.next() as UInt64) }
    var alpha = [Float](repeating: 1.0, count: rows)
    for i in 0..<rows { alpha[i] = Float.random(in: 0.5...1.5) }
    let beta = (0..<manifest.block).map { _ in Float.random(in: -1...1) }
    let x = (0..<manifest.block).map { _ in Float.random(in: -1...1) }

    let outBuffer = device.makeBuffer(length: rows * 4, options: .storageModeShared)!
    let buffers = [makeBuffer(device, s0), makeBuffer(device, codes), makeBuffer(device, alpha),
                   makeBuffer(device, beta), makeBuffer(device, x), outBuffer]
    let state = pipeline("ksc_throughput")

    func once() -> Double {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
        enc.setComputePipelineState(state)
        for (i, buffer) in buffers.enumerated() { enc.setBuffer(buffer, offset: 0, index: i) }
        var scale: Float = 1.0
        var r32 = UInt32(rows)
        var p32 = passes
        enc.setBytes(&scale, length: 4, index: 6)
        enc.setBytes(&r32, length: 4, index: 7)
        enc.setBytes(&p32, length: 4, index: 9)
        enc.setThreadgroupMemoryLength(2 * manifest.block * MemoryLayout<Float>.stride, index: 0)
        let w = state.threadExecutionWidth
        let h = max(state.maxTotalThreadsPerThreadgroup / w, 1)
        enc.dispatchThreads(MTLSize(width: rows, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
        enc.endEncoding()
        let start = DispatchTime.now().uptimeNanoseconds
        cb.commit()
        cb.waitUntilCompleted()
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }
    _ = once(); _ = once()
    var samples = [Double]()
    for _ in 0..<9 { samples.append(once()) }
    samples.sort()
    let best = samples.first!
    let median = samples[samples.count / 2]
    // 18 bytes of weight payload per row: 16 code bytes, 2 header bytes.
    // alpha/beta/x are activation-side and deliberately not counted.
    let weightBytes = Double(rows) * 18.0 * Double(passes)
    kscThroughput["rows"] = rows
    kscThroughput["passes"] = Int(passes)
    kscThroughput["weight_bytes"] = Int(weightBytes)
    kscThroughput["gb_per_s_best"] = weightBytes / best / 1e9
    kscThroughput["gb_per_s_median"] = weightBytes / median / 1e9
    kscThroughput["samples"] = samples.count

    // Same weight stream through the lane-parallel kernel, so the two numbers
    // differ only by decode strategy.
    let simdState = pipeline("ksc_decode_simd")
    func simdOnce() -> Double {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
        enc.setComputePipelineState(simdState)
        for (i, buffer) in buffers.enumerated() { enc.setBuffer(buffer, offset: 0, index: i) }
        var scale: Float = 1.0
        var r32 = UInt32(rows)
        enc.setBytes(&scale, length: 4, index: 6)
        enc.setBytes(&r32, length: 4, index: 7)
        enc.setThreadgroupMemoryLength(2 * manifest.block * MemoryLayout<Float>.stride, index: 0)
        enc.dispatchThreads(MTLSize(width: rows * 32, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        let start = DispatchTime.now().uptimeNanoseconds
        cb.commit()
        cb.waitUntilCompleted()
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }
    _ = simdOnce(); _ = simdOnce()
    var simdSamples = [Double]()
    for _ in 0..<9 { simdSamples.append(simdOnce()) }
    simdSamples.sort()
    let simdBytes = Double(rows) * 18.0
    kscThroughput["simd_gb_per_s_best"] = simdBytes / simdSamples.first! / 1e9
    kscThroughput["simd_gb_per_s_median"] = simdBytes / simdSamples[simdSamples.count / 2] / 1e9
    kscThroughput["scalar_gb_per_s_best"] = weightBytes / best / 1e9
    kscThroughput["speedup_vs_scalar"] = (simdBytes / simdSamples.first!) / (weightBytes / best)

}

do {
    // 512 MiB is well past the SLC, so the probe measures DRAM streaming rather
    // than cache hits. Four passes keep the run near 20 ms, long enough to beat
    // launch noise without inviting thermal drift.
    let bytes = 512 << 20
    let words = bytes / 16
    let passes: UInt32 = 4
    guard let src = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
        fail("stream buffer allocation failed")
    }
    // More threads and a deeper grid keep the device saturated; the earlier
    // 64K-thread version came in below what uzu achieves in real decode.
    let threads = 1 << 18
    let outBuffer = device.makeBuffer(length: threads * 4, options: .storageModeShared)!
    let state = pipeline("stream_probe")
    // warm-up
    func once() -> Double {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
        enc.setComputePipelineState(state)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(outBuffer, offset: 0, index: 1)
        var w = UInt64(words)
        var s = UInt64(threads)
        var p = passes
        enc.setBytes(&w, length: 8, index: 2)
        enc.setBytes(&s, length: 8, index: 3)
        enc.setBytes(&p, length: 4, index: 4)
        let w2 = state.threadExecutionWidth
        let h2 = max(state.maxTotalThreadsPerThreadgroup / w2, 1)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w2, height: h2, depth: 1))
        enc.endEncoding()
        let start = DispatchTime.now().uptimeNanoseconds
        cb.commit()
        cb.waitUntilCompleted()
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }
    for _ in 0..<3 { _ = once() }
    var samples = [Double]()
    for _ in 0..<15 { samples.append(once()) }
    samples.sort()
    let best = samples.first!
    let median = samples[samples.count / 2]
    let worst = samples.last!
    let streamed = Double(bytes) * Double(passes)
    let gbs = streamed / best / 1e9
    let gbsMedian = streamed / median / 1e9
    let gbsWorst = streamed / worst / 1e9
    bandwidth["stream_gb_per_s"] = gbs
    bandwidth["stream_gb_per_s_median"] = gbsMedian
    bandwidth["stream_gb_per_s_worst"] = gbsWorst
    bandwidth["stream_bytes"] = bytes
    bandwidth["stream_passes"] = Int(passes)
    bandwidth["stream_samples"] = samples.count
    bandwidth["stream_best_seconds"] = best
    bandwidth["device"] = device.name

    // reuse the probe at the 2.38075 bpw decode geometry
    let bpw = 2.38075
    let params9b = 9.0e9
    let weightBytes = params9b * bpw / 8.0
    bandwidth["roofline_weight_bytes_for_9b_238bpw"] = weightBytes
    bandwidth["roofline_tps_ceiling_9b"] = gbs * 1e9 / weightBytes
}

// ---- ksc_stream_simd (DRAM-scale, single pass): comparable to a real decode
var streamDecode: [String: Any] = [:]
do {
    // 1M block rows = 16 MiB of codes, far beyond L2 on this part, so the pass
    // is DRAM-bound the way a 5-10 GB checkpoint decode is.
    let rows = 1 << 20
    var rng = SystemRandomNumberGenerator()
    var s0 = [UInt16](repeating: 0, count: rows)
    for i in 0..<rows { s0[i] = UInt16(truncatingIfNeeded: rng.next() as UInt64) }
    var codes = [UInt8](repeating: 0, count: rows * manifest.block / 4)
    for i in 0..<codes.count { codes[i] = UInt8(truncatingIfNeeded: rng.next() as UInt64) }
    var alpha = [Float](repeating: 1.0, count: rows)
    for i in 0..<rows { alpha[i] = Float.random(in: 0.5...1.5) }
    let beta = (0..<manifest.block).map { _ in Float.random(in: -1...1) }
    let x = (0..<manifest.block).map { _ in Float.random(in: -1...1) }

    let outBuffer = device.makeBuffer(length: rows * 4, options: .storageModeShared)!
    let buffers = [makeBuffer(device, s0), makeBuffer(device, codes), makeBuffer(device, alpha),
                   makeBuffer(device, beta), makeBuffer(device, x), outBuffer]
    let state = pipeline("ksc_stream_simd")
    // 8 rows per group, so one group covers a strip; 256 threads = 8 groups.
    let width = (rows / 8) * 32

    func once() -> Double {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
        enc.setComputePipelineState(state)
        for (i, buffer) in buffers.enumerated() { enc.setBuffer(buffer, offset: 0, index: i) }
        var scale: Float = 1.0
        var r32 = UInt32(rows)
        enc.setBytes(&scale, length: 4, index: 6)
        enc.setBytes(&r32, length: 4, index: 7)
        enc.setThreadgroupMemoryLength(2 * manifest.block * MemoryLayout<Float>.stride, index: 0)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        let start = DispatchTime.now().uptimeNanoseconds
        cb.commit()
        cb.waitUntilCompleted()
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }
    _ = once(); _ = once()
    var samples = [Double]()
    for _ in 0..<7 { samples.append(once()) }
    samples.sort()
    let bytes = Double(rows) * 18.0
    streamDecode["rows"] = rows
    streamDecode["code_bytes"] = rows * manifest.block / 4
    streamDecode["weight_bytes_per_pass"] = Int(bytes)
    streamDecode["gb_per_s_best"] = bytes / samples.first! / 1e9
    streamDecode["gb_per_s_median"] = bytes / samples[samples.count / 2] / 1e9
    streamDecode["samples"] = samples.count
}

// ---- ksc_gemv_coalesced: same math, fully coalesced loads
var coalesced: [String: Any] = [:]
do {
    // Small deterministic case checked against a CPU reference, so a wrong
    // kernel cannot be excused as a dispatch artifact.
    let nRows = 1
    let kCols = 256
    let nBlocks = kCols / manifest.block      // 4 blocks
    let codeBytes = nRows * nBlocks * (manifest.block / 4)
    var codes = [UInt8](repeating: 0, count: codeBytes)
    var seed: UInt64 = 0x12345678
    for i in 0..<codeBytes {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        codes[i] = UInt8(truncatingIfNeeded: seed >> 33)
    }
    let alpha = (0..<nRows).map { Float(0.5 + 0.1 * Double($0)) }
    let x = (0..<kCols).map { Float(sin(Double($0) * 0.37) * 1.5) }
    let span: [Float] = [-3, -1, 1, 3]
    var expected = [Float](repeating: 0, count: nRows)
    for row in 0..<nRows {
        var sum: Float = 0
        for k in 0..<kCols {
            let byte = codes[row * nBlocks * 16 + (k >> 2)]
            let code = (UInt32(byte) >> (6 - 2 * (k & 3))) & 3
            sum += x[k] * (0.5 * 1.0 * span[Int(code)])
        }
        expected[row] = alpha[row] * sum
    }

    let outBuffer = device.makeBuffer(length: 4096, options: .storageModeShared)!
    let buffers = [makeBuffer(device, codes), makeBuffer(device, alpha), makeBuffer(device, x), outBuffer]
    _ = dispatchGroupsShared("ksc_gemv_coalesced", buffers, [(4, 1.0)],
                             [(5, UInt32(nRows)), (6, UInt32(nBlocks))],
                             groups: (nRows + 7) / 8, threadsPerGroup: 256,
                             sharedBytes: kCols * MemoryLayout<Float>.stride)
    let got = readFloats(outBuffer, nRows)
    var worst = 0.0
    for i in 0..<nRows { worst = max(worst, abs(Double(got[i]) - Double(expected[i]))) }
    let tol = 1e-3
    // This kernel is UNVERIFIED: its output does not yet match the CPU reference.
    // It is recorded but deliberately kept out of the parity gate so an
    // unverified optimization cannot masquerade as a passing result.
    coalesced["cpu_parity"] = ["max_abs_diff": worst, "tolerance": tol,
                               "pass": worst <= tol, "verified": false,
                               "note": "excluded from parity gate until it matches"]

    // Throughput at the production shape, measured separately.
    let bigRows = 12288
    let bigBlocks = 64
    var bigRng = SystemRandomNumberGenerator()
    var bigCodes = [UInt8](repeating: 0, count: bigRows * bigBlocks * 16)
    for i in 0..<bigCodes.count { bigCodes[i] = UInt8(truncatingIfNeeded: bigRng.next() as UInt64) }
    let bigX = (0..<4096).map { _ in Float.random(in: -1...1) }
    let bigOut = device.makeBuffer(length: bigRows * 4, options: .storageModeShared)!
    let bigBuffers = [makeBuffer(device, bigCodes), makeBuffer(device, alpha.isEmpty ? [Float](repeating: 1, count: bigRows) : [Float](repeating: 1, count: bigRows)), makeBuffer(device, bigX), bigOut]
    let state = pipeline("ksc_gemv_coalesced")
    let width = bigRows * 32
    func once() -> Double {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
        enc.setComputePipelineState(state)
        for (i, buffer) in bigBuffers.enumerated() { enc.setBuffer(buffer, offset: 0, index: i) }
        var scale: Float = 1.0
        var n32 = UInt32(bigRows)
        var b32 = UInt32(bigBlocks)
        enc.setBytes(&scale, length: 4, index: 4)
        enc.setBytes(&n32, length: 4, index: 5)
        enc.setBytes(&b32, length: 4, index: 6)
        enc.setThreadgroupMemoryLength(4096 * MemoryLayout<Float>.stride, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: (bigRows + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        let start = DispatchTime.now().uptimeNanoseconds
        cb.commit()
        cb.waitUntilCompleted()
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }
    _ = once(); _ = once()
    var samples = [Double]()
    for _ in 0..<9 { samples.append(once()) }
    samples.sort()
    let weightBytes = Double(bigRows) * Double(bigBlocks) * 18.0
    coalesced["gb_per_s_best"] = weightBytes / samples.first! / 1e9
    coalesced["gb_per_s_median"] = weightBytes / samples[samples.count / 2] / 1e9
    coalesced["samples"] = samples.count
}

// ---- ksc_gemv_k4096: uzu-shape GEMV (N=12288, K=4096) for a direct comparison
var gemv: [String: Any] = [:]
do {
    // Qwen3.5-9B shapes: model_dim 4096, hidden_dim 12288. One MLP up-projection
    // is K=4096 -> N=12288, i.e. 48 MiB of KSC codes at 18 bytes per 64 columns.
    let nRows = 12288
    let kCols = 4096
    let nBlocks = kCols / manifest.block
    let codeBytes = nRows * nBlocks * (manifest.block / 4)
    var rng = SystemRandomNumberGenerator()
    var codes = [UInt8](repeating: 0, count: codeBytes)
    for i in 0..<codeBytes { codes[i] = UInt8(truncatingIfNeeded: rng.next() as UInt64) }
    var alpha = [Float](repeating: 1.0, count: nRows)
    for i in 0..<nRows { alpha[i] = Float.random(in: 0.5...1.5) }
    let x = (0..<kCols).map { _ in Float.random(in: -1...1) }
    let outBuffer = device.makeBuffer(length: nRows * 4, options: .storageModeShared)!

    let buffers = [makeBuffer(device, codes), makeBuffer(device, alpha), makeBuffer(device, x), outBuffer]
    let state = pipeline("ksc_gemv_k4096")
    let width = nRows * 32

    func once() -> Double {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
        enc.setComputePipelineState(state)
        for (i, buffer) in buffers.enumerated() { enc.setBuffer(buffer, offset: 0, index: i) }
        var scale: Float = 1.0
        var n32 = UInt32(nRows)
        var b32 = UInt32(nBlocks)
        enc.setBytes(&scale, length: 4, index: 4)
        enc.setBytes(&n32, length: 4, index: 5)
        enc.setBytes(&b32, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        let start = DispatchTime.now().uptimeNanoseconds
        cb.commit()
        cb.waitUntilCompleted()
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }
    _ = once(); _ = once()
    var samples = [Double]()
    for _ in 0..<7 { samples.append(once()) }
    samples.sort()
    // Weight payload for this projection: 64 code bytes... 16 bytes per 64-col
    // block, plus 2 header bytes per block, times N rows.
    let weightBytes = Double(nRows) * Double(nBlocks) * 18.0
    gemv["n_rows"] = nRows
    gemv["k_cols"] = kCols
    gemv["n_blocks"] = nBlocks
    gemv["code_bytes"] = codeBytes
    gemv["weight_bytes"] = Int(weightBytes)
    gemv["gb_per_s_best"] = weightBytes / samples.first! / 1e9
    gemv["gb_per_s_median"] = weightBytes / samples[samples.count / 2] / 1e9
    gemv["samples"] = samples.count
}

report["device"] = device.name
report["parity"] = parity
report["parity_all_pass"] = pass
report["bandwidth"] = bandwidth
report["ksc_throughput"] = kscThroughput
report["stream_decode"] = streamDecode
report["gemv_k4096"] = gemv
report["gemv_coalesced"] = coalesced
report["generated_at_utc"] = ISO8601DateFormatter().string(from: Date())

let outURL = buildDir.appendingPathComponent("report.json")
let json = try! JSONSerialization.data(withJSONObject: report,
                                       options: [.prettyPrinted, .sortedKeys])
try! json.write(to: outURL)
print(String(data: json, encoding: .utf8)!)
if !pass { exit(2) }
