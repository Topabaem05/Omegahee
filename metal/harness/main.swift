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
    let threads = 1 << 16
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

report["device"] = device.name
report["parity"] = parity
report["parity_all_pass"] = pass
report["bandwidth"] = bandwidth
report["generated_at_utc"] = ISO8601DateFormatter().string(from: Date())

let outURL = buildDir.appendingPathComponent("report.json")
let json = try! JSONSerialization.data(withJSONObject: report,
                                       options: [.prettyPrinted, .sortedKeys])
try! json.write(to: outURL)
print(String(data: json, encoding: .utf8)!)
if !pass { exit(2) }
