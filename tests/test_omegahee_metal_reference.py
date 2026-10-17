"""CI checks for the Metal-pack reference semantics (numpy side).

The Metal kernels and the Swift harness run on Apple Silicon only. These tests
pin the numpy interpretation so a Linux CI runner still catches contract drift
before the GPU harness is exercised.
"""
import importlib.util
import math
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "omegahee_metal_reference", os.path.join(_ROOT, "src", "omegahee_metal_reference.py"))
M = importlib.util.module_from_spec(_spec)
sys.modules["omegahee_metal_reference"] = M
_spec.loader.exec_module(M)


class TestPackLayout(unittest.TestCase):
    def test_pack_2bit_roundtrip_msb_first(self):
        codes = [0, 1, 2, 3] * 16
        packed = M.pack_2bit(codes)
        self.assertEqual(len(packed), 16)
        self.assertEqual(packed[0], 0b00011011)
        bits = ''.join(f"{byte:08b}" for byte in packed)
        self.assertEqual(bits[:8], "00011011")

    def test_bit_reader_is_msb_first(self):
        self.assertEqual(M._bit(bytes([0b10000000]), 0), 1)
        self.assertEqual(M._bit(bytes([0b10000000]), 1), 0)
        self.assertEqual(M._bit(bytes([0b00000001]), 7), 1)
        self.assertEqual(M._bit(bytes([0b00000001]), 8), 0)  # past the row
        self.assertEqual(M._pair(bytes([0b11000000]), 0), 3)


class TestKscDecode(unittest.TestCase):
    def test_header_carries_first_seven_codes(self):
        # s0 bit 13 is stream bit 0, so the decode chunks (bits 13:12, 11:10, ...)
        # reproduce stream pairs at offsets 0, 2, ..., 12: the first seven codes.
        # The loop then shifts in pairs from offset 14 onward, which are codes 7+
        # and are therefore not duplicated in the header.
        case = M.make_ksc_case(rows=4, seed=21)
        for r in range(4):
            state = int(case["s0"][r])
            header_codes = [(state >> shift) & 0x3 for shift in (12, 10, 8, 6, 4, 2, 0)]
            packed = case["codes"][r]
            stream_lo = [M._pair(packed, 2 * j) for j in range(7)]
            stream_hi = [M._pair(packed, M.HEADER_BITS + 2 * j) for j in range(7)]
            self.assertEqual(header_codes, stream_lo, "header mirrors stream bits 0..13")
            self.assertNotEqual(header_codes, stream_hi, "header must not mirror codes 7..13")

    def test_manual_dot_matches_batch_helper(self):
        case = M.make_ksc_case(rows=3, seed=22)
        lut = M.codebook(float(case["scale"]))
        for r in range(3):
            manual = 0.0
            state = int(case["s0"][r])
            for j in range(M.BLOCK):
                code = (state >> 12) & 0x3
                manual += float(case["alpha"][r]) * float(case["beta"][j]) * float(lut[code]) * float(case["x"][j])
                state = ((state << 2) | M._pair(case["codes"][r], M.HEADER_BITS + 2 * j)) & M.CODE_MASK
            got = M.ksc_dot(int(case["s0"][r]), case["codes"][r], float(case["alpha"][r]),
                            case["beta"], case["x"], float(case["scale"]))
            self.assertAlmostEqual(got, manual, places=12)

    def test_shift_past_stream_contributes_zero(self):
        # The last seven iterations (j = 57..63) shift past the 128-bit window.
        # Those shifts must feed zero, never an earlier column.
        late = [j for j in range(M.BLOCK) if M.HEADER_BITS + 2 * j >= M.STREAM_BITS]
        self.assertEqual(late, list(range(57, 64)))

        # Drive the tail with an all-zero committed state: the columns after the
        # window then carry only the header-derived zero codes, so a zero header
        # and an all-zero beta/x cannot produce a non-zero tail contribution.
        state = 0
        zeros = bytes(16)
        for j in range(M.BLOCK):
            state = ((state << 2) | M._pair(zeros, M.HEADER_BITS + 2 * j)) & M.CODE_MASK
        self.assertEqual(state, 0)

        # A row whose stream is all ones must not leak past the window either:
        # both sides of the port read zero there by construction.
        ones = bytes([0xFF] * 16)
        self.assertEqual(M._pair(ones, 128), 0)
        self.assertEqual(M._pair(ones, 126), 3)


class TestAwsrc(unittest.TestCase):
    def test_splitmix_is_deterministic_and_balanced(self):
        first = [M.splitmix_sign(1234, i) for i in range(512)]
        second = [M.splitmix_sign(1234, i) for i in range(512)]
        self.assertEqual(first, second)
        ones = sum(1 for v in first if v > 0)
        self.assertLess(abs(ones - 256), 90)  # loose balance sanity, not a quality test

    def test_zero_scales_give_zero_correction(self):
        case = M.make_awsrc_case(rows=4, seed=23)
        zeros = np.zeros_like(case["gammas"])
        out = M.awsrc_dot_batch(case["seeds"], zeros, case["x"])
        self.assertTrue(np.allclose(out, 0.0))


class TestReadkv(unittest.TestCase):
    def test_prefix_semantics(self):
        # Bit 0 of a plane is channel 31, so use channel 31 to keep the shifts
        # readable. Plane p carries weight 2^(7-p).
        planes = np.zeros(8, dtype=np.uint32)
        planes[0] = 1 << 31  # plane 0, bit 31 -> weight 2^7
        planes[2] = 1 << 31  # plane 2, bit 31 -> weight 2^5
        self.assertEqual(M.readkv_prefix_value(planes, 31, 0), 0)
        self.assertEqual(M.readkv_prefix_value(planes, 31, 2), 0b10000000)
        self.assertEqual(M.readkv_prefix_value(planes, 31, 4), 0b10100000)
        self.assertEqual(M.readkv_prefix_value(planes, 31, 8), 0b10100000)

    def test_unread_planes_never_contribute(self):
        planes = np.full(8, 0xFFFFFFFF, dtype=np.uint32)
        self.assertEqual(M.readkv_prefix_value(planes, 5, 0), 0)
        self.assertEqual(M.readkv_prefix_value(planes, 5, 2), 0b11000000)
        self.assertEqual(M.readkv_prefix_value(planes, 5, 4), 0b11110000)
        self.assertEqual(M.readkv_prefix_value(planes, 5, 8), 0xFF)

    def test_budget_grid_rejected(self):
        with self.assertRaises(ValueError):
            M.readkv_prefix_value(np.zeros(8, dtype=np.uint32), 0, 3)


class TestLilicorr(unittest.TestCase):
    def test_correlation_matches_direct_cosine(self):
        case = M.make_lilicorr_case(groups=2, k=3, dc=8, tau=1.5, seed=24)
        got = M.lilicorr_batch(case["uo"].astype(np.float64), case["ui"].astype(np.float64), 1.5)
        # The stored vectors are float32, so normalize the float64 promotion of
        # those same values to match the batch helper bit for bit.
        a = case["uo"][0, 1].astype(np.float64)
        b = case["ui"][1, 2].astype(np.float64)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        self.assertAlmostEqual(float(got[0, 1, 2]), 1.5 * cos, places=9)

    def test_correlation_is_bounded_by_tau(self):
        case = M.make_lilicorr_case(groups=3, k=4, dc=16, tau=2.0, seed=25)
        got = M.lilicorr_batch(case["uo"].astype(np.float64), case["ui"].astype(np.float64), 2.0)
        self.assertLessEqual(float(np.max(np.abs(got))), 2.0 + 1e-9)

    def test_tau_scales_linearly(self):
        case = M.make_lilicorr_case(groups=2, k=2, dc=4, tau=1.0, seed=26)
        uo = case["uo"].astype(np.float64)
        ui = case["ui"].astype(np.float64)
        one = M.lilicorr_batch(uo, ui, 1.0)
        three = M.lilicorr_batch(uo, ui, 3.0)
        self.assertTrue(np.allclose(three, 3.0 * one, atol=1e-12))


class TestReferenceReport(unittest.TestCase):
    def test_report_shape(self):
        report = M.reference_report()
        self.assertEqual(report["ksc_rows"], 8)
        self.assertEqual(report["lilicorr_shape"], [3, 6, 6])
        self.assertLessEqual(abs(report["lilicorr_absmax"]), 1.0 + 1e-6)
        self.assertTrue(math.isfinite(report["ksc_first"]))
        self.assertTrue(math.isfinite(report["awsrc_first"]))


if __name__ == "__main__":
    unittest.main()
