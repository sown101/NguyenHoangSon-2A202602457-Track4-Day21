"""Kiểm chứng CP2 bằng hình học biết trước, lỗi dữ liệu và calib synthetic."""
import unittest

import numpy as np

from starter.kitti_io import KittiCalib, load_calib
from starter.projection import cam_to_image, velo_to_cam


class ProjectionChecks(unittest.TestCase):
    def test_rectification_and_translation_order(self):
        # R0 xoay 90 độ quanh z, Tr tịnh tiến x: thứ tự R0 @ Tr là bắt buộc.
        rect = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        tr = np.column_stack((np.eye(3), [2., 0, 0]))
        calib = KittiCalib(np.zeros((3, 4)), rect, tr)
        np.testing.assert_allclose(velo_to_cam(np.array([[1., 0, 5]]), calib), [[0, 3, 5]])

    def test_mask_order_invalid_depth_and_image_boundaries(self):
        p = np.array([[100., 0, 50, 0], [0, 100, 40, 0], [0, 0, 1, 0]])
        xyz = np.array([[0, 0, 2], [np.nan, 0, 2], [0, 0, -1], [1, 0, 2],
                        [0, np.inf, 2], [-1, 0, 2], [0, 0, .1], [0, .8, 2]])
        uv, depth, mask = cam_to_image(xyz, p, (80, 100, 3))
        np.testing.assert_array_equal(mask, [True, False, False, False, False, True, False, False])
        np.testing.assert_allclose(uv, [[50, 40], [0, 40]])
        np.testing.assert_allclose(depth, [2, 2])

    def test_projection_uses_homogeneous_denominator(self):
        # s = z + 1, khác z: phải chia cho s nhưng trả depth = z.
        p = np.array([[100., 0, 0, 0], [0, 100, 0, 0], [0, 0, 1, 1]])
        uv, depth, mask = cam_to_image(np.array([[1., 1, 1]]), p, (100, 100))
        np.testing.assert_allclose(uv, [[50, 50]])
        np.testing.assert_allclose(depth, [1])
        self.assertTrue(mask[0])

    def test_empty_and_zero_denominator(self):
        uv, depth, mask = cam_to_image(np.empty((0, 3)), np.zeros((3, 4)), (80, 100))
        self.assertEqual(uv.shape, (0, 2))
        self.assertEqual(depth.shape, (0,))
        self.assertEqual(mask.shape, (0,))
        uv, _, mask = cam_to_image(np.array([[0., 0, 2]]), np.zeros((3, 4)), (80, 100))
        self.assertEqual(len(uv), 0)
        self.assertFalse(mask[0])

    def test_synthetic_reference_point(self):
        calib = load_calib("data/synthetic/training/calib/000000.txt")
        cam = velo_to_cam(np.array([[10., 0, 0]]), calib)
        uv, _, mask = cam_to_image(cam, calib.P2, (375, 1242))
        self.assertAlmostEqual(cam[0, 2], 9.73, delta=.03)
        np.testing.assert_allclose(uv[0], [614, 175], atol=2)
        self.assertTrue(mask[0])
        print(f"Synthetic reference: z_cam={cam[0, 2]:.4f}, uv={uv[0].round(3)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
