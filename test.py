import unittest
import numpy as np

def calculate_fov(f_ndc):
    image_size = 128
    focal_length = f_ndc * image_size
    return 2 * np.arctan((image_size / 2) / focal_length)

def is_within_frustum(point, fov_half, near_plane, far_plane):
    distance = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    if distance < near_plane or distance > far_plane:
        return False
    horizontal_angle = np.arctan2(point[0], point[2])
    if abs(horizontal_angle) > fov_half:
        return False
    return True

def find_optimal_pose(point, fov_half, near_plane, far_plane):
    initial_pose = {'x': 0, 'z': 0, 'yaw': 0}
    if is_within_frustum(point, fov_half, near_plane, far_plane):
        angle_to_point = np.arctan2(point[0], point[2])
        return {'x': 0, 'z': 0, 'yaw': angle_to_point}
    optimal_pose = None
    min_distance = float('inf')
    for dx in np.linspace(-10, 10, num=20):
        for dz in np.linspace(-10, 10, num=20):
            candidate_pose = {'x': dx, 'z': dz, 'yaw': np.arctan2(point[0] - dx, point[2] - dz)}
            if is_within_frustum((point[0] - dx, point[1], point[2] - dz), fov_half, near_plane, far_plane):
                distance = np.sqrt(dx**2 + dz**2) + abs(initial_pose['yaw'] - candidate_pose['yaw'])
                if distance < min_distance:
                    min_distance = distance
                    optimal_pose = candidate_pose
    return optimal_pose

class TestCameraSystem(unittest.TestCase):

    def test_calculate_fov(self):
        # Test if FoV calculation is correct for a few sample f_ndc values
        self.assertAlmostEqual(calculate_fov(1.5), 2 * np.arctan(64 / (1.5 * 128)), places=6)
        self.assertAlmostEqual(calculate_fov(2.0), 2 * np.arctan(64 / (2.0 * 128)), places=6)

    def test_is_within_frustum(self):
        # Test points inside and outside the frustum
        self.assertTrue(is_within_frustum((2, 0, 3), calculate_fov(1.5) / 2, 0.7 * 128 * 1.5, 10 * 128 * 1.5))
        self.assertFalse(is_within_frustum((300, 0, 300), calculate_fov(1.5) / 2, 0.7 * 128 * 1.5, 10 * 128 * 1.5))

    def test_find_optimal_pose(self):
        # Test if the optimal pose is computed correctly for a simple scenario
        point = (2, 0, 3)
        fov_half = calculate_fov(1.5) / 2
        near_plane = 0.7 * 128 * 1.5
        far_plane = 10 * 128 * 1.5
        pose = find_optimal_pose(point, fov_half, near_plane, far_plane)
        self.assertAlmostEqual(pose['yaw'], np.arctan2(2, 3), places=6)

if __name__ == '__main__':
    unittest.main()
