import numpy as np

def calculate_fov(f_ndc):
    """
    Calculate the horizontal field of view (FoV) based on the focal length in normalized device coordinates.
    """
    image_size = 128  # Pixel width of the image
    focal_length = f_ndc * image_size  # Convert to the same units as the image size
    return 2 * np.arctan((image_size / 2) / focal_length)

def is_within_frustum(point, fov_half, near_plane, far_plane):
    """
    Check if a point is within the camera's view frustum considering the FoV and depth planes.
    """
    distance = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    if distance < near_plane or distance > far_plane:
        return False
    horizontal_angle = np.arctan2(point[0], point[2])
    if abs(horizontal_angle) > fov_half:
        return False
    return True

def find_optimal_pose(point, fov_half, near_plane, far_plane):
    """
    Determine the optimal camera pose to observe the point, minimizing the geodesic distance.
    """
    initial_pose = {'x': 0, 'z': 0, 'yaw': 0}
    
    # Try to observe the point from the origin
    if is_within_frustum(point, fov_half, near_plane, far_plane):
        angle_to_point = np.arctan2(point[0], point[2])
        return {'x': 0, 'z': 0, 'yaw': angle_to_point}
    
    # Adjust the position to ensure visibility if not visible from the origin
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

def main():
    # Set up the scene with points and camera parameters
    n_scenes = int(input("Enter the number of scenes: "))
    points = []
    fovs_half = []
    near_planes = []
    far_planes = []

    for scene_index in range(n_scenes):
        print(f"Processing Scene {scene_index + 1}")
        f_ndc = float(input(f"Enter the normalized focal length f_{scene_index + 1} (f >= 1): "))
        if f_ndc < 1:
            print("Normalized focal length should be at least 1. Please enter a valid value.")
            continue
        fov_half = calculate_fov(f_ndc) / 2
        fovs_half.append(fov_half)
        
        near_plane = 0.7 * 128 * f_ndc
        far_plane = 10 * 128 * f_ndc
        near_planes.append(near_plane)
        far_planes.append(far_plane)
        
        x = float(input(f"Enter the x-coordinate of the PoI for Scene {scene_index + 1}: "))
        y = float(input(f"Enter the y-coordinate of the PoI for Scene {scene_index + 1}: "))
        z = float(input(f"Enter the z-coordinate of the PoI for Scene {scene_index + 1}: "))
        
        points.append((x, y, z))

    # Calculate optimal poses for all points
    for idx, point in enumerate(points):
        pose = find_optimal_pose(point, fovs_half[idx], near_planes[idx], far_planes[idx])
        print(f"Optimal Pose for PoI in Scene {idx + 1}:")
        print(f"  Position: (x={pose['x']}, z={pose['z']})")
        print(f"  Yaw: {pose['yaw']} radians")

if __name__ == "__main__":
    main()
