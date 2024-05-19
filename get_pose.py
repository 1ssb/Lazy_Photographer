import numpy as np

def calculate_fov():
    """
    Calculate the horizontal field of view (FoV) based on a fixed normalized focal length of 0.45.
    """
    image_size = 128
    f_ndc = 0.45
    focal_length = f_ndc * image_size
    return 2 * np.arctan((image_size / 2) / focal_length)

def is_within_frustum(point, fov_half, near_plane=0.5, far_plane=10, camera_pose=(0, 0, 0)):
    """
    Check if a point is within the camera's view frustum from a given camera pose.
    """
    dx, dy, dz = camera_pose
    point = (point[0] - dx, point[1] - dy, point[2] - dz)
    distance = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    horizontal_angle = np.arctan2(point[0], point[2])
    return near_plane <= distance <= far_plane and abs(horizontal_angle) <= fov_half

def find_heuristic_midpoint(points):
    """
    Calculate a heuristic midpoint based on the average coordinates of the points and including the origin.
    """
    # Include the origin in the point set for midpoint calculation
    all_points = [(0, 0, 0)] + points
    mean_x = np.mean([p[0] for p in all_points])
    mean_z = np.mean([p[2] for p in all_points])
    return mean_x, 0, mean_z

def optimize_camera_pose(points, fov_half):
    """
    Use a heuristic method to optimize the camera pose to observe all points.
    """
    initial_pose = find_heuristic_midpoint(points)
    best_pose = initial_pose
    max_visible = 0

    # Adjust pose iteratively to maximize visible points
    for dx in np.linspace(initial_pose[0] - 5, initial_pose[0] + 5, 10):
        for dz in np.linspace(initial_pose[2] - 5, initial_pose[2] + 5, 10):
            current_pose = (dx, 0, dz)
            visible_count = sum(is_within_frustum(p, fov_half, 0.5, 10, current_pose) for p in points)
            if visible_count > max_visible:
                max_visible = visible_count
                best_pose = current_pose

            # Early stop if all points are visible
            if visible_count == len(points):
                return best_pose, np.arctan2(-dx, -dz)

    return best_pose, np.arctan2(-best_pose[0], -best_pose[2])

def main():
    points = []
    n_points = int(input("Enter the number of 3D points: "))
    for i in range(n_points):
        x = float(input(f"Enter the x-coordinate of PoI {i + 1}: "))
        y = float(input(f"Enter the y-coordinate of PoI {i + 1}: "))
        z = float(input(f"Enter the z-coordinate of PoI {i + 1}: "))
        points.append((x, y, z))

    fov_half = calculate_fov() / 2
    best_pose, yaw = optimize_camera_pose(points, fov_half)

    print("Optimal Pose to observe all PoIs:")
    print(f"Position: (x={best_pose[0]}, y={best_pose[1]}, z={best_pose[2]})")
    print(f"Yaw: {yaw} radians")

if __name__ == "__main__":
    main()
