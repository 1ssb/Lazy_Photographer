import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import plotly.io as pio
import matplotlib.pyplot as plt

def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    Create a rotation matrix from Euler angles.
    
    Parameters:
    - roll: Rotation around x-axis in degrees
    - pitch: Rotation around y-axis in degrees
    - yaw: Rotation around z-axis in degrees

    Returns:
    - Rotation matrix as a numpy array
    """
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def calculate_frustum_points(camera_position, roll, pitch, yaw, fx, fy, cx, cy, near_dist, far_dist, img_width, img_height):
    """
    Calculate the points of the camera frustum.
    
    Parameters:
    - camera_position: Position of the camera in world coordinates
    - roll: Rotation around x-axis in degrees
    - pitch: Rotation around y-axis in degrees
    - yaw: Rotation around z-axis in degrees
    - fx: Focal length along x-axis in pixels
    - fy: Focal length along y-axis in pixels
    - cx: Principal point x-coordinate in pixels
    - cy: Principal point y-coordinate in pixels
    - near_dist: Near plane distance
    - far_dist: Far plane distance
    - img_width: Image width in pixels
    - img_height: Image height in pixels

    Returns:
    - near_plane: Coordinates of the near plane corners
    - far_plane: Coordinates of the far plane corners
    """
    if not isinstance(camera_position, (list, np.ndarray)) or len(camera_position) != 3:
        raise ValueError("camera_position must be a list or numpy array with 3 elements")
    if not all(isinstance(param, (int, float)) and param > 0 for param in [fx, fy, near_dist, far_dist, img_width, img_height]):
        raise ValueError("fx, fy, near_dist, far_dist, img_width, and img_height must be positive numbers")
    if not isinstance(cx, (int, float)) or not isinstance(cy, (int, float)):
        raise ValueError("cx and cy must be numbers")
    if not isinstance(far_dist, (int, float)) or far_dist <= near_dist:
        raise ValueError("far_plane_dist must be greater than near_plane_dist")
    
    fov_x = 2 * np.arctan(img_width / (2 * fx))
    fov_y = 2 * np.arctan(img_height / (2 * fy))
    near_height = 2 * np.tan(fov_y / 2) * near_dist
    near_width = 2 * np.tan(fov_x / 2) * near_dist
    far_height = 2 * np.tan(fov_y / 2) * far_dist
    far_width = 2 * np.tan(fov_x / 2) * far_dist
    near_centroid = np.array([0, 0, near_dist])
    far_centroid = np.array([0, 0, far_dist])
    near_plane = np.array([near_centroid + np.array([x, y, 0]) for x in [-near_width / 2, near_width / 2] for y in [-near_height / 2, near_height / 2]])
    far_plane = np.array([far_centroid + np.array([x, y, 0]) for x in [-far_width / 2, far_width / 2] for y in [-far_height / 2, far_height / 2]])
    R = rotation_matrix_from_euler(roll, pitch, yaw)
    near_plane = (R @ near_plane.T).T + camera_position
    far_plane = (R @ far_plane.T).T + camera_position
    return near_plane, far_plane

def draw_camera_axes(camera_position, roll, pitch, yaw, length=1.0):
    """
    Draw x-y-z axes to signify the orientation of the camera.
    
    Parameters:
    - camera_position: Position of the camera in world coordinates
    - roll: Rotation around x-axis in degrees
    - pitch: Rotation around y-axis in degrees
    - yaw: Rotation around z-axis in degrees
    - length: Length of the axes (default is 1.0)

    Returns:
    - origin: Camera position
    - x_axis: End point of the x-axis
    - y_axis: End point of the y-axis
    - z_axis: End point of the z-axis
    """
    R = rotation_matrix_from_euler(roll, pitch, yaw)
    origin = camera_position
    x_axis = origin + R @ np.array([length, 0, 0])
    y_axis = origin + R @ np.array([0, length, 0])
    z_axis = origin + R @ np.array([0, 0, length])
    return origin, x_axis, y_axis, z_axis

def draw_frustum_and_cameras(cameras, points_inside, points_outside, points_multiple):
    """
    Draw the frustum and points using Plotly.
    
    Parameters:
    - cameras: List of camera configurations
    - points_inside: Points inside exactly one frustum
    - points_outside: Points outside all frustums
    - points_multiple: Points inside multiple frustums

    Returns:
    - fig: Plotly figure object
    """
    fig = go.Figure()
    
    # Generate colors for cameras using a colormap
    cmap = plt.colormaps.get_cmap('Set1')

    for idx, camera in enumerate(cameras):
        camera_position = np.array(camera['camera_pos'])
        roll, pitch, yaw = camera['cam_orien']
        fx = camera['fx']
        fy = camera['fy']
        cx = camera['cx']
        cy = camera['cy']
        img_width = camera['img_width']
        img_height = camera['img_height']
        near_dist = camera['near_plane_dist']
        far_dist = camera['far_plane_dist']
        
        # Convert color from colormap to hexadecimal
        color = cmap(idx)
        hex_color = '#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255))

        near_plane, far_plane = calculate_frustum_points(camera_position, roll, pitch, yaw, fx, fy, cx, cy, near_dist, far_dist, img_width, img_height)

        # Add near and far planes as transparent surfaces with legend
        fig.add_trace(go.Mesh3d(
            x=near_plane[:, 0], y=near_plane[:, 1], z=near_plane[:, 2],
            color=hex_color, opacity=0.5, i=[0, 1, 2, 0, 2, 3], j=[1, 2, 3, 3, 0, 1], k=[2, 3, 0, 2, 3, 1],
            name=f'Camera {camera["name"]} Near Plane', showlegend=True
        ))
        fig.add_trace(go.Mesh3d(
            x=far_plane[:, 0], y=far_plane[:, 1], z=far_plane[:, 2],
            color=hex_color, opacity=0.5, i=[0, 1, 2, 0, 2, 3], j=[1, 2, 3, 3, 0, 1], k=[2, 3, 0, 2, 3, 1],
            name=f'Camera {camera["name"]} Far Plane', showlegend=True
        ))

        # Add edges for the frustum without legend
        for i in range(4):
            fig.add_trace(go.Scatter3d(x=[near_plane[i, 0], near_plane[(i + 1) % 4, 0]], 
                                       y=[near_plane[i, 1], near_plane[(i + 1) % 4, 1]], 
                                       z=[near_plane[i, 2], near_plane[(i + 1) % 4, 2]], 
                                       mode='lines', line=dict(color=hex_color, width=2), showlegend=False))
            fig.add_trace(go.Scatter3d(x=[far_plane[i, 0], far_plane[(i + 1) % 4, 0]], 
                                       y=[far_plane[i, 1], far_plane[(i + 1) % 4, 1]], 
                                       z=[far_plane[i, 2], far_plane[(i + 1) % 4, 2]], 
                                       mode='lines', line=dict(color=hex_color, width=2), showlegend=False))
            fig.add_trace(go.Scatter3d(x=[near_plane[i, 0], far_plane[i, 0]], 
                                       y=[near_plane[i, 1], far_plane[i, 1]], 
                                       z=[near_plane[i, 2], far_plane[i, 2]], 
                                       mode='lines', line=dict(color=hex_color, width=2), showlegend=False))

        # Add camera position with legend
        fig.add_trace(go.Scatter3d(x=[camera_position[0]], y=[camera_position[1]], z=[camera_position[2]],
                                   mode='markers', marker=dict(color=hex_color, size=6), name=f'Camera {camera["name"]} Position', showlegend=True))

        # Add camera orientation axes without legend
        origin, x_axis, y_axis, z_axis = draw_camera_axes(camera_position, roll, pitch, yaw)
        fig.add_trace(go.Scatter3d(x=[origin[0], x_axis[0]], y=[origin[1], x_axis[1]], z=[origin[2], x_axis[2]],
                                   mode='lines', line=dict(color='red', width=2), showlegend=False, name=f'Camera {camera["name"]} X-axis'))
        fig.add_trace(go.Scatter3d(x=[origin[0], y_axis[0]], y=[origin[1], y_axis[1]], z=[origin[2], y_axis[2]],
                                   mode='lines', line=dict(color='green', width=2), showlegend=False, name=f'Camera {camera["name"]} Y-axis'))
        fig.add_trace(go.Scatter3d(x=[origin[0], z_axis[0]], y=[origin[1], z_axis[1]], z=[origin[2], z_axis[2]],
                                   mode='lines', line=dict(color='blue', width=2), showlegend=False, name=f'Camera {camera["name"]} Z-axis'))

    # Add user-defined points with legend
    fig.add_trace(go.Scatter3d(x=points_inside[:, 0], y=points_inside[:, 1], z=points_inside[:, 2],
                               mode='markers', marker=dict(color='green', size=4), name='Points Inside One Frustum', showlegend=True))
    fig.add_trace(go.Scatter3d(x=points_multiple[:, 0], y=points_multiple[:, 1], z=points_multiple[:, 2],
                               mode='markers', marker=dict(color='orange', size=4), name='Points Inside Multiple Frustums', showlegend=True))
    fig.add_trace(go.Scatter3d(x=points_outside[:, 0], y=points_outside[:, 1], z=points_outside[:, 2],
                               mode='markers', marker=dict(color='yellow', size=4), name='Points Outside Frustums', showlegend=True))

    # Set plot labels and title
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'),
        title='3D Camera Frustum Visualization'
    )
    
    return fig

def point_in_hull(points, hull):
    """
    Check if points are inside the convex hull.
    
    Parameters:
    - points: Array of points to check
    - hull: Convex hull object

    Returns:
    - Boolean array indicating whether each point is inside the hull
    """
    return np.all(np.add(np.dot(hull.equations[:, :-1], points.T), hull.equations[:, -1][:, np.newaxis]) <= 1e-12, axis=0)

def get_viz(user_inputs):
    """
    Main function to visualize the camera frustum and points.
    
    Parameters:
    - user_inputs: Dictionary containing camera configurations and points

    Returns:
    - fig: Plotly figure object
    """
    if not isinstance(user_inputs, dict):
        raise ValueError("user_inputs must be a dictionary")

    if 'cameras' not in user_inputs or 'points' not in user_inputs:
        raise ValueError("user_inputs dictionary must contain 'cameras' and 'points' keys")

    cameras = user_inputs['cameras']
    points = np.array(user_inputs['points'])

    if not all(isinstance(camera, dict) for camera in cameras):
        raise ValueError("Each camera configuration must be a dictionary")
    
    if not isinstance(points, np.ndarray):
        raise ValueError("points must be a numpy array")

    all_frustum_points = []
    points_in_frustums = np.zeros(len(points), dtype=int)
    
    for camera in cameras:
        if not all(key in camera for key in ['camera_pos', 'cam_orien', 'fx', 'fy', 'cx', 'cy', 'img_width', 'img_height', 'near_plane_dist', 'far_plane_dist']):
            raise ValueError("Each camera dictionary must contain 'camera_pos', 'cam_orien', 'fx', 'fy', 'cx', 'cy', 'img_width', 'img_height', 'near_plane_dist', and 'far_plane_dist' keys")
        
        camera_position = np.array(camera['camera_pos'])
        roll, pitch, yaw = camera['cam_orien']
        fx = camera['fx']
        fy = camera['fy']
        cx = camera['cx']
        cy = camera['cy']
        img_width = camera['img_width']
        img_height = camera['img_height']
        near_dist = camera['near_plane_dist']
        far_dist = camera['far_plane_dist']

        near_plane, far_plane = calculate_frustum_points(camera_position, roll, pitch, yaw, fx, fy, cx, cy, near_dist, far_dist, img_width, img_height)
        all_frustum_points.append(near_plane)
        all_frustum_points.append(far_plane)

        frustum_hull = ConvexHull(np.vstack((near_plane, far_plane)))
        points_in_frustums += point_in_hull(points, frustum_hull)

    points_inside = points[points_in_frustums == 1]
    points_multiple = points[points_in_frustums > 1]
    points_outside = points[points_in_frustums == 0]

    fig = draw_frustum_and_cameras(cameras, points_inside, points_outside, points_multiple)
    return fig

def save_viz_to_html(fig, file_name, auto_open=True):
    """
    Save the Plotly figure to an HTML file.
    
    Parameters:
    - fig: Plotly figure object
    - file_name: Name of the HTML file
    - auto_open: Boolean indicating whether to automatically open the file in a web browser (default is True)
    """
    if not isinstance(file_name, str):
        raise ValueError("file_name must be a string")
    pio.write_html(fig, file=file_name, auto_open=auto_open)

def viz_cameras(cam_dict):
    """
    Visualize the camera frustums and points.
    
    Parameters:
    - cam_dict: Dictionary containing camera configurations and points
    """
    fig = get_viz(cam_dict)
    save_viz_to_html(fig, 'camera_frustum.html', auto_open=True)

if __name__ == "__main__":
    """
    Example usage:

    The script visualizes camera frustums and points in 3D space. 
    It requires a dictionary `user_inputs` with two keys: 'cameras' and 'points'.

    'cameras': A list of dictionaries, each representing a camera with the following keys:
        - 'name': A unique identifier for the camera.
        - 'camera_pos': A list of 3 elements representing the camera position in world coordinates [x, y, z].
        - 'cam_orien': A list of 3 elements representing the camera orientation in degrees [roll, pitch, yaw].
        - 'fx': Focal length along the x-axis in pixels.
        - 'fy': Focal length along the y-axis in pixels.
        - 'cx': Principal point x-coordinate in pixels.
        - 'cy': Principal point y-coordinate in pixels.
        - 'img_width': Image width in pixels.
        - 'img_height': Image height in pixels.
        - 'near_plane_dist': The distance to the near plane of the frustum.
        - 'far_plane_dist': The distance to the far plane of the frustum.

    'points': A list of 3D points to be visualized in the space.

    The visualization will show the frustums of the two cameras, the points inside one frustum in green, 
    points inside multiple frustums in orange, and points outside all frustums in yellow.
    """
    user_inputs = {
        'cameras': [
            {
                'name': '1',
                'camera_pos': [0, 0, 0],
                'cam_orien': [0, 0, 0],
                'fx': 100,
                'fy': 100,
                'cx': 64,
                'cy': 64,
                'img_width': 128,
                'img_height': 128,
                'near_plane_dist': 0.5,
                'far_plane_dist': 10.0
            },
            {
                'name': '2',
                'camera_pos': [5, 5, 5],
                'cam_orien': [30, 45, 60],
                'fx': 100,
                'fy': 100,
                'cx': 64,
                'cy': 64,
                'img_width': 128,
                'img_height': 128,
                'near_plane_dist': 0.5,
                'far_plane_dist': 10.0
            }
        ],
        'points': [
            [1, 1, 1],
            [-2, -2, -2], 
            [0, 0, 5],
            [6.4, 6.4, 10]
        ]
    }
    viz_cameras(user_inputs)
