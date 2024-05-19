import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import plotly.io as pio

def rotation_matrix_from_euler(roll, pitch, yaw):
    """Create a rotation matrix from Euler angles."""
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def calculate_frustum_points(camera_position, roll, pitch, yaw, f, near_dist, far_dist, img_size):
    """Calculate the points of the camera frustum."""
    fov = 2 * np.arctan(img_size / (2 * f))
    near_height = near_width = 2 * np.tan(fov / 2) * near_dist
    far_height = far_width = 2 * np.tan(fov / 2) * far_dist
    near_centroid = np.array([0, 0, near_dist])
    far_centroid = np.array([0, 0, far_dist])
    near_plane = np.array([near_centroid + np.array([x, y, 0]) for x in [-near_width/2, near_width/2] for y in [-near_height/2, near_height/2]])
    far_plane = np.array([far_centroid + np.array([x, y, 0]) for x in [-far_width/2, far_width/2] for y in [-far_height/2, far_height/2]])
    R = rotation_matrix_from_euler(roll, pitch, yaw)
    near_plane = (R @ near_plane.T).T + camera_position
    far_plane = (R @ far_plane.T).T + camera_position
    return near_plane, far_plane

def draw_frustum(near_plane, far_plane, camera_position, points_inside, points_outside):
    """Draw the frustum and points using Plotly."""
    fig = go.Figure()

    # Add near and far planes as transparent surfaces with legend
    fig.add_trace(go.Mesh3d(
        x=near_plane[:, 0], y=near_plane[:, 1], z=near_plane[:, 2],
        color='blue', opacity=0.5, i=[0, 1, 2, 0, 2, 3], j=[1, 2, 3, 3, 0, 1], k=[2, 3, 0, 2, 3, 1],
        name='Near Plane', showlegend=True
    ))
    fig.add_trace(go.Mesh3d(
        x=far_plane[:, 0], y=far_plane[:, 1], z=far_plane[:, 2],
        color='cyan', opacity=0.5, i=[0, 1, 2, 0, 2, 3], j=[1, 2, 3, 3, 0, 1], k=[2, 3, 0, 2, 3, 1],
        name='Far Plane', showlegend=True
    ))

    # Add edges for the frustum without legend
    for i in range(4):
        fig.add_trace(go.Scatter3d(x=[near_plane[i, 0], near_plane[(i + 1) % 4, 0]], 
                                   y=[near_plane[i, 1], near_plane[(i + 1) % 4, 1]], 
                                   z=[near_plane[i, 2], near_plane[(i + 1) % 4, 2]], 
                                   mode='lines', line=dict(color='blue', width=2), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[far_plane[i, 0], far_plane[(i + 1) % 4, 0]], 
                                   y=[far_plane[i, 1], far_plane[(i + 1) % 4, 1]], 
                                   z=[far_plane[i, 2], far_plane[(i + 1) % 4, 2]], 
                                   mode='lines', line=dict(color='blue', width=2), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[near_plane[i, 0], far_plane[i, 0]], 
                                   y=[near_plane[i, 1], far_plane[i, 1]], 
                                   z=[near_plane[i, 2], far_plane[i, 2]], 
                                   mode='lines', line=dict(color='red', width=2), showlegend=False))

    # Add camera position with legend
    fig.add_trace(go.Scatter3d(x=[camera_position[0]], y=[camera_position[1]], z=[camera_position[2]],
                               mode='markers', marker=dict(color='red', size=6), name='Camera Position', showlegend=True))

    # Add user-defined points with legend
    fig.add_trace(go.Scatter3d(x=points_inside[:, 0], y=points_inside[:, 1], z=points_inside[:, 2],
                               mode='markers', marker=dict(color='green', size=4), name='Points Inside Frustum', showlegend=True))
    fig.add_trace(go.Scatter3d(x=points_outside[:, 0], y=points_outside[:, 1], z=points_outside[:, 2],
                               mode='markers', marker=dict(color='yellow', size=4), name='Points Outside Frustum', showlegend=True))

    # Set plot labels and title
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'),
        title='3D Camera Frustum Visualization'
    )
    
    return fig

def point_in_hull(points, hull):
    """Check if points are inside the convex hull."""
    return np.all(np.add(np.dot(hull.equations[:, :-1], points.T), hull.equations[:, -1][:, np.newaxis]) <= 1e-12, axis=0)

def get_viz(user_inputs):
    """Main function to visualize the camera frustum and points."""
    camera_position = np.array(user_inputs['camera_pos'])
    roll, pitch, yaw = user_inputs['cam_orien']
    f = user_inputs['focal_length']
    img_size = user_inputs['img_size']
    near_dist = user_inputs['near_plane_dist']
    far_dist = user_inputs['far_plane_dist']
    points = np.array(user_inputs['points'])

    near_plane, far_plane = calculate_frustum_points(camera_position, roll, pitch, yaw, f, near_dist, far_dist, img_size)
    frustum_hull = ConvexHull(np.vstack((near_plane, far_plane)))

    in_hull = point_in_hull(points, frustum_hull)
    points_inside = points[in_hull]
    points_outside = points[~in_hull]

    fig = draw_frustum(near_plane, far_plane, camera_position, points_inside, points_outside)
    return fig

def save_viz_to_html(fig, file_name, show):
    """Save the Plotly figure to an HTML file."""
    pio.write_html(fig, file=file_name, auto_open=show)
    
# Example usage
if __name__ == "__main__":
    user_inputs = {
        'camera_pos': [0, 0, 0],
        'cam_orien': [0, 0, 0],
        'focal_length': 100,
        'img_size': 128,
        'near_plane_dist': 0.5,
        'far_plane_dist': 10.0,
        'points': [[1, 1, 1], [5, 5, 5], [-2, -2, -2], [0, 0, 5]]
    }
    fig = get_viz(user_inputs)
    # fig.show()
    save_viz_to_html(fig, 'camera_frustum.html', show=True)

