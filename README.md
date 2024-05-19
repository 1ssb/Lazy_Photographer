# The Lazy Photgrapher Problem

What would it take for a lazy photographer to have all the scene best characteristics while moving the least?

The single guard art gallery problem solved for autonomous systems in SE(2), simply.

---
![A Lazy Photographer who has given up for us to find an optimal solution for him to adjust his pose to take the best image](./meta/image.webp)

---

# 3D Camera Frustum Visualization

This script visualizes camera frustums and points in 3D space using Plotly. It allows for multiple camera configurations and highlights points based on their visibility within the frustums.

## Features

- Visualize multiple camera frustums in 3D space.
- Display points inside exactly one frustum in green.
- Display points inside multiple frustums in orange.
- Display points outside all frustums in yellow.
- Different colors for each camera to distinguish them easily.
- Save the visualization as an HTML file for easy sharing and viewing.

## Installation

1. Clone the repository or download the script.
2. Install the required Python packages:

```sh
pip install numpy plotly scipy matplotlib
```

## Usage

### Script Parameters

The script requires a dictionary `user_inputs` with two keys: `cameras` and `points`.

#### `cameras`

A list of dictionaries, each representing a camera with the following keys:

- `name`: A unique identifier for the camera.
- `camera_pos`: A list of 3 elements representing the camera position in world coordinates `[x, y, z]`.
- `cam_orien`: A list of 3 elements representing the camera orientation in degrees `[roll, pitch, yaw]`.
- `fx`: Focal length along the x-axis in pixels.
- `fy`: Focal length along the y-axis in pixels.
- `cx`: Principal point x-coordinate in pixels.
- `cy`: Principal point y-coordinate in pixels.
- `img_width`: Image width in pixels.
- `img_height`: Image height in pixels.
- `near_plane_dist`: The distance to the near plane of the frustum.
- `far_plane_dist`: The distance to the far plane of the frustum.

#### `points`

A list of 3D points to be visualized in the space.

### Example Input

```python
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
```

### Running the Script

To run the script and generate the visualization:

1. Ensure the script and your input data are in the same directory.
2. Execute the script:

```sh
python viz_cam.py
```

This will generate an HTML file named `camera_frustum.html` and automatically open it in your default web browser.

### Functions

- `rotation_matrix_from_euler(roll, pitch, yaw)`: Creates a rotation matrix from Euler angles.
- `calculate_frustum_points(camera_position, roll, pitch, yaw, fx, fy, cx, cy, near_dist, far_dist, img_width, img_height)`: Calculates the points of the camera frustum.
- `draw_camera_axes(camera_position, roll, pitch, yaw, length)`: Draws x-y-z axes to signify the orientation of the camera.
- `draw_frustum_and_cameras(cameras, points_inside, points_outside, points_multiple)`: Draws the frustums and points using Plotly.
- `point_in_hull(points, hull)`: Checks if points are inside the convex hull.
- `get_viz(user_inputs)`: Main function to visualize the camera frustum and points.
- `save_viz_to_html(fig, file_name, auto_open)`: Saves the Plotly figure to an HTML file.
- `viz_cameras(cam_dict)`: Visualizes the camera frustums and points.

## License

This project is licensed under the MIT License.

## Contribution

Feel free to contribute or report issues to me at Subhransu.Bhattacharjee@anu.edu.au.
