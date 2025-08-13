import numpy as np
import open3d as o3d

height, width = 480, 640
depth_image = np.zeros((height, width), dtype=np.uint16)

# Create a horizontal gradient (farther on the right side)
for x in range(width):
    depth_image[:, x] = 500 + int(1500 * (x / width))  # depths from 500mm to 2000mm

# Add a circular "closer object" in the middle
center_x, center_y = width // 2, height // 2
radius = 100
for y in range(height):
    for x in range(width):
        if (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2:
            depth_image[y, x] = 800  # closer circle at 800mm


# Define camera intrinsics (Example values — replace with real ones if available)
width, height = 640, 480
fx, fy = 525.0, 525.0   # focal lengths
cx, cy = width / 2, height / 2  # principal point

intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Convert depth to point cloud
# Convert depth from mm to meters if needed (scale = 1000.0)
depth_scale = 1000.0  
pcd = o3d.geometry.PointCloud.create_from_depth_image(
    depth_image,
    intrinsic,
    depth_scale=depth_scale,
    project_valid_depth_only=True
)

# Visualize
o3d.visualization.draw_geometries([pcd])

# Optionally save as PCD file
o3d.io.write_point_cloud("output_point_cloud.pcd", pcd)
