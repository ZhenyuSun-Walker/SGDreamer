import trimesh
import open3d as o3d
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--glb', type=str, required=True, help='Path to the input .glb file')
parser.add_argument('--ply', type=str, required=True, help='Path to the output .ply file')

args = parser.parse_args()

# 加载 .glb 文件并强制转换为 Trimesh 对象
mesh = trimesh.load(args.glb, force='mesh')

# 将 mesh 转换为点云
point_cloud = mesh.sample(100000)  # 采样点数可以根据需要调整

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# 保存为 .ply 文件
o3d.io.write_point_cloud(args.ply, pcd)
