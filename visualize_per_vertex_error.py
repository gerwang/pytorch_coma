import argparse
import os

import open3d as o3d
import openmesh as om
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from config_parser import read_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization of per vertex errors')
    parser.add_argument('-c', '--conf', help='path of config file', default='sparse_baseline.cfg')
    args = parser.parse_args()
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config(args.conf)
    template_fname = config['template_fname']
    om_template = om.read_trimesh(template_fname)

    total_errors = np.load(os.path.join(config['visual_output_dir'], 'total_errors.npy'))
    print(np.mean(total_errors) * 1e3)
    print('min: {}, max: {}, mean: {}, std: {}'.format(total_errors.min(), total_errors.max(),
                                                       total_errors.mean(), total_errors.std()))
    jet = plt.get_cmap('jet')
    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=4e-3)  # 3mm maximum
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)
    vis_colors = scalarMap.to_rgba(total_errors)[:, :3]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(om_template.points())
    mesh.triangles = o3d.utility.Vector3iVector(om_template.face_vertex_indices())
    mesh.vertex_colors = o3d.utility.Vector3dVector(vis_colors)
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    om_template.request_vertex_colors()
    om_template.vertex_colors()[:, :3] = vis_colors
    om_template.vertex_colors()[:, 3] = 1
    om.write_mesh(filename=os.path.join(config['visual_output_dir'], 'vis_total_errors.ply'), mesh=om_template,
                  vertex_color=True)
