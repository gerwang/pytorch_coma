import argparse
import os

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import openmesh as om
from matplotlib import pyplot
from sklearn.decomposition import PCA
import numba

# set vmax HERE!!!

parser = argparse.ArgumentParser()
# parser.add_argument('--cnn', help='path to dataset', action='append', nargs=1)
# parser.add_argument('--data_path', help='path to dataset')
parser.add_argument('--train_path', help='path to train model')
parser.add_argument('--test_path', help='path to test model')
parser.add_argument('--nz', type=int, help='size of the latent z vector')
parser.add_argument('--template', help='template path')
parser.add_argument('--no_color', action='store_true')
parser.add_argument('--vmax', type=float, default=4e-3)

opt = parser.parse_args()

vmax = opt.vmax

#
train_np = np.load(opt.train_path)
test_np = np.load(opt.test_path)
mean_np = np.mean(train_np, axis=0)
std_np = np.std(train_np, axis=0)
pca_n_comp = opt.nz
pca = PCA(n_components=pca_n_comp)
pca.fit(train_np.reshape(train_np.shape[0], -1))

n_vertex = train_np.shape[1]

pca_outputs = pca.inverse_transform(pca.transform(test_np.reshape(test_np.shape[0], -1)))

pca_vertices = pca_outputs.reshape(pca_outputs.shape[0], -1, 3)

print(test_np.shape)
print(pca_vertices.shape)

reference_mesh_file = opt.template

# calculate errors
face_number = test_np.shape[0]
total_point_number = test_np.shape[1]
errors = np.zeros(total_point_number)


@numba.njit
def f(errors, pca_vertices, test_np):
    for cur_face in range(face_number):
        for idx in range(total_point_number):
            errors[idx] += np.linalg.norm(pca_vertices[cur_face, idx] - test_np[cur_face, idx])
    errors /= face_number


f(errors, pca_vertices, test_np)

# print(errors)
print(max(errors))
#
jet = pyplot.get_cmap('jet')
cNorm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)  # 3mm maximum
scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)

reference_mesh = om.read_trimesh(reference_mesh_file)


def create_o3d_mesh():
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.triangles = o3d.utility.Vector3iVector(reference_mesh.face_vertex_indices())
    o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    o3d_mesh.vertices = o3d.utility.Vector3dVector(reference_mesh.points())
    o3d_mesh.compute_vertex_normals()
    error_color = o3d.utility.Vector3dVector(scalarMap.to_rgba(errors)[:, :3])
    print(scalarMap.to_rgba(errors).shape)
    o3d_mesh.vertex_colors = error_color
    return o3d_mesh


o3d_meshes = create_o3d_mesh()
visualizers = o3d.visualization.VisualizerWithKeyCallback()


def vis_step(vis):
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()


def all_step():
    vis_step(visualizers)


output_path = './'


def save_mesh(_):
    this_output_path = os.path.join(output_path, 'error_density.png')
    img = np.asarray(visualizers.capture_screen_float_buffer(do_render=True))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(this_output_path, img * 255)
    return False


should_stop = False


def stop(_):
    global should_stop
    should_stop = True
    return False


callbacks = {
    ord('['): save_mesh,
    ord(']'): stop,
}

width_vis = 603
height_vis = 1072

visualizers.create_window(window_name="error_density", width=width_vis, height=height_vis)
visualizers.add_geometry(o3d_meshes)
visualizers.update_geometry()

for key, value in callbacks.items():
    visualizers.register_key_callback(key, value)

while not should_stop:
    all_step()

visualizers.destroy_window()
