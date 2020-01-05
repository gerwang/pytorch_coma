import argparse
import os

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import openmesh as om
from matplotlib import pyplot
from sklearn.decomposition import PCA

from deep_learning.nonlinear_per_channel import per_vertex_l2_np

parser = argparse.ArgumentParser()
parser.add_argument('--cnn', help='path to dataset', action='append', nargs=1)
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--nz', type=int, help='size of the latent z vector')
parser.add_argument('--template', help='template path')
parser.add_argument('--no_color', action='store_true')

opt = parser.parse_args()

jet = pyplot.get_cmap('jet')
cNorm = matplotlib.colors.Normalize(vmin=0, vmax=2e-5)  # 3mm maximum
scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)

reference_mesh_file = opt.template
train_np = np.load(os.path.join(opt.data, 'train.npy'))
test_np = np.load(os.path.join(opt.data, 'test.npy'))
mean_np = np.mean(train_np, axis=0)
std_np = np.std(train_np, axis=0)
pca_n_comp = opt.nz
pca = PCA(n_components=pca_n_comp)
pca.fit(train_np.reshape(train_np.shape[0], -1))

n_vertex = train_np.shape[1]

pca_outputs = pca.inverse_transform(pca.transform(test_np.reshape(test_np.shape[0], -1)))

pca_vertices = pca_outputs.reshape(pca_outputs.shape[0], -1, 3)

mm_constant = 1000

err = mm_constant * np.mean(np.sqrt(np.sum((pca_vertices - test_np) ** 2, axis=2)))

print('pca: euclidean distance in mm=', err)

test_vertices = test_np

vertices = [test_vertices, pca_vertices]
names = ['test', 'pca']
use_color = not opt.no_color

for cnn_args in opt.cnn:
    cnn_path = cnn_args[0]
    cnn_outputs = np.load(cnn_path)[:, :n_vertex]
    cnn_vertices = (cnn_outputs * std_np) + mean_np
    vertices.append(cnn_vertices)
    names.append(os.path.basename(cnn_path))

reference_mesh = om.read_trimesh(reference_mesh_file)


def create_o3d_mesh():
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.triangles = o3d.utility.Vector3iVector(reference_mesh.face_vertex_indices())
    o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    return o3d_mesh


n_test = test_np.shape[0]
cur_idx = 0

n_vis = len(vertices)

for i in range(n_vis):
    vertices[i] = np.ascontiguousarray(vertices[i].astype(np.float32))

o3d_meshes = [create_o3d_mesh() for _ in range(n_vis)]
visualizers = [o3d.visualization.VisualizerWithKeyCallback() for _ in range(n_vis)]


def vis_step(vis):
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()


def all_step():
    for vis in visualizers:
        vis_step(vis)


def set_index(idx):
    for i in range(n_vis):
        o3d_meshes[i].vertices = o3d.utility.Vector3dVector(vertices[i][idx])
        o3d_meshes[i].compute_vertex_normals()
        if use_color and i != 0:
            error = per_vertex_l2_np(vertices[i][idx], test_vertices[idx], std_np)
            print(error.mean())
            error_color = o3d.utility.Vector3dVector(scalarMap.to_rgba(error)[:, :3])
            o3d_meshes[i].vertex_colors = error_color
        else:
            o3d_meshes[i].paint_uniform_color([0.75, 0.75, 0.75])
        visualizers[i].update_geometry()
    print(cur_idx)


def shift_mesh(delta):
    def func(_):
        global cur_idx
        cur_idx = (cur_idx + delta + n_test) % n_test
        set_index(cur_idx)
        return False

    return func


output_path = '/home/gerw/Documents/git-task/coma/save/cnn'


def save_mesh(_):
    this_output_path = os.path.join(output_path, '{}.png'.format(cur_idx))
    imgs = []
    for vis in visualizers:
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        imgs.append(img)
    img = np.concatenate(imgs, axis=1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(this_output_path, img * 255)
    return False


should_stop = False


def align_pose(self_vis):
    fk_filename = '/tmp/cam.json'
    for vis in visualizers:
        if self_vis is vis:
            opt = vis.get_render_option()
            opt.save_to_json(fk_filename)
    for vis in visualizers:
        if self_vis is not vis:
            opt = vis.get_render_option()
            opt.load_from_json(fk_filename)
            vis.update_renderer()


def stop(_):
    global should_stop
    should_stop = True
    return False


def switch_color(_):
    global use_color
    use_color = not use_color
    print('use_color: {}'.format(use_color))


callbacks = {
    ord('='): shift_mesh(1),
    ord('-'): shift_mesh(-1),
    ord('['): save_mesh,
    ord(']'): stop,
    ord(';'): align_pose,
    ord(','): switch_color
}

width_vis = 603
height_vis = 1072

set_index(cur_idx)

for i in range(len(visualizers)):
    visualizers[i].create_window(window_name=names[i], width=width_vis, height=height_vis)
    visualizers[i].add_geometry(o3d_meshes[i])
    for key, value in callbacks.items():
        visualizers[i].register_key_callback(key, value)

while not should_stop:
    all_step()

for vis in visualizers:
    vis.destroy_window()
