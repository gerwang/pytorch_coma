import mesh_operations
from psbody.mesh import Mesh
from main import scipy_to_torch_sparse

if __name__ == '__main__':
    m = Mesh(filename='../template/template.obj')
    adj = mesh_operations.get_vert_connectivity(m.v, m.f).tocoo()
    adj = scipy_to_torch_sparse(adj).to_dense()
