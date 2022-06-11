
import numpy as np

import open3d as o3d


def generate_alpha_shape(points_np: np.array, displaying: bool = False, alpha_value: float = 0.88,
                         view_name: str = 'default'):
    '''

    :param points_np:
    :param displaying:
    :param alpha_value: bigger than 0.85, sqrt(3)/2, the delaunay triangulation would be successful.
        Because of the random bias, we need to add more than 0.01 to the 0.866
    :param view_name:
    :return: the mesh
    '''
    m_pcd = o3d.geometry.PointCloud()

    m_pcd.points = o3d.utility.Vector3dVector(points_np)  # observe the points with np.asarray(pcd.points)

    # the mesh is developed from http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html
    m_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(m_pcd, alpha_value)

    # print('mesh info',m_mesh)
    # print('edge manifold', m_mesh.is_edge_manifold(allow_boundary_edges=True))
    # print('edge manifold boundary', m_mesh.is_edge_manifold(allow_boundary_edges=False))
    # print('vertex manifold', m_mesh.is_vertex_manifold())
    # print('self intersection ', m_mesh.is_self_intersecting())
    # print('watertight', m_mesh.is_watertight())
    # print(f"alpha={alpha_value:.3f}")

    if displaying:


        # add normals, add light effect
        m_mesh.compute_vertex_normals()

        # make the non manifold vertices become red
        vertex_colors = 0.75 * np.ones((len(m_mesh.vertices), 3))
        for boundary in m_mesh.get_non_manifold_edges():
            for vertex_id in boundary:
                vertex_colors[vertex_id] = [1, 0, 0]
        m_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        o3d.visualization.draw_geometries([m_pcd, m_mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
                                          window_name=view_name)
    return m_mesh