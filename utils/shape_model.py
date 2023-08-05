#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import dependency library
from copy import deepcopy
from json import load
from time import time

import open3d as o3d
import numpy as np
import os
import urllib.request
import tarfile
import gzip
import shutil


# import user defined library


def get_contact_surface_mesh(cell_key: int, surface_data: dict, surface_contact_data: dict,
                             m_mesh: o3d.geometry.TriangleMesh,
                             displaying=False):
    '''
    getting contact surface and build its triangulation mesh

    :param surface_data:
    :param surface_contact_data:
    :param cell_key:
    :param m_mesh:
    :param displaying:
    :return:
    '''
    cell_vertices = np.asarray(m_mesh.vertices).astype(int)

    display_key_list = []
    for idx in surface_contact_data.keys():
        label1_2 = idx.split('_')
        if str(cell_key) in label1_2:
            display_key_list.append(idx)

    item_count = 1
    print('===========>contact with cell number', len(display_key_list), len(surface_data[str(cell_key)]))

    contact_list = []
    cell_volume_size = m_mesh.get_volume()
    cell_surface_area = m_mesh.get_surface_area()
    # enumerate each contact surface (cell - cell)
    for idx in display_key_list:
        # print(cell_key,idx)
        # ---------------------directly------------------------------------------
        # contact_vertices_loc_tmp=None
        # contact_mask=[]
        # # contact_vertices = None
        # time0 = time()
        contact_mask_not = [True for i in range(len(cell_vertices))]
        for item_str in surface_contact_data[idx]:
            x, y, z = item_str.split('_')
            x, y, z = int(x), int(y), int(z)
            # print(np.prod(cell_vertices == [x, y, z], axis=-1))
            contact_vertices_loc_tmp = np.where(np.prod(cell_vertices == [x, y, z], axis=-1))

            # contact_vertices=np.concatenate(contact_vertices,contact_mesh_direct.vertices[])
            # if len(contact_vertices_loc_tmp[0]) != 0:
            #     contact_mask.append(contact_vertices_loc_tmp[0][0])
            # t1,
            if len(contact_vertices_loc_tmp[0]) != 0:
                contact_mask_not[contact_vertices_loc_tmp[0][0]] = False
        # ----------------mesh -------------------------
        # contact_vertices_loc=np.where(np.prod(cell_vertices == [x, y, z], axis=-1))
        # contact_mask_not=np.logical_not(np.prod(cell_vertices == [x, y, z], axis=-1))

        contact_mesh = deepcopy(m_mesh)
        contact_mesh.remove_vertices_by_mask(contact_mask_not)
        # print('timing', time() - time0)

        # print('mesh info', contact_mesh)
        # print('edge manifold', contact_mesh.is_edge_manifold(allow_boundary_edges=True))
        # print('edge manifold boundary', contact_mesh.is_edge_manifold(allow_boundary_edges=False))

        if displaying:
            contact_mesh.compute_vertex_normals()

            vertex_colors = 0.75 * np.ones((len(contact_mesh.vertices), 3))
            for boundary in contact_mesh.get_non_manifold_edges(allow_boundary_edges=False):
                for vertex_id in boundary:
                    vertex_colors[vertex_id] = [1, 0, 0]
            contact_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            print(cell_key, ' volume', cell_volume_size, ' surface area', cell_surface_area,
                  '  alpha shape method contace surface area', contact_mesh.get_surface_area())

            o3d.visualization.draw_geometries([contact_mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
        # p = Process(target=generate_alpha_shape,
        #             args=(np.array(draw_points_list), True,2,))
        # p.start()

        item_count += 1
        contact_list.append(contact_mesh.get_surface_area())
    print(cell_volume_size, cell_surface_area, contact_list)


def get_armadillo_mesh():
    armadillo_path = r"../test_data/Armadillo.ply"
    if not os.path.exists(armadillo_path):
        print("downloading armadillo mesh")
        url = "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz"
        urllib.request.urlretrieve(url, armadillo_path + ".gz")
        print("extract armadillo mesh")
        with gzip.open(armadillo_path + ".gz", "rb") as fin:
            with open(armadillo_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        os.remove(armadillo_path + ".gz")
    mesh = o3d.io.read_triangle_mesh(armadillo_path)
    mesh.compute_vertex_normals()
    return mesh


def get_bunny_mesh():
    bunny_path = r"../test_data/Bunny.ply"
    if not os.path.exists(bunny_path):
        print("downloading bunny mesh")
        url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
        urllib.request.urlretrieve(url, bunny_path + ".tar.gz")
        print("extract bunny mesh")
        with tarfile.open(bunny_path + ".tar.gz") as tar:
            tar.extractall(path=os.path.dirname(bunny_path))
        shutil.move(
            os.path.join(
                os.path.dirname(bunny_path),
                "bunny",
                "reconstruction",
                "bun_zipper.ply",
            ),
            bunny_path,
        )
        os.remove(bunny_path + ".tar.gz")
        shutil.rmtree(os.path.join(os.path.dirname(bunny_path), "bunny"))
    mesh = o3d.io.read_triangle_mesh(bunny_path)
    print(np.asarray(mesh.vertices).tolist())
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.vertices))

    return mesh


def generate_alpha_shape(points_np: np.array, displaying: bool = False, alpha_value: float = 0.88,
                         view_name: str = 'default', print_info: bool = False):
    '''

    :param points_np:
    :param displaying:
    :param alpha_value: bigger than 0.85, sqrt(3)/2, the delaunay triangulation would be successful.
        Because of the random bias, we need to add more than 0.01 to the 0.866
    :param view_name:
    :return:
    '''
    m_pcd = o3d.geometry.PointCloud()

    m_pcd.points = o3d.utility.Vector3dVector(points_np)  # observe the points with np.asarray(pcd.points)

    # the mesh is developed from http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html
    m_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(m_pcd, alpha_value)

    if print_info:
        print('mesh info', m_mesh)
        print('edge manifold', m_mesh.is_edge_manifold(allow_boundary_edges=True))
        print('edge manifold boundary', m_mesh.is_edge_manifold(allow_boundary_edges=False))
        print('vertex manifold', m_mesh.is_vertex_manifold())
        print('watertight', m_mesh.is_watertight())
        print(f"alpha={alpha_value:.3f}")

    if displaying:
        # add normals, add light effect
        m_mesh.compute_vertex_normals()

        # make the non manifold vertices become red
        vertex_colors = 0.75 * np.ones((len(m_mesh.vertices), 3))
        for boundary in m_mesh.get_non_manifold_edges():
            for vertex_id in boundary:
                vertex_colors[vertex_id] = [1, 0, 0]
        m_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        o3d.visualization.draw_geometries([m_mesh], mesh_show_back_face=True, mesh_show_wireframe=True,
                                          window_name=view_name)
    return m_mesh


if __name__ == '__main__':
    generate_alpha_shape(np.random.uniform(size=(10, 3)), displaying=True, alpha_value=1)
