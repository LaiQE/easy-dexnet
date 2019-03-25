import numpy as np
import trimesh
import pyrender
from .colcor import cnames


class DexScene(pyrender.Scene):
    def add_obj(self, mesh, matrix=np.eye(4), color='lightblue'):
        tri = mesh.tri_mesh
        if isinstance(color, (str)):
            color = trimesh.visual.color.hex_to_rgba(cnames[color])
        color[-1] = 200
        tri.visual.face_colors = color
        tri.visual.vertex_colors = color
        render_mesh = pyrender.Mesh.from_trimesh(tri)
        n = pyrender.Node(mesh=render_mesh, matrix=matrix)
        self.add_node(n)

    def add_grasp(self, grasp, matrix=np.eye(4), radius=0.0025, color='red'):
        def vector_to_rotation(vector):
            z = np.array(vector)
            z = z / np.linalg.norm(z)
            x = np.array([1, 0, 0])
            x = x - z*(x.dot(z)/z.dot(z))
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            return np.c_[x, y, z]
        grasp_vision = trimesh.creation.capsule(grasp.width, radius)
        rotation = vector_to_rotation(grasp.axis)
        trasform = np.eye(4)
        trasform[:3, :3] = rotation
        center = grasp.center - (grasp.width / 2) * grasp.axis
        trasform[:3, 3] = center
        grasp_vision.apply_transform(trasform)
        if isinstance(color, (str)):
            color = trimesh.visual.color.hex_to_rgba(cnames[color])
        grasp_vision.visual.face_colors = color
        grasp_vision.visual.vertex_colors = color
        render_mesh = pyrender.Mesh.from_trimesh(grasp_vision)
        n = pyrender.Node(mesh=render_mesh, matrix=matrix)
        self.add_node(n)

    def add_grasp_center(self, grasp, matrix=np.eye(4), radius=0.003, color='black'):
        point_vision = trimesh.creation.uv_sphere(radius)
        trasform = np.eye(4)
        trasform[:3, 3] = grasp.center
        point_vision.apply_transform(trasform)
        if isinstance(color, (str)):
            color = trimesh.visual.color.hex_to_rgba(cnames[color])
        point_vision.visual.face_colors = color
        point_vision.visual.vertex_colors = color
        render_mesh = pyrender.Mesh.from_trimesh(point_vision)
        n = pyrender.Node(mesh=render_mesh, matrix=matrix)
        self.add_node(n)
