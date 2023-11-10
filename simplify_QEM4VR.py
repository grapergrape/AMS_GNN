import numpy as np
from pygsp import graphs
import trimesh
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation as R
from old_downsample import *


class QEM4VR_Simplifier:
    def __init__(self, path):
        self.mesh = trimesh.load_mesh(path)
        self.mesh = downsample_with_knn(self.mesh, 1)
        self.compute_vertex_curvature()
        self.compute_texture_boundary()
        
    def compute_vertex_curvature(self):
        adjacency = self.mesh.vertex_adjacency_graph
        self.curvatures = np.zeros(self.mesh.vertices.shape[0])
        for vertex in range(self.mesh.vertices.shape[0]):
            neighbor_indices = list(adjacency[vertex].keys())
            curvature = 0
            for i in range(len(neighbor_indices)):
                j = (i + 1) % len(neighbor_indices)
                v1 = self.mesh.vertices[neighbor_indices[i]] - self.mesh.vertices[vertex]
                v2 = self.mesh.vertices[neighbor_indices[j]] - self.mesh.vertices[vertex]
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                curvature += angle
            curvature = (2 * np.pi) - curvature
            if np.abs(curvature) > 1e-5:
                self.curvatures[vertex] = curvature
                
    def compute_texture_boundary(self):
        # assuming color is used to mark texture
        vertex_colors = self.mesh.visual.vertex_colors[:, :3]  # excluding alpha
        self.texture_boundary = KNeighborsClassifier(n_neighbors=1).fit(vertex_colors, range(len(vertex_colors)))

    def simplify_by_qem(self, threshold):
        quadrics = self.calculate_quadrics()

        while True:
            error_metric, face_index = self.compute_error_metric(quadrics)
            if np.min(error_metric) > threshold:
                break
            print(np.min(error_metric))
            self.apply_edge_flip(face_index[np.argmin(error_metric)], quadrics)

        final_mesh = self.mesh.submesh(np.unique(self.mesh.faces, axis=0), append=True)
        # Recalculate texture boundary and vertex curvatures
        self.mesh = final_mesh
        self.compute_vertex_curvature()
        self.compute_texture_boundary()

        return self.mesh

    def calculate_quadrics(self):
        quadrics = np.zeros((len(self.mesh.vertices), 4, 4))

        for face in self.mesh.faces:
            point = self.mesh.vertices[face[0]]
            normal = np.cross(self.mesh.vertices[face[1]] - point, self.mesh.vertices[face[2]] - point)
            normal /= np.linalg.norm(normal)
            d = -np.dot(point, normal)
            Q = np.outer(np.append(normal, d), np.append(normal, d))

            for vertex in face:
                quadrics[vertex] += Q

        return quadrics

    def compute_error_metric(self, quadrics):
        error_metric = np.zeros(len(self.mesh.faces))
        face_index = np.zeros(len(self.mesh.faces), dtype=int)

        for i, face in enumerate(self.mesh.faces):
            q_total = sum(quadrics[vertex] for vertex in face)

            try:
                v_optimal = np.linalg.solve(q_total, [0, 0, 0, 1])
            except np.linalg.LinAlgError:
                v_optimal = np.linalg.pinv(q_total) @ [0, 0, 0, 1]

            v_error = sum(np.dot(quadrics[vertex] @ v_optimal, v_optimal) for vertex in face)
            error_metric[i] = v_error * np.exp(max(self.curvatures[face]))

            face_index[i] = i

        return error_metric, face_index

    def apply_edge_flip(self, face_index, quadrics):
        face = self.mesh.faces[face_index]
        edge = self.mesh.edges_sorted[face_index]

        if self.texture_boundary.predict([self.mesh.visual.vertex_colors[edge[0], :3]])[0] != \
                self.texture_boundary.predict([self.mesh.visual.vertex_colors[edge[1], :3]])[0]:
            return

        new_vertex = (self.mesh.vertices[edge[0]] + self.mesh.vertices[edge[1]]) / 2

        quadrics[edge[0]] = quadrics[edge[1]] = quadrics[face[0]] + quadrics[face[1]]
        self.mesh.vertices[edge[0]] = self.mesh.vertices[edge[1]] = new_vertex
        self.mesh.visual.vertex_colors[edge[0]] = self.mesh.visual.vertex_colors[edge[1]] = \
            (self.mesh.visual.vertex_colors[edge[0]] + self.mesh.visual.vertex_colors[edge[1]]) / 2
        



simplifier = QEM4VR_Simplifier('Registration Cases/Case_01_CTA_PT00109_20101117.obj')

# Print the original number of vertices and faces
print("Original")
print("Number of vertices: ", len(simplifier.mesh.vertices))
print("Number of faces: ", len(simplifier.mesh.faces))

# Perform the simplification
simplified_mesh = simplifier.simplify_by_qem(0.00005)

# Print the simplified number of vertices and faces
print("Simplified")
print("Number of vertices: ", len(simplified_mesh.vertices))
print("Number of faces: ", len(simplified_mesh.faces))

# Display the original and simplified meshes
simplifier.mesh.show()
simplified_mesh.show()