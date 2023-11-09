import numpy as np
import trimesh
import networkx as nx
import scipy.integrate as spi
from distribution_probabilities import exponential_gamma
from old_downsample import *
from itertools import combinations

class GraphConstructor:
    def __init__(self, mesh_path, threshold):
        self.mesh = trimesh.load_mesh(mesh_path)
        self.mesh = downsample_with_knn(self.mesh, 2)
        self.threshold = threshold
        self.edge_weights = None
        self.wasserstein_distance = None
        self.ricci_curvatures = None

    def projection_onto_plane(self, v, n):
        "Project vector v onto plane defined by normal vector n"
        return v - n * np.dot(v, n) / np.dot(n, n)

    def calculate_edge_weights(self, k=8):
        curvature = {}
        weight_matrix = {}

        # Calculate the normals for the vertices
        vertex_normals = self.mesh.vertex_normals

        # Iterate over each vertex
        for i, vi in enumerate(self.mesh.vertices):
            # Get the normal vector for the current vertex
            ni = vertex_normals[i]

            # Get the indices of the vertices adjacent to the current vertex
            adj_indices = self.mesh.vertex_neighbors[i]

            # Create a dictionary to hold the curvature values for each adjacent vertex
            curvature[i] = np.zeros((k, len(adj_indices)))
            weight_matrix[i] = {}

            for idx, q in enumerate(adj_indices):
                # Get the position of the adjacent vertex
                viq = self.mesh.vertices[q]

                # Calculate the edge vector
                eiiq = viq - vi
                eiiq /= np.linalg.norm(eiiq)  # Normalize the edge vector

                # Calculate the curvature along the edge
                ciiq = np.dot(ni, eiiq)

                # Generate k random directions and project them onto the plane orthogonal to eiiq
                directions = np.random.randn(k, 3)
                # Normalize these directions
                directions = np.apply_along_axis(lambda v: self.projection_onto_plane(v, eiiq), 1, directions)
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)

                # Store the curvature value
                curvature[i][:, idx] = ciiq * np.linalg.norm(directions, axis=1)

        # After we calculated all the curvatures, we'll calculate the edge weights
        for i in curvature.keys():
            for q in self.mesh.vertex_neighbors[i]:
                # Ensure that we have calculated the curvature for the vertex q
                if q in curvature:
                    Di = curvature[i]
                    Diq = curvature[q][:, :len(self.mesh.vertex_neighbors[q])]  # Ensure dimensionality matches

                    # Create projection matrices A and A_q
                    A = np.random.randn(k, Di.shape[1])  # Or use a custom method
                    A_q = np.random.randn(k, Diq.shape[1])  # Or use a custom method

                    # Calculate aligned directional curvatures
                    proj_Di = np.matmul(A, Di.T)
                    proj_Diq = np.matmul(A_q, Diq.T)

                    # Only keep positive projections
                    proj_Di = np.maximum(0, proj_Di)
                    proj_Diq = np.maximum(0, proj_Diq)

                    # Compute curvature difference
                    Siiq = np.linalg.norm(proj_Di - proj_Diq)

                    # Store the edge weight
                    weight_matrix[i][q] = Siiq

        self.edge_weights = weight_matrix

    def adaptive_clustering_based_on_ricci_flow(self):
        # Create a weighted graph from the edge weights
        G = nx.Graph()
        for vi, neighbors in self.edge_weights.items():
            for vj, weight in neighbors.items():
                G.add_edge(vi, vj, weight=weight)

        # Define a function that calculates the cost of the shortest path between any two vertices
        def calculate_shortest_path_cost(vi, vj):
            return nx.dijkstra_path_length(G, vi, vj)

        def calculate_wasserstein_distance(vertex_i, vertex_j):
            X = list(G.neighbors(vertex_i))
            Y = list(G.neighbors(vertex_j))

            min_cost = np.inf
            for x in X:
                for y in Y:
                    cost = calculate_shortest_path_cost(x, y)  # Use the calculate_shortest_path_cost function
                    gamma = exponential_gamma(x, y)  # Assumes that this returns a float

                    integral_cost, error = spi.quad(lambda x: cost, 0, 1) 
                    integral_cost *= gamma
                    min_cost = min(integral_cost, min_cost)

            return min_cost

        self.wasserstein_distance = calculate_wasserstein_distance

    def compute_ollivier_ricci_curvature(self):
        ricci_curvatures = {}

        # For each vertex vi
        for vi, neighbors in self.edge_weights.items():
            ricci_curvatures[vi] = {}

            # For every adjacent vertex vj
            for vj, sij in neighbors.items():

                # Calculate Wasserstein distance W(vi, vj)
                wasserstein_distance = self.wasserstein_distance(vi, vj)

                # Calculate the Ollivier-Ricci curvature
                kappa = 1 - (wasserstein_distance / sij)

                # Store the curvature value
                ricci_curvatures[vi][vj] = kappa

        self.ricci_curvatures = ricci_curvatures

    def update_edge_weights(self, alpha):
        updated_edge_weights = {}

        # For each vertex vi
        for vi, neighbors in self.edge_weights.items():
            updated_edge_weights[vi] = {}

            # For each neighboring vertex vj
            for vj, sij in neighbors.items():
                # Compute the new edge weight
                new_sij = (1 - self.ricci_curvatures[vi][vj]**(alpha - 1)) * sij

                # Store the new edge weight
                updated_edge_weights[vi][vj] = new_sij

        self.edge_weights = updated_edge_weights

    def create_patches(self):
        patches = {}
        patch_id = 1

        # For each vertex
        for vi in self.ricci_curvatures.keys():
            # For each neighboring vertex
            for vj in self.ricci_curvatures[vi].keys():
                # Check the curvature and decide whether to cluster the vertices together
                curvature = self.ricci_curvatures[vi][vj]

                # If the curvature is above the threshold, add to the same patch
                if curvature > self.threshold:
                    if vi not in patches:
                        patches[vi] = [vi]

                    patches[vi].append(vj)

                # If the curvature is below the threshold, create a new patch
                elif curvature < -self.threshold:
                    patch_id += 1
                    patches[patch_id] = [vj]

        return patches

    def construct_hierarchical_structure(self, patches):
        # Create a new graph to hold the hierarchical structure
        H = nx.Graph()

        # Create source nodes first
        for source in self.edge_weights.keys():
            H.add_node(source, layer='source', pos=self.mesh.vertices[source])

        # Get per-patch information after creating source nodes
        patch_centroids = {patch: np.mean([H.nodes[source]['pos'] for source in sources], axis=0) for patch, sources in patches.items()}
        patch_sizes = {patch: len(sources) for patch, sources in patches.items()}

        # Create patch nodes
        for patch in patches:
            H.add_node(patch, layer='patch', pos=patch_centroids[patch], size=patch_sizes[patch])

        # Create the global node
        H.add_node('global', layer='global', pos=np.mean([data['pos'] for node, data in H.nodes(data=True)], axis=0))

        # Create edges within each patch
        for patch, sources in patches.items():
            for source in sources:
                H.add_edge(patch, source, weight=np.linalg.norm(H.nodes[patch]['pos'] - H.nodes[source]['pos']))

        # Create edges between patches
        for patch1, patch2 in combinations(patches.keys(), 2):
            # Calculate the edge weight as the sum of the weights of all edges between source nodes across these two patches (as per your own definition)
            weight = 1
            H.add_edge(patch1, patch2, weight=weight)

        # Add edges to the global node
        N = len(H.nodes)
        for node in H.nodes:
            if node != 'global':
                H.add_edge('global', node, weight=1/N)

        return H

    def construct_graph(self):
        self.calculate_edge_weights()
        self.adaptive_clustering_based_on_ricci_flow()
        self.compute_ollivier_ricci_curvature()
        for alpha in range(1, 41):
            self.update_edge_weights(alpha)
        patches = self.create_patches()
        hierarchical_structure = self.construct_hierarchical_structure(patches)
        return hierarchical_structure

if __name__ == "__main__":
    mesh_path = 'Registration Cases/Case_01_CTA_PT00109_20101117.obj'
    threshold = 0.5
    graph_constructor = GraphConstructor(mesh_path, threshold)
    hierarchical_structure = graph_constructor.construct_graph()
