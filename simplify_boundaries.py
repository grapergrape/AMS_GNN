from pygsp import graphs
from scipy.sparse.linalg import eigs
import numpy as np
from numpy.linalg import LinAlgError
import trimesh
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import os
from os import path


from curvatures import *
#for testing
from old_downsample import *


def get_last_mesh_in_temp_dir(temp_dir):
    # sorts files based on modification time
    files = sorted(os.listdir(temp_dir), key=lambda x: os.path.getmtime(path.join(temp_dir, x)))
    # find last .obj file
    for file in reversed(files):
        if file.endswith(".obj"):
            return os.path.splitext(file)[0]  # Returns filename without extension
    return None


class MeshSimplifier:
    def __init__(self, path):
        self.mesh = None
        self.shape_descriptor = None
        self.probability_matrix = None
        #self.threshold = 0.5   
        self.path = path

    def calculate_shape_diameter(self, mesh, num_rays=100):
        shape_diameters = []

        # For each vertex in the mesh
        for vertex in mesh.vertices:
            
            ray_lengths = []

            # Shoot num_rays rays from each vertex
            for _ in range(num_rays):
                
                # Generate a random direction for the ray
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)  # normalize to a unit vector

                origin = (vertex + np.array([0.001]))[np.newaxis, :]
                direction = direction[np.newaxis, :]

                # Check for intersection between the ray and the mesh
                locations, index_ray, index_tri = mesh.ray.intersects_location(origin, direction)
                
                # If the ray intersects the mesh, calculate the length of the ray
                if len(locations) > 0:
                    ray_lengths.append(np.linalg.norm(locations[0] - vertex))  # compute distance to the first intersection point

            # Average the ray lengths to get the shape diameter for the current vertex
            if ray_lengths:
                shape_diameters.append(np.mean(ray_lengths))
            else:
                shape_diameters.append(0)

        return np.array(shape_diameters)
    
    def calculate_heat_kernel_signatures(self, mesh, time=10):
        """Compute the Heat Kernel Signature (HKS) for each vertex"""
        # Construct a PyGSP graph from the mesh
        adjacency_matrix = mesh.edges_sparse  # The adjacency matrix of the mesh
        coords = mesh.vertices
        graph = graphs.Graph(adjacency_matrix, coords=coords)

        # Compute the graph Laplacian
        graph.compute_laplacian(lap_type='combinatorial')
        # Compute the eigen-decomposition of the Laplacian
        eigvals, eigvecs = eigs(graph.L.astype('float64'), k=100)  # Select the number of eigenvectors. Typically, 100 is a good start

        # Ensure that the eigenvalues are real numbers (imaginary part should be zero)
        eigvals = np.abs(eigvals)
        eigvecs = eigvecs.real
        # Calculate the heat kernel signature
        t = np.linspace(0, time, 100)  # Time values
        hks = np.sum(eigvecs * np.exp(-t * eigvals), axis=1)

        return hks
    
    def compute_fpfh(self, mesh, n_bins=11, radius=0.1):
        normals =  mesh.vertex_normals
        vertices = mesh.vertices
        tree = KDTree(vertices)
        
        # Compute point neighbours in a given radius
        _, indices = tree.query_radius(vertices, radius, return_distance=True)
        
        fpfh_descriptors = []
        for i in range(len(normals)):
            # Convert indices to array of integers
            indices_int = np.array(indices[i], dtype=int)
            
            # Compute the relative position of the neighbours to the considered point
            relative_pos = vertices[indices_int] - vertices[i]
            relative_norm = normals[indices_int] - normals[i]
            
            # Compute the angles needed for the fpfh
            alpha = np.arctan2(np.linalg.norm(np.cross(np.repeat(normals[i][np.newaxis, :], relative_norm.shape[0], axis=0), relative_norm), axis=1), 
                   np.einsum('ij,ij->i', np.repeat(normals[i][np.newaxis, :], relative_norm.shape[0], axis=0), relative_norm))
            phi = np.arctan2(np.linalg.norm(np.cross(np.repeat(normals[i][np.newaxis, :], relative_pos.shape[0], axis=0), relative_pos), axis=1), 
                 np.einsum('ij,ij->i', np.repeat(normals[i][np.newaxis, :], relative_pos.shape[0], axis=0), relative_pos))

            theta = np.arctan2(np.linalg.norm(np.cross(np.repeat(relative_norm, relative_pos.shape[0], axis=0), relative_pos), axis=1), 
                            np.einsum('ij,ij->i', np.repeat(relative_norm, relative_pos.shape[0], axis=0), relative_pos))
            # Compute the fpfh
            fpfh = np.concatenate([np.histogram(alpha, bins=n_bins)[0],
                                np.histogram(phi, bins=n_bins)[0],
                                np.histogram(theta, bins=n_bins)[0]])

            fpfh_descriptors.append(fpfh)
            
        return np.array(fpfh_descriptors)
    
    def calculate_scale_invariant_heat_kernel_signatures(self, mesh, t_min=1, t_max=100):
        """Compute the Scale-Invariant Heat Kernel Signature (SIHKS) for each vertex"""
        # Compute the HKS for each vertex at a range of time values
        t_values = np.logspace(np.log(t_min), np.log(t_max), num=100, base=np.e)
        hks_values = np.array([self.calculate_heat_kernel_signatures(mesh, t) for t in t_values])

        # Integrate over all time values to get the SIHKS
        sihks = np.trapz(hks_values, t_values, axis=0)

        return sihks

            
    def calculate_curvatures(self):
        mean, gaussian_curvature, max , min = calculate_curvatures(self.path)
        return np.array([mean, gaussian_curvature, max, min]).T

    def compute_shape_descriptor(self, mesh):
        """Compute the shape descriptor for each vertex"""
        # PCA transformation
        pca = PCA(n_components=3)
        pca_coordinates = pca.fit_transform(mesh.vertices)

        # Vertex normals
        normals = mesh.vertex_normals

        # Calculate shape diameters, fast point feature histograms, four kinds of curvatures
        shape_diameters = self.calculate_shape_diameter(mesh)
        feature_histograms = self.compute_fpfh(mesh)
        curvatures = self.calculate_curvatures()

        # Calculate heat kernel signatures and scale-invariant heat kernel signatures
        heat_kernel_signatures = self.calculate_heat_kernel_signatures(mesh)
        scale_invariant_heat_kernel_signatures = self.calculate_scale_invariant_heat_kernel_signatures(mesh)

            # Reshape 1D arrays into 2D arrays by adding an extra dimension
        shape_diameters = shape_diameters[:, np.newaxis]
        heat_kernel_signatures = heat_kernel_signatures[:, np.newaxis]
        scale_invariant_heat_kernel_signatures = scale_invariant_heat_kernel_signatures[:, np.newaxis]

        # check if any of the features contain complex numbers and print which list contains them
        if np.iscomplexobj(pca_coordinates) or np.iscomplexobj(normals) or np.iscomplexobj(shape_diameters) or np.iscomplexobj(feature_histograms) or np.iscomplexobj(curvatures) or np.iscomplexobj(heat_kernel_signatures) or np.iscomplexobj(scale_invariant_heat_kernel_signatures):
            print("Complex numbers detected in the feature vectors")
            print("pca_coordinates: ", np.iscomplexobj(pca_coordinates))
            print("normals: ", np.iscomplexobj(normals))
            print("shape_diameters: ", np.iscomplexobj(shape_diameters))
            print("feature_histograms: ", np.iscomplexobj(feature_histograms))
            print("curvatures: ", np.iscomplexobj(curvatures))
            print("heat_kernel_signatures: ", np.iscomplexobj(heat_kernel_signatures))
            print("scale_invariant_heat_kernel_signatures: ", np.iscomplexobj(scale_invariant_heat_kernel_signatures))


            # Concatenate features
        vertex_features = np.concatenate([
                                            pca_coordinates, 
                                            normals, 
                                            shape_diameters, 
                                            feature_histograms, 
                                            curvatures, 
                                            heat_kernel_signatures, 
                                            scale_invariant_heat_kernel_signatures
                                        ], axis=1)

        return vertex_features
    
    def fetch_labels(self):
        """Generate labels by checking the red color component of each vertex"""
        # Access vertex colors directly
        colors = self.mesh.visual.vertex_colors
        labels = (colors[:, 0] > 0).astype(int)  # colors are in RGBA format
        return labels
    
    def compute_probability_matrix(self, shape_descriptor):
        """Train an XGBoost model and calculate the probability matrix"""
        model = XGBRegressor()
        labels = self.fetch_labels()  # Get labels from the vertex colors
        model.fit(shape_descriptor, labels)
        probability_matrix = model.predict(shape_descriptor)

        return probability_matrix

    def calculate_quadrics(self, mesh):
        """Calculate the quadrics for all vertices in the mesh"""
        quadrics = np.zeros((len(mesh.vertices), 4, 4))

        # For each face
        for face in mesh.faces:
            # Calculate the plane of the face
            point = mesh.vertices[face[0]]
            normal = np.cross(mesh.vertices[face[1]] - point, mesh.vertices[face[2]] - point)
            normal /= np.linalg.norm(normal)  # Normalize the normal vector
            d = -np.dot(point, normal)

            # Form the quadric of the face
            Q = np.outer(np.append(normal, d), np.append(normal, d))

            # Accumulate the face quadric to the quadrics of the vertices
            for vertex in face:
                quadrics[vertex] += Q

        return quadrics

    def compute_error_metric(self, vi, vj, quadrics):
        """Compute the new error metric for an edge"""
        qij = self.compute_quadratic_error(vi, vj, quadrics)  
        pi, pj = self.probability_matrix[vi], self.probability_matrix[vj]
        return qij * np.exp(max(pi, pj))
    
    def compute_quadratic_error(self, vi, vj, quadrics):
        """Compute the quadratic error for an edge (vi, vj)"""
        # Calculate the optimized vertex position
        q_total = quadrics[vi] + quadrics[vj]
        try:
            v_optimal = np.linalg.solve(q_total, np.array([0, 0, 0, 1]))
        except LinAlgError:
            # If the quadric is singular, use the pseudo-inverse instead
            v_optimal = np.dot(np.linalg.pinv(q_total), np.array([0, 0, 0, 1]))        
        # Calculate the error from the optimized vertex to both original vertices
        v1_error = np.dot(np.dot(quadrics[vi], v_optimal), v_optimal)
        v2_error = np.dot(np.dot(quadrics[vj], v_optimal), v_optimal)
        
        # Return the sum of the errors
        return v1_error + v2_error
    
    def remove_duplicate_faces(self, faces):
        # Sort each face's vertices by index
        sorted_faces = np.sort(faces, axis=1)

        # Sort the vertices of each face in ascending order, for both original and flipped cases
        sorted_faces_direct = np.sort(faces, axis=1)
        sorted_faces_flipped = np.sort(faces[:, ::-1], axis=1)

        # Stack the original and flipped faces
        all_faces = np.vstack((sorted_faces_direct, sorted_faces_flipped))

        # Get the unique faces
        unique_faces = np.unique(all_faces, axis=0)

        # Due to duplicates, the original and flipped faces might be both present 
        # To remove them, we sort again and keep only unique, ensuring one copy (either original or flipped) is kept
        sorted_unique_faces = np.sort(unique_faces, axis=0)
        final_faces = np.unique(sorted_unique_faces, axis=0)

        return final_faces

    def simplify_mesh(self, mesh, path, threshold=200):
        """Simplify the mesh while preserving the segmentation boundaries"""
        self.mesh = mesh

        self.shape_descriptor = self.compute_shape_descriptor(mesh)
        self.probability_matrix = self.compute_probability_matrix(self.shape_descriptor)

        # Convert to list for easy manipulation
        simplified_vertices = list(mesh.vertices)
        simplified_faces = list(mesh.faces)

        quadrics = self.calculate_quadrics(mesh)

        vertex_replacement_indices = {}  # Tracks which vertices are replaced

        # loop over the edges of the mesh
        for edge_index, (v1, v2) in enumerate(mesh.edges):
            # compute the new error metric for this edge
            error_metric = self.compute_error_metric(v1, v2, quadrics)
            if error_metric < threshold:
                new_vertex = (simplified_vertices[v1] + simplified_vertices[v2]) / 2
                new_vertex_index = len(simplified_vertices)

                # Keep track of the vertices replaced by the new one
                vertex_replacement_indices[v1] = new_vertex_index
                vertex_replacement_indices[v2] = new_vertex_index

                # Replace the vertices in the faces
                for face in simplified_faces:
                    face[face == v1] = new_vertex_index
                    face[face == v2] = new_vertex_index

                # Add the new vertex
                simplified_vertices.append(new_vertex)

        # Correct vertex references in faces
        for face in simplified_faces:
            for i in range(3):
                if face[i] in vertex_replacement_indices:
                    face[i] = vertex_replacement_indices[face[i]]

        # Remove degenerate faces
        simplified_faces = [face for face in simplified_faces if len(set(face)) == 3]

        return trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces)

if __name__ == "__main__":

    # Define directories
    input_dir = 'Registration Cases'
    temp_dir = 'simplfied'

    # Get files in input directory
    files = sorted(os.listdir(input_dir))

    # Get the last converted file
    last_mesh = get_last_mesh_in_temp_dir(temp_dir)
    print(f"Last converted file: {last_mesh}")
    started_conversion = last_mesh is None
    total_files = len(files)

    # Iterate through each file
    for i, file in enumerate(files, start=1):
        if file.endswith('.obj'):
            filename = os.path.splitext(file)[0]  # Filename without extension
            if not started_conversion:
                started_conversion = last_mesh == filename
                continue  # Skip this file as it has been already converted
            print(f"Converting file {i} of {total_files} : {file}")
            full_path = path.join(input_dir, file)
            # Load mesh
            mesh = trimesh.load_mesh(full_path)
            # Simplify mesh
            print(f"Number of faces in the mesh: {len(mesh.faces)}")
            print(f"Number of vertices in the mesh: {len(mesh.vertices)}")

            temp_path = path.join("temp/", f"{filename}_down.obj")
            mesh.export(temp_path)
            # Create an instance of the MeshSimplifier and simplify the mesh
            simplifier = MeshSimplifier(temp_path)
            simplified_mesh = simplifier.simplify_mesh(mesh, full_path)
            #simplified_mesh = downsample_with_knn(simplified_mesh, 1)

            # Save mesh to .obj file in temp directory
            new_path = path.join(temp_dir, f"{filename}_simp.obj")
            simplified_mesh.export(new_path)
            print(f"Simplified mesh exported to: {new_path}")
            print(f"Number of faces in the mesh after: {len(simplified_mesh.faces)}")
            print(f"Number of vertices in the mesh after: {len(simplified_mesh.vertices)}")
            # remove the file in temp directory at temp_path
            os.remove(temp_path)