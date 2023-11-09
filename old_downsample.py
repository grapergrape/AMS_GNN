from sklearn.neighbors import KNeighborsClassifier
import trimesh

def simplify_mesh(mesh, steps):
    """
    Simplify a 3D mesh by reducing the number of faces.

    Args:
        mesh (trimesh.Trimesh): The input 3D mesh.
        steps (int): The number of simplification steps.

    Returns:
        trimesh.Trimesh: The simplified mesh.
    """
    for _ in range(steps):
        # Apply quadratic decimation to reduce the face count by half in each step.
        mesh = mesh.simplify_quadric_decimation(mesh.faces.shape[0] // 2)
    return mesh

def downsample_with_knn(mesh, steps):
    """
    Downsample a 3D mesh while preserving color information using a K-nearest neighbors (KNN) approach.

    Args:
        mesh (trimesh.Trimesh): The input 3D mesh with color information.
        steps (int): The number of downsampling steps.

    Returns:
        trimesh.Trimesh: The downsampled mesh with predicted colors.
    """
    # Step 1: Train a KNN model on how the input mesh is colored.

    # Get the vertices of the original mesh
    original_mesh_vertices = mesh.vertices

    # Assuming color information is stored in vertex_colors
    original_mesh_colors = mesh.visual.vertex_colors

    # Create a KNN classifier with 3 neighbors
    neigh = KNeighborsClassifier(n_neighbors=3)

    # Train the KNN model on the original mesh's vertices and their colors
    neigh.fit(original_mesh_vertices, original_mesh_colors)

    # Step 2: Clone the original mesh
    mesh2 = mesh.copy()

    # Step 3: Perform mesh simplification to reduce the face count
    mesh2 = simplify_mesh(mesh2, steps)

    # Step 4: Get the vertices of the downsampled mesh
    downsampled_mesh_vertices = mesh2.vertices

    # Step 5: Predict color labels for the downsampled mesh using the KNN model
    predicted_labels = neigh.predict(downsampled_mesh_vertices)

    # Step 6: Add the predicted labels to the downsampled mesh
    mesh2.visual.vertex_colors = predicted_labels

    return mesh2
