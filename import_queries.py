import os
import time
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import trimesh
from dotenv import load_dotenv
from neo_properties import *

load_dotenv()  

class MeshToNeo4j(object):

    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def add_mesh(self, file_name):
        # Load the original mesh
        original_mesh = trimesh.load_mesh(file_name)

        vertices = original_mesh.vertices

        # Start a new session
        with self._driver.session() as session:
            # Calculate properties
            curvatures = calc_curvature(original_mesh)

            # Iterate over vertices
            for index, v in enumerate(vertices):
                session.run("""
                CREATE (a:Vertex {index: $index, x: $x, y: $y, z: $z, 
                curvature: $curvature})
                """, index=index, x=v[0], y=v[1], z=v[2], curvature=curvatures[index])

            # Extract edges from adjacency structure
            adjacency = original_mesh.vertex_adjacency_graph
            edge_list = adjacency.tolil().rows.tolist()

            # Iterate over edges and create relationships between vertices
            for idx, links in enumerate(edge_list):
                for link in links:
                    session.run("""
                    MATCH (v1:Vertex {index: $v1_index})
                    MATCH (v2:Vertex {index: $v2_index})
                    CREATE (v1)-[:CONNECTS]->(v2)
                    """, v1_index=idx, v2_index=link)

        print("Graph for mesh", file_name, "created")

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

meshToNeo4j = None
while meshToNeo4j is None:
    try:
        meshToNeo4j = MeshToNeo4j(uri, user, password)
    except ServiceUnavailable:
        print("Neo4j is not available yet. Retrying in 5 seconds...")
        time.sleep(5)  # Wait for 5 seconds before trying to connect again

dir_path = 'Registration Cases'
# Get a list of all .obj files in the directory
file_names = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if file_name.endswith('.obj')]
# Loop over all file_names and call the function to add each mesh to the graph
for file_name in file_names:
    meshToNeo4j.add_mesh(file_name)

print("All meshes added to the graph")

meshToNeo4j.close()