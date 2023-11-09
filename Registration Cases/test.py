import trimesh
import numpy as np
mesh = trimesh.load_mesh("/Users/ziga/Desktop/BMI-LaboratorijskeVaje 2 python/Case_17_CTA_PT00157_20080725.obj")
vertex_colors = mesh.visual.vertex_colors
vessels = mesh.visual.vertex_colors[:,0]==0 #ARRAY kere točke spadajo k zilju
anevrism = mesh.visual.vertex_colors[:,0]>0 # ARRAY, kere tocke spadajo k anevrizmi. 