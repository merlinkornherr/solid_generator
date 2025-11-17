import json
import trimesh
import numpy as np

# -------------------------
# Load JSON files
# -------------------------
with open("platonic.json") as f:
    platonic = json.load(f)

with open("archimedean.json") as f:
    archimedean = json.load(f)

groups = {
    "platonic": platonic,
    "archimedean": archimedean
}

# -------------------------
# Ask user for input
# -------------------------
print("Choose solid group:")
group_choice = input("platonic / archimedean: ").strip().lower()

while group_choice not in groups:
    print("Invalid choice.")
    group_choice = input("platonic / archimedean: ").strip().lower()

solids = groups[group_choice]

print("\nAvailable solids:")
for name in solids.keys():
    print(" -", name)

solid_choice = input("\nChoose solid: ").strip()

while solid_choice not in solids:
    print("Invalid solid.")
    solid_choice = input("Choose solid: ").strip()

edge_length = float(input("Edge length: "))

# -------------------------
# Build + scale the mesh
# -------------------------
vertices = np.array(solids[solid_choice]["vertices"], dtype=float)

# Determine original edge length from hull
tmp_mesh = trimesh.Trimesh(vertices=vertices).convex_hull
edges = tmp_mesh.edges_unique
current_edge_length = np.mean(
    np.linalg.norm(tmp_mesh.vertices[edges[:, 0]] - tmp_mesh.vertices[edges[:, 1]], axis=1)
)

scale_factor = edge_length / current_edge_length
vertices_scaled = vertices * scale_factor

# Construct final convex hull mesh (triangulated)
mesh = trimesh.Trimesh(vertices=vertices_scaled).convex_hull

# -------------------------
# PREVIEW
# -------------------------
preview = input("Show 3D preview? (y/n): ").strip().lower()

if preview == "y":
    print("Opening 3D viewer... Close the window to continue.")
    mesh.show()  # Interactive trimesh viewer

# -------------------------
# Export OBJ
# -------------------------
filename = solid_choice.replace(" ", "_").lower() + ".obj"
mesh.export(filename)

print(f"\nExported OBJ file: {filename}")
print("Done.")
