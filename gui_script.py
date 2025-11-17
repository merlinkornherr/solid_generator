import json
import os
import sys
import traceback
import numpy as np
import trimesh

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QFileDialog, QMessageBox, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -------------------------
# Utility: load json files
# -------------------------
def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# -------------------------
# Mesh building + scaling
# -------------------------
def build_mesh_from_vertices(vertices):
    # Create trimesh and return its convex hull (triangulated)
    mesh = trimesh.Trimesh(vertices=np.array(vertices, dtype=float), process=False)
    hull = mesh.convex_hull
    return hull

def measure_mean_edge_length(mesh):
    # measure mean of unique edges in given mesh
    edges = mesh.edges_unique
    v = mesh.vertices
    lengths = np.linalg.norm(v[edges[:,0]] - v[edges[:,1]], axis=1)
    return float(np.mean(lengths)), float(np.min(lengths)), float(np.max(lengths)), lengths

def scale_vertices_to_edge(vertices, target_edge_length, max_iter=6, tol_rel=1e-6):
    """
    Scale vertices so that the final mesh mean edge length equals target_edge_length.
    We iterate to correct numerical mismatch (should converge quickly).
    Returns: final_mesh, final_mean, error_percent
    """
    verts = np.array(vertices, dtype=float)
    # initial hull
    mesh = build_mesh_from_vertices(verts)
    mean_e, mn, mx, lengths = measure_mean_edge_length(mesh)
    if mean_e == 0:
        raise ValueError("Current mean edge length is zero, invalid geometry.")
    scale = target_edge_length / mean_e
    verts = verts * scale
    mesh = build_mesh_from_vertices(verts)
    for i in range(max_iter):
        mean_e, mn, mx, lengths = measure_mean_edge_length(mesh)
        rel_err = (mean_e - target_edge_length) / target_edge_length
        if abs(rel_err) <= tol_rel:
            break
        # adjust scale multiplicatively
        corr = target_edge_length / mean_e
        verts = verts * corr
        mesh = build_mesh_from_vertices(verts)
    final_mean, final_min, final_max, final_lengths = measure_mean_edge_length(mesh)
    error_percent = 100.0 * (final_mean - target_edge_length) / target_edge_length
    return mesh, final_mean, final_min, final_max, final_lengths, error_percent

# -------------------------
# GUI Application
# -------------------------
class SolidGeneratorGUI(QWidget):
    def __init__(self, platonic, archimedean):
        super().__init__()
        self.setWindowTitle("Solid Generator — GUI")
        self.resize(1000, 700)

        self.data = {
            "platonic": platonic,
            "archimedean": archimedean
        }

        # UI Elements
        main_layout = QHBoxLayout(self)

        # Left: controls
        controls = QGroupBox("Controls")
        c_layout = QVBoxLayout()
        controls.setLayout(c_layout)

        form = QFormLayout()
        self.group_box = QComboBox()
        self.group_box.addItems(["platonic", "archimedean"])
        form.addRow("Group:", self.group_box)

        self.solid_box = QComboBox()
        form.addRow("Solid:", self.solid_box)

        self.edge_input = QDoubleSpinBox()
        self.edge_input.setDecimals(6)
        self.edge_input.setRange(1e-6, 1e6)
        self.edge_input.setValue(1.0)
        self.edge_input.setSingleStep(0.1)
        form.addRow("Edge length:", self.edge_input)

        self.preview_btn = QPushButton("Preview (build & show)")
        self.export_btn = QPushButton("Export OBJ")

        # Status labels
        self.status_label = QLabel("Ready.")
        self.measured_label = QLabel("Measured mean edge: -")
        self.error_label = QLabel("Error: -")

        c_layout.addLayout(form)
        c_layout.addWidget(self.preview_btn)
        c_layout.addWidget(self.export_btn)
        c_layout.addSpacing(10)
        c_layout.addWidget(self.measured_label)
        c_layout.addWidget(self.error_label)
        c_layout.addStretch()
        c_layout.addWidget(self.status_label)

        # Right: Matplotlib 3D canvas
        self.fig = Figure(figsize=(6,6))
        self.canvas = FigureCanvas(self.fig)

        main_layout.addWidget(controls, 0)
        main_layout.addWidget(self.canvas, 1)

        # Connections
        self.group_box.currentTextChanged.connect(self.update_solids)
        self.preview_btn.clicked.connect(self.on_preview)
        self.export_btn.clicked.connect(self.on_export)

        # init
        self.update_solids(self.group_box.currentText())
        self.current_mesh = None

    def update_solids(self, group_name):
        self.solid_box.clear()
        entries = sorted(self.data[group_name].keys())
        self.solid_box.addItems(entries)

    def build_and_scale(self):
        group = self.group_box.currentText()
        name = self.solid_box.currentText()
        target_edge = float(self.edge_input.value())
        self.status_label.setText(f"Building '{name}' and scaling to edge {target_edge:.6f}...")
        QApplication.processEvents()

        try:
            vertices = self.data[group][name]["vertices"]
            mesh, final_mean, final_min, final_max, final_lengths, err = scale_vertices_to_edge(
                vertices, target_edge, max_iter=8, tol_rel=1e-9
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Fehler beim Erstellen/Skalieren:\n{e}")
            return None

        # update status labels
        self.measured_label.setText(f"Measured mean edge: {final_mean:.9f}  (min {final_min:.9f}, max {final_max:.9f})")
        self.error_label.setText(f"Relative error: {err:.6f} %")
        if abs(err) > 0.01:
            self.status_label.setText("Warning: edge length differs by more than 0.01%")
        else:
            self.status_label.setText("OK: Kantenlänge stimmt (within 0.01%).")
        QApplication.processEvents()

        self.current_mesh = mesh
        return mesh

    def draw_mesh_on_canvas(self, mesh):
        # draw mesh faces using Poly3DCollection
        self.fig.clf()
        ax = self.fig.add_subplot(111, projection='3d')
        verts = mesh.vertices
        faces = mesh.faces
        poly3d = [verts[face] for face in faces]
        collection = Poly3DCollection(poly3d, alpha=0.8, linewidths=0.2)
        collection.set_edgecolor((0.05,0.05,0.05,0.6))
        ax.add_collection3d(collection)

        # autoscale
        all_pts = verts
        xlim = (np.min(all_pts[:,0]), np.max(all_pts[:,0]))
        ylim = (np.min(all_pts[:,1]), np.max(all_pts[:,1]))
        zlim = (np.min(all_pts[:,2]), np.max(all_pts[:,2]))
        max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
        if max_range <= 0:
            max_range = 1.0
        mid_x = 0.5*(xlim[1]+xlim[0])
        mid_y = 0.5*(ylim[1]+ylim[0])
        mid_z = 0.5*(zlim[1]+zlim[0])
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

        ax.set_box_aspect((1,1,1))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        self.canvas.draw()

    def on_preview(self):
        mesh = self.build_and_scale()
        if mesh is not None:
            self.draw_mesh_on_canvas(mesh)

    def on_export(self):
        mesh = self.current_mesh
        if mesh is None:
            # build first
            mesh = self.build_and_scale()
            if mesh is None:
                return
        # ask where to save
        suggested = self.solid_box.currentText().replace(" ", "_").lower() + ".obj"
        path, _ = QFileDialog.getSaveFileName(self, "Save OBJ", suggested, "Wavefront OBJ (*.obj)")
        if not path:
            return
        try:
            mesh.export(path)
            QMessageBox.information(self, "Export", f"Exported OBJ to:\n{path}\n\nMeasured mean edge: {self.measured_label.text()}\n{self.error_label.text()}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Export error", f"Could not export OBJ:\n{e}")

# -------------------------
# Main
# -------------------------
def main():
    app = QApplication(sys.argv)

    # try to load JSONs from current dir
    base = os.getcwd()
    p_path = os.path.join(base, "platonic.json")
    a_path = os.path.join(base, "archimedean.json")

    if not os.path.exists(p_path) or not os.path.exists(a_path):
        QMessageBox.critical(None, "Missing files", f"platonic.json and archimedean.json must be in the working directory:\n{base}")
        return

    try:
        platonic = load_json_file(p_path)
        archimedean = load_json_file(a_path)
    except Exception as e:
        QMessageBox.critical(None, "JSON load error", f"Could not load JSON files:\n{e}")
        return

    w = SolidGeneratorGUI(platonic, archimedean)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
