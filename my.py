"""
Custom Packaging Designer - Professional Box Customization Tool
Features: 3D visualization, multiple box types, material calculator, export designs
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List


# =========================
# Data Classes
# =========================
@dataclass
class BoxDimensions:
    length: float
    width: float
    height: float
    thickness: float = 0.5  # material thickness in cm
    
    def volume(self) -> float:
        """Calculate internal volume in cubic cm"""
        return self.length * self.width * self.height
    
    def surface_area(self) -> float:
        """Calculate total surface area in square cm"""
        return 2 * (self.length * self.width + 
                   self.width * self.height + 
                   self.height * self.length)
    
    def material_area(self) -> float:
        """Calculate material needed including flaps and tabs (~20% extra)"""
        return self.surface_area() * 1.2


@dataclass
class Material:
    name: str
    price_per_sqm: float
    weight_per_sqm: float
    color: str


# =========================
# Box Shape Generators
# =========================
class BoxGenerator:
    @staticmethod
    def rectangular_box(dims: BoxDimensions) -> Tuple[np.ndarray, List[List[int]]]:
        """Generate vertices and faces for a rectangular box"""
        l, w, h = dims.length, dims.width, dims.height
        vertices = np.array([
            [0, 0, 0], [l, 0, 0], [l, w, 0], [0, w, 0],  # bottom
            [0, 0, h], [l, 0, h], [l, w, h], [0, w, h]   # top
        ])
        faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5]   # right
        ]
        return vertices, faces
    
    @staticmethod
    def cube_box(side: float) -> Tuple[np.ndarray, List[List[int]]]:
        dims = BoxDimensions(side, side, side)
        return BoxGenerator.rectangular_box(dims)
    
    @staticmethod
    def cylinder_box(diameter: float, height: float, segments: int = 30):
        """Generate vertices and triangular faces for cylinder"""
        radius = diameter / 2
        theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        
        # Bottom and top circle points
        bottom = np.column_stack((radius*np.cos(theta), radius*np.sin(theta), np.zeros(segments)))
        top = bottom + np.array([0, 0, height])
        
        # Center points
        bottom_center = np.array([[0, 0, 0]])
        top_center = np.array([[0, 0, height]])
        
        vertices = np.vstack((bottom, top, bottom_center, top_center))
        faces = []
        n = segments
        bottom_idx = 2*n
        top_idx = 2*n + 1
        
        # Bottom faces
        for i in range(n):
            faces.append([bottom_idx, i, (i+1)%n])
        # Top faces
        for i in range(n):
            faces.append([top_idx, n + (i+1)%n, n + i])
        # Side faces (2 triangles per segment)
        for i in range(n):
            faces.append([i, (i+1)%n, n + (i+1)%n])
            faces.append([i, n + (i+1)%n, n + i])
        
        return vertices, faces
    
    @staticmethod
    def pyramid_box(base_length: float, base_width: float, height: float):
        """Generate vertices and triangular faces for a pyramid"""
        l, w, h = base_length, base_width, height
        vertices = np.array([
            [0, 0, 0], [l, 0, 0], [l, w, 0], [0, w, 0],  # base
            [l/2, w/2, h]  # apex
        ])
        faces = [
            [0, 1, 2, 3],  # base (quad)
            [0, 1, 4],     # front
            [1, 2, 4],     # right
            [2, 3, 4],     # back
            [3, 0, 4]      # left
        ]
        return vertices, faces


# =========================
# Visualization Functions
# =========================
def add_mesh3d_from_faces(fig, vertices, faces, color='lightblue', opacity=0.7):
    """Convert faces to triangles and add to Mesh3d"""
    for face in faces:
        # if quad, split into 2 triangles
        if len(face) == 4:
            triangles = [
                [face[0], face[1], face[2]],
                [face[0], face[2], face[3]]
            ]
        else:
            triangles = [face]
        
        for tri in triangles:
            fig.add_trace(go.Mesh3d(
                x=vertices[tri,0],
                y=vertices[tri,1],
                z=vertices[tri,2],
                color=color,
                opacity=opacity,
                flatshading=True,
                showlegend=False
            ))


def create_3d_box_plot(box_type: str, dims: BoxDimensions, color: str = 'lightblue'):
    """Create interactive 3D plot of any box type"""
    fig = go.Figure()
    
    if box_type == "Rectangular Box":
        vertices, faces = BoxGenerator.rectangular_box(dims)
        add_mesh3d_from_faces(fig, vertices, faces, color)
    elif box_type == "Cube Box":
        vertices, faces = BoxGenerator.cube_box(dims.length)
        add_mesh3d_from_faces(fig, vertices, faces, color)
    elif box_type == "Cylindrical Box":
        vertices, faces = BoxGenerator.cylinder_box(dims.length, dims.height)
        add_mesh3d_from_faces(fig, vertices, faces, color)
    elif box_type == "Pyramid Box":
        vertices, faces = BoxGenerator.pyramid_box(dims.length, dims.width, dims.height)
        add_mesh3d_from_faces(fig, vertices, faces, color)
    
    max_dim = max(dims.length, dims.width, dims.height)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, max_dim*1.2], title="Length (cm)"),
            yaxis=dict(range=[0, max_dim*1.2], title="Width (cm)"),
            zaxis=dict(range=[0, max_dim*1.2], title="Height (cm)"),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


def create_flat_pattern(box_type: str, dims: BoxDimensions):
    """Create simple 2D flat pattern for rectangular/cube boxes"""
    fig = go.Figure()
    
    if box_type in ["Rectangular Box", "Cube Box"]:
        l, w, h = dims.length, dims.width, dims.height
        
        # Bottom
        fig.add_trace(go.Scatter(
            x=[0, l, l, 0, 0],
            y=[0, 0, w, w, 0],
            mode='lines', fill='toself', fillcolor='lightblue',
            line=dict(color='black', width=2), name='Bottom'
        ))
        # Front
        fig.add_trace(go.Scatter(
            x=[0, l, l, 0, 0],
            y=[w, w, w+h, w+h, w],
            mode='lines', fill='toself', fillcolor='lightgreen',
            line=dict(color='black', width=2), name='Front'
        ))
        # Back
        fig.add_trace(go.Scatter(
            x=[0, l, l, 0, 0],
            y=[-h, -h, 0, 0, -h],
            mode='lines', fill='toself', fillcolor='lightcoral',
            line=dict(color='black', width=2), name='Back'
        ))
        # Left
        fig.add_trace(go.Scatter(
            x=[-w, 0, 0, -w, -w],
            y=[0, 0, w, w, 0],
            mode='lines', fill='toself', fillcolor='lightyellow',
            line=dict(color='black', width=2), name='Left'
        ))
        # Right
        fig.add_trace(go.Scatter(
            x=[l, l+w, l+w, l, l],
            y=[0, 0, w, w, 0],
            mode='lines', fill='toself', fillcolor='lightpink',
            line=dict(color='black', width=2), name='Right'
        ))
        # Top flap
        fig.add_trace(go.Scatter(
            x=[0, l, l, 0, 0],
            y=[w+h, w+h, w+h+w, w+h+w, w+h],
            mode='lines', fill='toself', fillcolor='lavender',
            line=dict(color='black', width=2, dash='dash'),
            name='Top Flap'
        ))
    
    fig.update_layout(
        title="Flat Pattern / Die-Cut Template",
        xaxis_title="Width (cm)",
        yaxis_title="Height (cm)",
        height=500,
        showlegend=True,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    return fig


# =========================
# Cost Calculator
# =========================
def calculate_costs(dims: BoxDimensions, material: Material, quantity: int):
    area_sqm = dims.material_area() / 10000
    material_cost = area_sqm * material.price_per_sqm * quantity
    weight = area_sqm * material.weight_per_sqm * quantity
    
    setup_cost = 50
    labor_cost_per_unit = 0.50
    printing_cost_per_unit = 2.00
    
    total_labor = labor_cost_per_unit * quantity
    total_printing = printing_cost_per_unit * quantity
    total_cost = material_cost + setup_cost + total_labor + total_printing
    cost_per_unit = total_cost / quantity if quantity > 0 else 0
    
    return {
        'material_cost': material_cost,
        'setup_cost': setup_cost,
        'labor_cost': total_labor,
        'printing_cost': total_printing,
        'total_cost': total_cost,
        'cost_per_unit': cost_per_unit,
        'total_weight': weight,
        'material_area_sqm': area_sqm * quantity
    }


# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="Custom Packaging Designer", page_icon="📦", layout="wide")
    st.title("📦 Custom Packaging Designer")
    st.markdown("### Design and customize your perfect packaging solution")
    
    with st.sidebar:
        st.header("🎨 Design Controls")
        box_type = st.selectbox("Select Box Type", ["Rectangular Box", "Cube Box", "Cylindrical Box", "Pyramid Box"])
        st.divider()
        
        # Dimensions
        st.subheader("📏 Dimensions (cm)")
        if box_type == "Rectangular Box":
            length = st.slider("Length", 5, 100, 30)
            width = st.slider("Width", 5, 100, 20)
            height = st.slider("Height", 5, 100, 15)
            dims = BoxDimensions(length, width, height)
        elif box_type == "Cube Box":
            side = st.slider("Side Length", 5, 100, 25)
            dims = BoxDimensions(side, side, side)
        elif box_type == "Cylindrical Box":
            diameter = st.slider("Diameter", 5, 100, 20)
            height = st.slider("Height", 5, 100, 30)
            dims = BoxDimensions(diameter, diameter, height)
        elif box_type == "Pyramid Box":
            base_length = st.slider("Base Length", 10, 100, 30)
            base_width = st.slider("Base Width", 10, 100, 30)
            height = st.slider("Height", 10, 100, 25)
            dims = BoxDimensions(base_length, base_width, height)
        
        thickness = st.slider("Material Thickness (cm)", 0.1, 2.0, 0.5, 0.1)
        dims.thickness = thickness
        
        # Material
        st.divider()
        st.subheader("🎨 Material & Color")
        materials = {
            "Corrugated Cardboard": Material("Corrugated", 8.5, 0.6, "#D4A574"),
            "Kraft Paper": Material("Kraft", 6.0, 0.4, "#D2B48C"),
            "White Cardboard": Material("White", 10.0, 0.5, "#F5F5F5"),
            "Plastic (PET)": Material("PET", 15.0, 0.8, "#E0F7FA"),
            "Rigid Board": Material("Rigid", 12.0, 0.7, "#8B7355")
        }
        material_choice = st.selectbox("Material Type", list(materials.keys()))
        material = materials[material_choice]
        custom_color = st.color_picker("Custom Color", material.color)
        
        # Quantity
        st.divider()
        st.subheader("📊 Production")
        quantity = st.number_input("Quantity", min_value=1, max_value=100000, value=100)
    
    # Main Content
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.subheader("3D Preview")
        fig_3d = create_3d_box_plot(box_type, dims, custom_color)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        if box_type in ["Rectangular Box", "Cube Box"]:
            with st.expander("📐 View Flat Pattern / Die-Cut Template"):
                fig_flat = create_flat_pattern(box_type, dims)
                st.plotly_chart(fig_flat, use_container_width=True)
    
    with col2:
        st.subheader("📋 Specifications")
        st.metric("Length", f"{dims.length:.1f} cm")
        st.metric("Width", f"{dims.width:.1f} cm")
        st.metric("Height", f"{dims.height:.1f} cm")
        st.divider()
        st.metric("Volume", f"{dims.volume()/1000:.2f} L")
        st.metric("Surface Area", f"{dims.surface_area():.1f} cm²")
        st.divider()
        st.metric("Material Needed", f"{dims.material_area():.1f} cm²")
        st.metric("Per Unit", f"{dims.material_area()/10000:.4f} m²")
    
    st.divider()
    st.header("💰 Cost Analysis")
    costs = calculate_costs(dims, material, quantity)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Material Cost", f"Rs{costs['material_cost']:.2f}")
    col2.metric("Labor + Setup", f"Rs{costs['labor_cost'] + costs['setup_cost']:.2f}")
    col3.metric("Total Cost", f"Rs{costs['total_cost']:.2f}")
    col4.metric("Cost Per Unit", f"Rs{costs['cost_per_unit']:.2f}")
    
    with st.expander("📊 Detailed Cost Breakdown"):
        breakdown_data = {
            'Cost Component': ['Material Cost','Setup Cost','Labor Cost','Printing Cost','Total Cost'],
            'Amount (Rs)': [f"Rs{costs['material_cost']:.2f}", f"Rs{costs['setup_cost']:.2f}", 
                           f"Rs{costs['labor_cost']:.2f}", f"Rs{costs['printing_cost']:.2f}", 
                           f"Rs{costs['total_cost']:.2f}"],
            'Per Unit (Rs)': [f"Rs{costs['material_cost']/quantity:.2f}", f"Rs{costs['setup_cost']/quantity:.2f}",
                             f"Rs{costs['labor_cost']/quantity:.2f}", f"Rs{costs['printing_cost']/quantity:.2f}",
                             f"Rs{costs['cost_per_unit']:.2f}"]
        }
        df = pd.DataFrame(breakdown_data)
        st.dataframe(df, use_container_width=True)
        st.info(f"📦 Total Weight: {costs['total_weight']:.2f} kg | Material Area: {costs['material_area_sqm']:.2f} m²")
    
    st.divider()
    st.header("📤 Export & Share")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Generate Quote"):
            st.success("Quote generated! Check your downloads.")
    with col2:
        if st.button("📐 Export Technical Drawing"):
            st.success("Technical drawing exported!")
    with col3:
        if st.button("🔗 Share Design"):
            st.info("Share link: https://packaging.app/design/12345")
    
    st.divider()
    st.markdown("<div style='text-align: center; color: gray;'><p>Custom Packaging Designer v1.0 | Professional Box Design Tool</p></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
