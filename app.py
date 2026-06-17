"""
Composite Manufacturing Simulation - Declam Forming Process
Simulates the declam (decorative laminate) forming process on a heated table
with focus on material utilization and waste reduction.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DeclammSheet:
    """Represents a sheet of declam material."""
    width: float  # mm
    height: float  # mm
    cost_per_sqm: float = 50.0  # USD per square meter
    
    @property
    def area(self) -> float:
        """Area in square millimeters."""
        return self.width * self.height
    
    @property
    def cost(self) -> float:
        """Cost of one sheet."""
        area_sqm = self.area / 1_000_000
        return area_sqm * self.cost_per_sqm


@dataclass
class Part:
    """Represents the composite part being manufactured."""
    length: float  # mm
    width: float  # mm
    curvature_radius: float  # mm (radius of the rounded contour)
    shape_type: str = "irregular"  # "rectangle", "rounded", "irregular"
    shape_efficiency: float = 0.7  # For irregular shapes, approximate % of bounding box
    
    @property
    def bounding_box_area(self) -> float:
        """Bounding box area that the part occupies."""
        return self.length * self.width
    
    @property
    def footprint_area(self) -> float:
        """Actual area needed for the part footprint including curved sections."""
        if self.shape_type == "rectangle":
            return self.length * self.width
        elif self.shape_type == "rounded":
            # Approximate area with rounded corners
            return self.length * self.width * 0.9
        else:  # irregular
            # Use efficiency factor for irregular curved shapes
            return self.length * self.width * self.shape_efficiency


@dataclass
class FormingTable:
    """Represents the heated forming table."""
    length: float  # mm
    width: float  # mm
    temperature: float = 80.0  # Celsius
    pressure: float = 0.5  # MPa
    
    @property
    def area(self) -> float:
        """Table area in square millimeters."""
        return self.length * self.width


class CompositeManuSimulation:
    """Simulation for composite manufacturing process."""
    
    def __init__(self, sheet: DeclammSheet, part: Part, table: FormingTable):
        self.sheet = sheet
        self.part = part
        self.table = table
        
    def calculate_utilization(self) -> dict:
        """Calculate material utilization metrics."""
        # Use bounding box for actual material needed from sheet
        utilization_rate = (self.part.bounding_box_area / self.sheet.area) * 100
        waste_area = self.sheet.area - self.part.bounding_box_area
        waste_percentage = (waste_area / self.sheet.area) * 100
        
        return {
            'sheet_area_mm2': self.sheet.area,
            'part_area_mm2': self.part.footprint_area,
            'waste_area_mm2': waste_area,
            'utilization_rate': utilization_rate,
            'waste_percentage': waste_percentage,
            'cost_per_sheet': self.sheet.cost,
            'cost_per_part': self.sheet.cost  # 1:1 currently, one sheet per part
        }
    
    def calculate_annual_waste(self, parts_per_year: int) -> dict:
        """Calculate annual waste and costs."""
        metrics = self.calculate_utilization()
        
        sheets_needed = parts_per_year
        total_waste_area = metrics['waste_area_mm2'] * sheets_needed
        total_cost = metrics['cost_per_sheet'] * sheets_needed
        waste_cost = total_cost * (metrics['waste_percentage'] / 100)
        
        return {
            'parts_per_year': parts_per_year,
            'sheets_needed': sheets_needed,
            'total_waste_area_m2': total_waste_area / 1_000_000,
            'total_annual_cost': total_cost,
            'annual_waste_cost': waste_cost
        }
    
    def simulate_nesting_optimization(self, num_parts: int = 2) -> dict:
        """
        Simulate multiple parts on one sheet to reduce waste.
        Simple linear nesting simulation.
        """
        # Try to fit multiple parts on one sheet
        parts_per_row = int(self.sheet.width / self.part.width)
        parts_per_col = int(self.sheet.height / self.part.length)
        parts_per_sheet = parts_per_row * parts_per_col
        
        if parts_per_sheet < 1:
            parts_per_sheet = 1
        
        total_part_area = self.part.bounding_box_area * parts_per_sheet
        utilization = (total_part_area / self.sheet.area) * 100
        
        return {
            'parts_per_sheet': parts_per_sheet,
            'utilization_rate': utilization,
            'waste_percentage': 100 - utilization,
            'improvement': utilization - self.calculate_utilization()['utilization_rate']
        }
    
    def visualize_layout(self, show_nesting: bool = False):
        """Visualize the declam sheet and part layout."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Current state - one part per sheet
        ax1.set_title('Current Layout: 1 Part per Sheet', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Width (mm)')
        ax1.set_ylabel('Length (mm)')
        ax1.set_aspect('equal')
        
        # Draw sheet
        sheet_rect = Rectangle((0, 0), self.sheet.width, self.sheet.height,
                               linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax1.add_patch(sheet_rect)
        
        # Draw part (centered on sheet)
        part_x = (self.sheet.width - self.part.width) / 2
        part_y = (self.sheet.height - self.part.length) / 2
        
        # Draw bounding box
        bbox_rect = Rectangle((part_x, part_y), self.part.width, self.part.length,
                              linewidth=2, edgecolor='blue', facecolor='none', 
                              linestyle='--', alpha=0.7, label='Bounding Box')
        ax1.add_patch(bbox_rect)
        
        # Draw actual irregular part shape
        if self.part.shape_type == "irregular":
            # Create an irregular curved shape to represent the actual part
            theta = np.linspace(0, 2*np.pi, 100)
            # Create an organic irregular shape using multiple frequency components
            r_variation = (0.5 + 0.3*np.sin(3*theta) + 0.15*np.cos(5*theta) + 0.1*np.sin(7*theta))
            x_shape = part_x + self.part.width/2 + (self.part.width/2 * r_variation * np.cos(theta))
            y_shape = part_y + self.part.length/2 + (self.part.length/2 * r_variation * np.sin(theta))
            
            ax1.fill(x_shape, y_shape, color='blue', alpha=0.5, label='Actual Part')
            ax1.plot(x_shape, y_shape, color='darkblue', linewidth=2)
        elif self.part.shape_type == "rounded":
            # Simple rounded rectangle
            part_rect = Rectangle((part_x, part_y), self.part.width, self.part.length,
                                  linewidth=2, edgecolor='blue', facecolor='blue', 
                                  alpha=0.5, label='Actual Part')
            ax1.add_patch(part_rect)
        else:
            # Regular rectangle
            part_rect = Rectangle((part_x, part_y), self.part.width, self.part.length,
                                  linewidth=2, edgecolor='blue', facecolor='blue', 
                                  alpha=0.5, label='Actual Part')
            ax1.add_patch(part_rect)
        
        ax1.legend(loc='upper right', fontsize=8)
        
        # Add waste annotation
        metrics = self.calculate_utilization()
        waste_text = f"Waste: {metrics['waste_percentage']:.1f}%\nUtilization: {metrics['utilization_rate']:.1f}%"
        ax1.text(self.sheet.width * 0.05, self.sheet.height * 0.95, waste_text,
                fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_xlim(-50, self.sheet.width + 50)
        ax1.set_ylim(-50, self.sheet.height + 50)
        ax1.grid(True, alpha=0.3)
        
        # Optimized nesting
        if show_nesting:
            nesting = self.simulate_nesting_optimization()
            ax2.set_title(f'Optimized Nesting: {nesting["parts_per_sheet"]} Parts per Sheet',
                         fontsize=12, fontweight='bold')
        else:
            ax2.set_title('Forming Table Configuration', fontsize=12, fontweight='bold')
        
        ax2.set_xlabel('Width (mm)')
        ax2.set_ylabel('Length (mm)')
        ax2.set_aspect('equal')
        
        # Draw sheet
        sheet_rect2 = Rectangle((0, 0), self.sheet.width, self.sheet.height,
                                linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax2.add_patch(sheet_rect2)
        
        if show_nesting:
            # Draw multiple parts in a grid
            nesting = self.simulate_nesting_optimization()
            parts_per_row = int(self.sheet.width / self.part.width)
            parts_per_col = int(self.sheet.height / self.part.length)
            
            for row in range(parts_per_col):
                for col in range(parts_per_row):
                    x = col * self.part.width
                    y = row * self.part.length
                    if x + self.part.width <= self.sheet.width and y + self.part.length <= self.sheet.height:
                        # Draw bounding box
                        bbox = Rectangle((x, y), self.part.width, self.part.length,
                                        linewidth=1, edgecolor='blue', facecolor='none', 
                                        linestyle='--', alpha=0.5)
                        ax2.add_patch(bbox)
                        
                        # Draw irregular shape
                        if self.part.shape_type == "irregular":
                            theta = np.linspace(0, 2*np.pi, 50)
                            r_variation = (0.5 + 0.3*np.sin(3*theta) + 0.15*np.cos(5*theta) + 0.1*np.sin(7*theta))
                            x_shape = x + self.part.width/2 + (self.part.width/2 * r_variation * np.cos(theta))
                            y_shape = y + self.part.length/2 + (self.part.length/2 * r_variation * np.sin(theta))
                            ax2.fill(x_shape, y_shape, color='blue', alpha=0.5)
                        else:
                            part_rect = Rectangle((x, y), self.part.width, self.part.length,
                                                 linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.5)
                            ax2.add_patch(part_rect)
            
            opt_text = f"Waste: {nesting['waste_percentage']:.1f}%\nUtilization: {nesting['utilization_rate']:.1f}%\nImprovement: +{nesting['improvement']:.1f}%"
        else:
            # Show table dimensions
            if self.table.length <= self.sheet.height and self.table.width <= self.sheet.width:
                table_rect = Rectangle((0, 0), self.table.width, self.table.length,
                                      linewidth=2, edgecolor='red', facecolor='red', alpha=0.2)
                ax2.add_patch(table_rect)
            
            opt_text = f"Table: {self.table.width}x{self.table.length} mm\nTemp: {self.table.temperature}°C\nPressure: {self.table.pressure} MPa"
        
        ax2.text(self.sheet.width * 0.05, self.sheet.height * 0.95, opt_text,
                fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax2.set_xlim(-50, self.sheet.width + 50)
        ax2.set_ylim(-50, self.sheet.height + 50)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_report(self, annual_production: int = 1000):
        """Print a comprehensive simulation report."""
        print("=" * 70)
        print("COMPOSITE MANUFACTURING SIMULATION REPORT")
        print("Declam Forming Process Analysis")
        print("=" * 70)
        print()
        
        print("CONFIGURATION:")
        print(f"  Sheet Size: {self.sheet.width} x {self.sheet.height} mm")
        print(f"  Part Size: {self.part.width} x {self.part.length} mm")
        print(f"  Part Curvature Radius: {self.part.curvature_radius} mm")
        print(f"  Table Size: {self.table.width} x {self.table.length} mm")
        print(f"  Forming Temperature: {self.table.temperature}°C")
        print(f"  Forming Pressure: {self.table.pressure} MPa")
        print()
        
        metrics = self.calculate_utilization()
        print("CURRENT MATERIAL UTILIZATION:")
        print(f"  Sheet Area: {metrics['sheet_area_mm2']:,.0f} mm² ({metrics['sheet_area_mm2']/1_000_000:.3f} m²)")
        print(f"  Part Bounding Box: {self.part.bounding_box_area:,.0f} mm² ({self.part.bounding_box_area/1_000_000:.3f} m²)")
        print(f"  Actual Part Area: {self.part.footprint_area:,.0f} mm² ({self.part.footprint_area/1_000_000:.3f} m²)")
        print(f"  Part Shape Efficiency: {self.part.shape_efficiency*100:.0f}% of bounding box")
        print(f"  Waste Area: {metrics['waste_area_mm2']:,.0f} mm² ({metrics['waste_area_mm2']/1_000_000:.3f} m²)")
        print(f"  Utilization Rate: {metrics['utilization_rate']:.2f}%")
        print(f"  Waste Rate: {metrics['waste_percentage']:.2f}%")
        print(f"  Cost per Part: ${metrics['cost_per_part']:.2f}")
        print()
        
        annual = self.calculate_annual_waste(annual_production)
        print(f"ANNUAL PROJECTIONS ({annual_production:,} parts/year):")
        print(f"  Sheets Required: {annual['sheets_needed']:,}")
        print(f"  Total Waste: {annual['total_waste_area_m2']:.2f} m²")
        print(f"  Annual Material Cost: ${annual['total_annual_cost']:,.2f}")
        print(f"  Annual Waste Cost: ${annual['annual_waste_cost']:,.2f}")
        print()
        
        nesting = self.simulate_nesting_optimization()
        print("OPTIMIZATION POTENTIAL (Nesting):")
        print(f"  Parts per Sheet: {nesting['parts_per_sheet']}")
        print(f"  Optimized Utilization: {nesting['utilization_rate']:.2f}%")
        print(f"  Optimized Waste: {nesting['waste_percentage']:.2f}%")
        print(f"  Improvement: +{nesting['improvement']:.2f}%")
        
        if nesting['parts_per_sheet'] > 1:
            optimized_sheets = int(np.ceil(annual_production / nesting['parts_per_sheet']))
            saved_sheets = annual['sheets_needed'] - optimized_sheets
            cost_savings = saved_sheets * self.sheet.cost
            print(f"  Sheets Saved Annually: {saved_sheets:,}")
            print(f"  Annual Cost Savings: ${cost_savings:,.2f}")
        
        print("=" * 70)


def main():
    """Main Streamlit app entry point."""
    
    st.set_page_config(page_title="Composite Manufacturing Simulation", layout="wide", page_icon="🏭")
    
    st.title("🏭 Composite Manufacturing Simulation")
    st.markdown("### Declam Forming Process - Material Utilization Analysis")
    st.markdown("---")
    
    # Sidebar for inputs
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("Declam Sheet")
    sheet_width = st.sidebar.number_input("Sheet Width (mm)", value=1220, min_value=100, max_value=5000, step=10)
    sheet_height = st.sidebar.number_input("Sheet Height (mm)", value=2440, min_value=100, max_value=5000, step=10)
    cost_per_sqm = st.sidebar.number_input("Cost per m² (USD)", value=50.0, min_value=1.0, max_value=500.0, step=5.0)
    
    st.sidebar.subheader("Part Dimensions")
    part_width = st.sidebar.number_input("Part Width (mm)", value=300, min_value=50, max_value=2000, step=10)
    part_length = st.sidebar.number_input("Part Length (mm)", value=400, min_value=50, max_value=3000, step=10)
    curvature_radius = st.sidebar.number_input("Curvature Radius (mm)", value=150, min_value=10, max_value=1000, step=10)
    
    shape_type = st.sidebar.selectbox("Part Shape", ["irregular", "rounded", "rectangle"], 
                                      help="Select the shape of your part")
    
    if shape_type == "irregular":
        shape_efficiency = st.sidebar.slider("Shape Efficiency (%)", 
                                            min_value=30, max_value=100, value=70, step=5,
                                            help="What % of the bounding box does your irregular part actually fill?") / 100
    else:
        shape_efficiency = 0.9 if shape_type == "rounded" else 1.0
    
    st.sidebar.subheader("Forming Table")
    table_width = st.sidebar.number_input("Table Width (mm)", value=1220, min_value=100, max_value=5000, step=10)
    table_length = st.sidebar.number_input("Table Length (mm)", value=2440, min_value=100, max_value=5000, step=10)
    temperature = st.sidebar.number_input("Temperature (°C)", value=80.0, min_value=20.0, max_value=200.0, step=5.0)
    pressure = st.sidebar.number_input("Pressure (MPa)", value=0.5, min_value=0.1, max_value=5.0, step=0.1)
    
    st.sidebar.subheader("Production Volume")
    annual_production = st.sidebar.number_input("Annual Production (parts/year)", value=1000, min_value=1, max_value=100000, step=100)
    
    # Create objects
    sheet = DeclammSheet(width=sheet_width, height=sheet_height, cost_per_sqm=cost_per_sqm)
    part = Part(length=part_length, width=part_width, curvature_radius=curvature_radius, 
                shape_type=shape_type, shape_efficiency=shape_efficiency)
    table = FormingTable(length=table_length, width=table_width, temperature=temperature, pressure=pressure)
    
    # Create simulation
    sim = CompositeManuSimulation(sheet, part, table)
    
    # Get metrics
    metrics = sim.calculate_utilization()
    annual = sim.calculate_annual_waste(annual_production)
    nesting = sim.simulate_nesting_optimization()
    
    # Display key metrics at the top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Material Utilization", f"{metrics['utilization_rate']:.1f}%", 
                 delta=None, delta_color="normal")
    
    with col2:
        st.metric("Waste per Sheet", f"{metrics['waste_percentage']:.1f}%", 
                 delta=None, delta_color="inverse")
    
    with col3:
        st.metric("Cost per Part", f"${metrics['cost_per_part']:.2f}", 
                 delta=None)
    
    with col4:
        st.metric("Annual Waste Cost", f"${annual['annual_waste_cost']:,.0f}", 
                 delta=None, delta_color="inverse")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Current State", "🔧 Optimization", "💰 Financial Analysis", "📈 Annual Projections"])
    
    with tab1:
        st.subheader("Current Manufacturing Setup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Material Utilization")
            st.write(f"**Sheet Area:** {metrics['sheet_area_mm2']:,.0f} mm² ({metrics['sheet_area_mm2']/1_000_000:.3f} m²)")
            st.write(f"**Part Bounding Box:** {part.bounding_box_area:,.0f} mm² ({part.bounding_box_area/1_000_000:.3f} m²)")
            st.write(f"**Actual Part Area:** {part.footprint_area:,.0f} mm² ({part.footprint_area/1_000_000:.3f} m²)")
            st.write(f"**Shape Efficiency:** {part.shape_efficiency*100:.0f}%")
            st.write(f"**Waste Area:** {metrics['waste_area_mm2']:,.0f} mm² ({metrics['waste_area_mm2']/1_000_000:.3f} m²)")
            st.write(f"**Utilization Rate:** {metrics['utilization_rate']:.2f}%")
            st.write(f"**Waste Rate:** {metrics['waste_percentage']:.2f}%")
        
        with col2:
            st.markdown("#### Process Parameters")
            st.write(f"**Table Size:** {table.width} x {table.length} mm")
            st.write(f"**Forming Temperature:** {table.temperature}°C")
            st.write(f"**Forming Pressure:** {table.pressure} MPa")
            st.write(f"**Part Curvature:** {part.curvature_radius} mm radius")
        
        # Visualization
        st.markdown("#### Layout Visualization")
        fig = sim.visualize_layout(show_nesting=False)
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        st.subheader("Nesting Optimization Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Optimization Potential")
            st.write(f"**Parts per Sheet:** {nesting['parts_per_sheet']}")
            st.write(f"**Optimized Utilization:** {nesting['utilization_rate']:.2f}%")
            st.write(f"**Optimized Waste:** {nesting['waste_percentage']:.2f}%")
            st.write(f"**Improvement:** +{nesting['improvement']:.2f}%")
            
            if nesting['parts_per_sheet'] > 1:
                optimized_sheets = int(np.ceil(annual_production / nesting['parts_per_sheet']))
                saved_sheets = annual['sheets_needed'] - optimized_sheets
                cost_savings = saved_sheets * sheet.cost
                
                st.markdown("#### Annual Savings")
                st.write(f"**Sheets Saved:** {saved_sheets:,}")
                st.write(f"**Cost Savings:** ${cost_savings:,.2f}")
                st.write(f"**Waste Reduction:** {annual['total_waste_area_m2'] - (annual['total_waste_area_m2'] * nesting['utilization_rate'] / metrics['utilization_rate']):.2f} m²")
            else:
                st.info("⚠️ Current configuration only fits 1 part per sheet. Consider smaller parts or larger sheets for nesting benefits.")
        
        with col2:
            st.markdown("#### Implementation Notes")
            st.info("""
            **Nesting Benefits:**
            - Reduced material waste
            - Lower material costs
            - Fewer forming cycles
            - Increased throughput
            
            **Considerations:**
            - Part spacing for cutting
            - Heat distribution uniformity
            - Pressure consistency across table
            - Potential for wrinkle formation
            """)
        
        # Optimized visualization
        st.markdown("#### Optimized Layout Visualization")
        fig = sim.visualize_layout(show_nesting=True)
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.subheader("Financial Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Costs")
            st.write(f"**Cost per Sheet:** ${metrics['cost_per_sheet']:.2f}")
            st.write(f"**Cost per Part:** ${metrics['cost_per_part']:.2f}")
            st.write(f"**Annual Material Cost:** ${annual['total_annual_cost']:,.2f}")
            st.write(f"**Annual Waste Cost:** ${annual['annual_waste_cost']:,.2f}")
        
        with col2:
            if nesting['parts_per_sheet'] > 1:
                optimized_sheets = int(np.ceil(annual_production / nesting['parts_per_sheet']))
                optimized_cost = optimized_sheets * sheet.cost
                savings = annual['total_annual_cost'] - optimized_cost
                
                st.markdown("#### Optimized Costs")
                st.write(f"**Sheets Needed:** {optimized_sheets:,} (vs {annual['sheets_needed']:,})")
                st.write(f"**Annual Material Cost:** ${optimized_cost:,.2f}")
                st.write(f"**Annual Savings:** ${savings:,.2f}")
                st.write(f"**ROI Potential:** {(savings/annual['total_annual_cost']*100):.1f}%")
            else:
                st.warning("No optimization available with current dimensions")
        
        # Cost breakdown chart
        st.markdown("#### Cost Breakdown")
        if nesting['parts_per_sheet'] > 1:
            optimized_sheets = int(np.ceil(annual_production / nesting['parts_per_sheet']))
            optimized_cost = optimized_sheets * sheet.cost
            
            cost_data = {
                'Scenario': ['Current', 'Optimized'],
                'Material Cost': [annual['total_annual_cost'], optimized_cost],
                'Waste Cost': [annual['annual_waste_cost'], optimized_cost * (nesting['waste_percentage']/100)]
            }
            
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(cost_data['Scenario']))
            width = 0.35
            
            ax.bar(x - width/2, cost_data['Material Cost'], width, label='Total Material Cost', color='steelblue')
            ax.bar(x + width/2, cost_data['Waste Cost'], width, label='Waste Cost', color='coral')
            
            ax.set_xlabel('Scenario')
            ax.set_ylabel('Cost (USD)')
            ax.set_title('Annual Cost Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(cost_data['Scenario'])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
    
    with tab4:
        st.subheader("Annual Production Projections")
        
        st.markdown("#### Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Parts per Year", f"{annual['parts_per_year']:,}")
            st.metric("Sheets Required", f"{annual['sheets_needed']:,}")
        
        with col2:
            st.metric("Total Waste", f"{annual['total_waste_area_m2']:.1f} m²")
            st.metric("Waste per Part", f"{(annual['total_waste_area_m2']/annual['parts_per_year']):.4f} m²")
        
        with col3:
            st.metric("Total Annual Cost", f"${annual['total_annual_cost']:,.2f}")
            st.metric("Material Cost per Part", f"${annual['total_annual_cost']/annual['parts_per_year']:.2f}")
        
        # Multi-year projection
        st.markdown("#### Multi-Year Projection")
        years = st.slider("Select number of years to project", 1, 10, 5)
        
        projection_data = []
        for year in range(1, years + 1):
            year_parts = annual_production * year
            year_sheets = int(np.ceil(year_parts))
            year_cost = year_sheets * sheet.cost
            year_waste = (metrics['waste_area_mm2'] * year_sheets) / 1_000_000
            
            projection_data.append({
                'Year': year,
                'Parts': year_parts,
                'Sheets': year_sheets,
                'Cost': year_cost,
                'Waste (m²)': year_waste
            })
        
        # Display projection chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        years_list = [d['Year'] for d in projection_data]
        costs = [d['Cost'] for d in projection_data]
        waste = [d['Waste (m²)'] for d in projection_data]
        
        ax1.plot(years_list, costs, marker='o', linewidth=2, color='steelblue')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Cumulative Cost (USD)')
        ax1.set_title('Material Cost Projection')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(years_list, waste, marker='s', linewidth=2, color='coral')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative Waste (m²)')
        ax2.set_title('Material Waste Projection')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Footer
    st.markdown("---")
    st.markdown("*Composite Manufacturing Simulation - Declam Forming Process Analysis*")


if __name__ == "__main__":
    main()
