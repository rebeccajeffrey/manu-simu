"""
Composite Manufacturing Simulation - Declam Forming Process & Planar Composite Feed
Simulates the declam (decorative laminate) forming process on a heated table
with focus on material utilization and waste reduction.
Also simulates planar machine composite feeding with material flow dynamics.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrow
from dataclasses import dataclass
from typing import List, Tuple
from matplotlib.animation import FuncAnimation


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


@dataclass
class CompositeRoll:
    """Represents a roll of composite material for planar feeding."""
    width: float  # mm
    thickness: float  # mm
    density: float = 1600  # kg/m³ typical for carbon fiber composite
    cost_per_kg: float = 40.0  # USD per kg
    roll_length: float = 100000  # mm (100 meters default)
    
    @property
    def area_per_meter(self) -> float:
        """Area per linear meter in mm²."""
        return self.width * 1000
    
    @property
    def weight_per_meter(self) -> float:
        """Weight per linear meter in kg."""
        volume_m3 = (self.width / 1000) * 1.0 * (self.thickness / 1000)
        return volume_m3 * self.density
    
    @property
    def cost_per_meter(self) -> float:
        """Cost per linear meter."""
        return self.weight_per_meter * self.cost_per_kg


@dataclass
class PlanarMachine:
    """Represents a planar composite feed machine."""
    feed_width: float  # mm - working width of the machine
    max_feed_rate: float = 10.0  # m/min
    tension_control: float = 5.0  # N - tension in the material
    temperature: float = 25.0  # Celsius - ambient or preheating
    num_rollers: int = 6  # Number of feed rollers
    cutting_enabled: bool = True
    
    @property
    def feed_rate_mm_per_sec(self) -> float:
        """Convert feed rate to mm/s."""
        return (self.max_feed_rate * 1000) / 60


@dataclass
class FeedingOperation:
    """Represents a single feeding operation/part."""
    length_required: float  # mm - length of material needed
    cutting_waste: float = 10.0  # mm - waste from cutting operation
    
    @property
    def total_material_used(self) -> float:
        """Total material including cutting waste."""
        return self.length_required + self.cutting_waste


@dataclass
class ZeroingTest:
    """Represents a zeroing test with before/after measurements."""
    test_name: str
    starting_length: float
    cut_to: float
    measurements: dict  # {'location': {'before': value, 'after': value}}
    
    def calculate_deltas(self) -> dict:
        """Calculate the difference between before and after for each measurement."""
        deltas = {}
        for location, values in self.measurements.items():
            delta = values['before'] - values['after']
            deltas[location] = delta
        return deltas
    
    def calculate_expected_cut(self) -> float:
        """Calculate expected material removed."""
        return self.starting_length - self.cut_to
    
    def calculate_deviation(self) -> dict:
        """Calculate deviation from expected cut for each measurement."""
        expected = self.calculate_expected_cut()
        deltas = self.calculate_deltas()
        deviations = {}
        for location, delta in deltas.items():
            deviation = delta - expected
            deviations[location] = deviation
        return deviations
    
    def is_zeroed(self, tolerance: float = 0.01) -> tuple[bool, dict]:
        """
        Determine if machine is zeroed within tolerance.
        Returns (is_zeroed, status_by_location)
        """
        deviations = self.calculate_deviation()
        status = {}
        all_within_tolerance = True
        
        for location, deviation in deviations.items():
            within_tolerance = abs(deviation) <= tolerance
            status[location] = {
                'deviation': deviation,
                'within_tolerance': within_tolerance
            }
            if not within_tolerance:
                all_within_tolerance = False
        
        return all_within_tolerance, status


class PlanarFeedSimulation:
    """Simulation for planar composite feed machine."""
    
    def __init__(self, roll: CompositeRoll, machine: PlanarMachine):
        self.roll = roll
        self.machine = machine
        self.current_position = 0.0  # mm
        self.parts_produced = 0
        self.total_waste = 0.0  # mm
        self.zeroing_tests = []  # Store zeroing test results
        
    def simulate_feed_cycle(self, operation: FeedingOperation) -> dict:
        """Simulate a single feed cycle."""
        # Calculate time to feed
        feed_time = operation.length_required / self.machine.feed_rate_mm_per_sec
        
        # Calculate material consumption
        material_used = operation.total_material_used
        self.current_position += material_used
        self.total_waste += operation.cutting_waste
        self.parts_produced += 1
        
        # Calculate costs
        material_used_meters = material_used / 1000
        cycle_cost = material_used_meters * self.roll.cost_per_meter
        
        return {
            'feed_time_sec': feed_time,
            'material_used_mm': material_used,
            'material_used_m': material_used_meters,
            'cutting_waste_mm': operation.cutting_waste,
            'cycle_cost': cycle_cost,
            'current_position_m': self.current_position / 1000,
            'parts_produced': self.parts_produced
        }
    
    def calculate_production_metrics(self, parts_per_day: int, operation: FeedingOperation) -> dict:
        """Calculate daily/annual production metrics."""
        single_cycle = self.simulate_feed_cycle(operation)
        
        # Reset for calculation
        self.current_position = 0.0
        self.parts_produced = 0
        self.total_waste = 0.0
        
        # Daily metrics
        daily_material = single_cycle['material_used_m'] * parts_per_day
        daily_waste = (operation.cutting_waste / 1000) * parts_per_day
        daily_cost = single_cycle['cycle_cost'] * parts_per_day
        daily_production_time = (single_cycle['feed_time_sec'] * parts_per_day) / 3600  # hours
        
        # Annual metrics (assuming 250 working days)
        working_days = 250
        annual_parts = parts_per_day * working_days
        annual_material = daily_material * working_days
        annual_waste = daily_waste * working_days
        annual_cost = daily_cost * working_days
        
        # Calculate utilization
        utilization_rate = (operation.length_required / operation.total_material_used) * 100
        waste_rate = (operation.cutting_waste / operation.total_material_used) * 100
        
        return {
            'parts_per_day': parts_per_day,
            'daily_material_m': daily_material,
            'daily_waste_m': daily_waste,
            'daily_cost': daily_cost,
            'daily_production_hours': daily_production_time,
            'annual_parts': annual_parts,
            'annual_material_m': annual_material,
            'annual_waste_m': annual_waste,
            'annual_cost': annual_cost,
            'utilization_rate': utilization_rate,
            'waste_rate': waste_rate,
            'cost_per_part': single_cycle['cycle_cost']
        }
    
    def visualize_machine(self, operation: FeedingOperation, show_feed_progress: bool = False):
        """Visualize the planar feed machine."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top view - Machine layout
        ax1.set_title('Planar Feed Machine - Top View', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Length (mm)')
        ax1.set_ylabel('Width (mm)')
        ax1.set_aspect('equal')
        
        machine_length = 3000  # mm - typical machine length
        
        # Draw machine frame
        frame = Rectangle((0, 0), machine_length, self.machine.feed_width,
                         linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.2)
        ax1.add_patch(frame)
        
        # Draw feed rollers
        roller_positions = np.linspace(500, machine_length - 500, self.machine.num_rollers)
        for i, pos in enumerate(roller_positions):
            roller = Circle((pos, self.machine.feed_width / 2), 40,
                          color='darkgray', ec='black', linewidth=2, zorder=3)
            ax1.add_patch(roller)
            ax1.text(pos, self.machine.feed_width / 2, f'R{i+1}',
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Draw composite material
        if show_feed_progress:
            # Show material being fed through
            material_start = 200
            material_end = material_start + operation.length_required
            
            # Material before cutting
            material_rect = Rectangle((material_start, (self.machine.feed_width - self.roll.width) / 2),
                                     operation.length_required, self.roll.width,
                                     linewidth=2, edgecolor='blue', facecolor='lightblue', 
                                     alpha=0.6, label='Composite Material')
            ax1.add_patch(material_rect)
            
            # Cutting zone
            cut_position = material_end + 200
            ax1.axvline(x=cut_position, color='red', linewidth=3, linestyle='--', 
                       label='Cutting Position', alpha=0.8)
            
            # Feed direction arrow
            arrow = FancyArrow(material_start - 100, self.machine.feed_width / 2,
                             150, 0, width=50, head_width=100, head_length=50,
                             color='green', alpha=0.7, label='Feed Direction')
            ax1.add_patch(arrow)
        else:
            # Show roll at entry
            roll_circle = Circle((200, self.machine.feed_width / 2), 150,
                               color='lightblue', ec='blue', linewidth=3, alpha=0.6)
            ax1.add_patch(roll_circle)
            ax1.text(200, self.machine.feed_width / 2, 'Roll',
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add machine info
        info_text = f"Feed Width: {self.machine.feed_width} mm\n"
        info_text += f"Feed Rate: {self.machine.max_feed_rate} m/min\n"
        info_text += f"Tension: {self.machine.tension_control} N\n"
        info_text += f"Rollers: {self.machine.num_rollers}"
        
        ax1.text(machine_length * 0.05, self.machine.feed_width * 0.95, info_text,
                fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_xlim(-100, machine_length + 100)
        ax1.set_ylim(-100, self.machine.feed_width + 100)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Side view - Material flow
        ax2.set_title('Material Flow - Side View', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Machine Length (mm)')
        ax2.set_ylabel('Height (mm)')
        
        # Draw machine profile
        machine_height = 500
        profile = Rectangle((0, 0), machine_length, machine_height,
                           linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.2)
        ax2.add_patch(profile)
        
        # Draw roller positions from side
        roller_diameter = 80
        for pos in roller_positions:
            # Roller
            roller_side = Circle((pos, machine_height / 2), roller_diameter / 2,
                                color='darkgray', ec='black', linewidth=2)
            ax2.add_patch(roller_side)
        
        # Draw material path
        material_height = machine_height / 2
        material_path_x = np.linspace(0, machine_length, 100)
        
        # Create a slight wave pattern to show tension/flexibility
        material_path_y = material_height + 5 * np.sin(material_path_x / 200)
        
        ax2.plot(material_path_x, material_path_y, 'b-', linewidth=3, label='Material Path', alpha=0.7)
        ax2.plot(material_path_x, material_path_y - self.roll.thickness, 'b-', linewidth=3, alpha=0.7)
        ax2.fill_between(material_path_x, material_path_y, 
                        material_path_y - self.roll.thickness,
                        color='lightblue', alpha=0.4)
        
        # Add tension indicators
        tension_positions = [1000, 2000]
        for pos in tension_positions:
            ax2.annotate('', xy=(pos, material_height + 80), xytext=(pos, material_height - 80),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax2.text(pos + 50, material_height, f'{self.machine.tension_control}N',
                    fontsize=9, color='red', fontweight='bold')
        
        ax2.set_xlim(-100, machine_length + 100)
        ax2.set_ylim(-50, machine_height + 150)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_production_timeline(self, parts_per_day: int, operation: FeedingOperation):
        """Plot production timeline and material consumption."""
        # Simulate a day of production
        single_cycle = self.simulate_feed_cycle(operation)
        
        # Reset
        self.current_position = 0.0
        self.parts_produced = 0
        self.total_waste = 0.0
        
        # Generate timeline data
        timeline = []
        cumulative_material = []
        cumulative_waste = []
        
        for part in range(parts_per_day):
            cycle = self.simulate_feed_cycle(operation)
            timeline.append(part + 1)
            cumulative_material.append(self.current_position / 1000)  # Convert to meters
            cumulative_waste.append(self.total_waste / 1000)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Material consumption over time
        ax1.plot(timeline, cumulative_material, marker='o', linewidth=2, 
                color='steelblue', label='Total Material Used')
        ax1.plot(timeline, cumulative_waste, marker='s', linewidth=2, 
                color='coral', label='Cumulative Waste')
        ax1.set_xlabel('Part Number')
        ax1.set_ylabel('Material (meters)')
        ax1.set_title('Daily Material Consumption')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency metrics
        utilization = (operation.length_required / operation.total_material_used) * 100
        waste_pct = (operation.cutting_waste / operation.total_material_used) * 100
        
        categories = ['Usable Material', 'Cutting Waste']
        values = [utilization, waste_pct]
        colors = ['steelblue', 'coral']
        
        ax2.pie(values, labels=categories, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Material Utilization per Part')
        
        plt.tight_layout()
        return fig
    
    def add_zeroing_test(self, test: ZeroingTest):
        """Add a zeroing test to the simulation."""
        self.zeroing_tests.append(test)
    
    def visualize_zeroing_test(self, test: ZeroingTest, tolerance: float = 0.01):
        """Visualize zeroing test results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate metrics
        expected_cut = test.calculate_expected_cut()
        deltas = test.calculate_deltas()
        deviations = test.calculate_deviation()
        is_zeroed, status = test.is_zeroed(tolerance)
        
        locations = list(test.measurements.keys())
        before_values = [test.measurements[loc]['before'] for loc in locations]
        after_values = [test.measurements[loc]['after'] for loc in locations]
        delta_values = [deltas[loc] for loc in locations]
        deviation_values = [deviations[loc] for loc in locations]
        
        # Plot 1: Before and After Measurements
        x = np.arange(len(locations))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, before_values, width, label='Before', 
                       color='steelblue', edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, after_values, width, label='After', 
                       color='lightblue', edgecolor='black', linewidth=1.5)
        
        ax1.axhline(y=test.starting_length, color='green', linestyle='--', 
                   linewidth=2, label=f'Starting Length ({test.starting_length}m)', alpha=0.7)
        ax1.axhline(y=test.cut_to, color='red', linestyle='--', 
                   linewidth=2, label=f'Target Length ({test.cut_to}m)', alpha=0.7)
        
        ax1.set_xlabel('Measurement Location', fontweight='bold')
        ax1.set_ylabel('Length (m)', fontweight='bold')
        ax1.set_title(f'{test.test_name} - Before/After Measurements', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(locations)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Material Removed (Delta)
        colors_delta = ['green' if abs(d - expected_cut) <= tolerance else 'orange' 
                       for d in delta_values]
        bars = ax2.bar(locations, delta_values, color=colors_delta, 
                      edgecolor='black', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=expected_cut, color='red', linestyle='--', linewidth=2,
                   label=f'Expected Cut ({expected_cut:.4f}m)', alpha=0.8)
        ax2.axhline(y=expected_cut + tolerance, color='orange', linestyle=':', 
                   linewidth=1.5, alpha=0.6)
        ax2.axhline(y=expected_cut - tolerance, color='orange', linestyle=':', 
                   linewidth=1.5, label=f'Tolerance (±{tolerance}m)', alpha=0.6)
        
        ax2.set_xlabel('Measurement Location', fontweight='bold')
        ax2.set_ylabel('Material Removed (m)', fontweight='bold')
        ax2.set_title('Actual Material Removed vs Expected', fontweight='bold', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, delta in zip(bars, delta_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{delta:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 3: Deviation from Expected
        colors_dev = ['green' if abs(d) <= tolerance else 'red' for d in deviation_values]
        bars = ax3.bar(locations, deviation_values, color=colors_dev, 
                      edgecolor='black', linewidth=1.5, alpha=0.7)
        ax3.axhline(y=0, color='blue', linestyle='-', linewidth=2, label='Zero (Perfect)')
        ax3.axhline(y=tolerance, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
        ax3.axhline(y=-tolerance, color='orange', linestyle='--', linewidth=1.5,
                   label=f'Tolerance (±{tolerance}m)', alpha=0.6)
        
        ax3.set_xlabel('Measurement Location', fontweight='bold')
        ax3.set_ylabel('Deviation (m)', fontweight='bold')
        ax3.set_title('Deviation from Expected Cut', fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, dev in zip(bars, deviation_values):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{dev:.4f}', ha='center', va=va, fontsize=9, fontweight='bold')
        
        # Plot 4: Summary Status
        ax4.axis('off')
        
        # Create status table
        status_text = f"{'='*60}\n"
        status_text += f"ZEROING TEST SUMMARY: {test.test_name}\n"
        status_text += f"{'='*60}\n\n"
        status_text += f"Test Parameters:\n"
        status_text += f"  Starting Length: {test.starting_length:.4f} m\n"
        status_text += f"  Cut To: {test.cut_to:.4f} m\n"
        status_text += f"  Expected Removal: {expected_cut:.4f} m\n"
        status_text += f"  Tolerance: ±{tolerance:.4f} m\n\n"
        
        status_text += f"Results by Location:\n"
        status_text += f"{'-'*60}\n"
        
        for location in locations:
            before = test.measurements[location]['before']
            after = test.measurements[location]['after']
            delta = deltas[location]
            deviation = deviations[location]
            within_tol = status[location]['within_tolerance']
            
            status_symbol = "✓" if within_tol else "✗"
            status_text += f"\n{location}:\n"
            status_text += f"  Before: {before:.4f} m  →  After: {after:.4f} m\n"
            status_text += f"  Removed: {delta:.4f} m (Expected: {expected_cut:.4f} m)\n"
            status_text += f"  Deviation: {deviation:+.4f} m  {status_symbol}\n"
        
        status_text += f"\n{'-'*60}\n"
        status_text += f"\nOVERALL STATUS: "
        
        if is_zeroed:
            status_text += "✓ MACHINE IS ZEROED\n"
            status_color = 'green'
        else:
            status_text += "✗ MACHINE NEEDS ADJUSTMENT\n"
            status_color = 'red'
        
        # Calculate average deviation
        avg_deviation = np.mean(list(deviations.values()))
        max_deviation = max(deviations.values(), key=abs)
        
        status_text += f"\nStatistics:\n"
        status_text += f"  Average Deviation: {avg_deviation:+.4f} m\n"
        status_text += f"  Max Deviation: {max_deviation:+.4f} m\n"
        status_text += f"  Locations Within Tolerance: {sum(s['within_tolerance'] for s in status.values())}/{len(locations)}\n"
        
        ax4.text(0.05, 0.95, status_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.2, edgecolor=status_color, linewidth=2))
        
        plt.tight_layout()
        return fig


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
    st.markdown("### Manufacturing Process Simulations")
    
    # Add simulation type selector
    simulation_type = st.sidebar.radio(
        "Select Simulation Type",
        ["Declam Forming Process", "Planar Feed Machine"],
        help="Choose which manufacturing process to simulate"
    )
    
    st.markdown("---")
    
    if simulation_type == "Declam Forming Process":
        run_declam_simulation()
    else:
        run_planar_feed_simulation()


def run_planar_feed_simulation():
    """Run the planar feed machine simulation."""
    st.markdown("### 🔄 Planar Composite Feed Machine Simulation")
    st.markdown("Analyze material flow, utilization, and production metrics for planar feeding operations")
    
    # Sidebar for inputs
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("Composite Roll")
    roll_width = st.sidebar.number_input("Roll Width (mm)", value=300, min_value=50, max_value=2000, step=10)
    roll_thickness = st.sidebar.number_input("Material Thickness (mm)", value=0.5, min_value=0.1, max_value=10.0, step=0.1)
    density = st.sidebar.number_input("Material Density (kg/m³)", value=1600, min_value=500, max_value=3000, step=50)
    cost_per_kg = st.sidebar.number_input("Cost per kg (USD)", value=40.0, min_value=1.0, max_value=500.0, step=5.0)
    roll_length = st.sidebar.number_input("Roll Length (m)", value=100, min_value=10, max_value=1000, step=10) * 1000
    
    st.sidebar.subheader("Planar Machine")
    feed_width = st.sidebar.number_input("Feed Width (mm)", value=350, min_value=100, max_value=2000, step=10)
    feed_rate = st.sidebar.number_input("Max Feed Rate (m/min)", value=10.0, min_value=0.5, max_value=50.0, step=0.5)
    tension = st.sidebar.number_input("Material Tension (N)", value=5.0, min_value=1.0, max_value=50.0, step=0.5)
    temperature = st.sidebar.number_input("Temperature (°C)", value=25.0, min_value=15.0, max_value=100.0, step=5.0)
    num_rollers = st.sidebar.number_input("Number of Rollers", value=6, min_value=2, max_value=12, step=1)
    cutting_enabled = st.sidebar.checkbox("Cutting Enabled", value=True)
    
    st.sidebar.subheader("Feeding Operation")
    part_length = st.sidebar.number_input("Part Length (mm)", value=500, min_value=50, max_value=5000, step=10)
    cutting_waste = st.sidebar.number_input("Cutting Waste (mm)", value=10, min_value=0, max_value=100, step=1)
    
    st.sidebar.subheader("Production Volume")
    parts_per_day = st.sidebar.number_input("Parts per Day", value=200, min_value=1, max_value=10000, step=10)
    
    # Create objects
    roll = CompositeRoll(width=roll_width, thickness=roll_thickness, density=density,
                         cost_per_kg=cost_per_kg, roll_length=roll_length)
    machine = PlanarMachine(feed_width=feed_width, max_feed_rate=feed_rate,
                           tension_control=tension, temperature=temperature,
                           num_rollers=num_rollers, cutting_enabled=cutting_enabled)
    operation = FeedingOperation(length_required=part_length, cutting_waste=cutting_waste)
    
    # Create simulation
    sim = PlanarFeedSimulation(roll, machine)
    
    # Add zeroing test data
    test1 = ZeroingTest(
        test_name="Test 1",
        starting_length=0.9,
        cut_to=0.85,
        measurements={
            'Front': {'before': 0.946, 'after': 0.8405},
            'Middle': {'before': 0.958, 'after': 0.865},
            'Middle2': {'before': 0.952, 'after': 0.8525},
            'Aft': {'before': 0.9575, 'after': 0.85}
        }
    )
    
    test2 = ZeroingTest(
        test_name="Test 2",
        starting_length=0.85,
        cut_to=0.82,
        measurements={
            'Front': {'before': 0.8405, 'after': 0.8175},
            'Middle': {'before': 0.865, 'after': 0.823},
            'Middle2': {'before': 0.8525, 'after': 0.826},
            'Aft': {'before': 0.85, 'after': 0.83}
        }
    )
    
    sim.add_zeroing_test(test1)
    sim.add_zeroing_test(test2)
    
    metrics = sim.calculate_production_metrics(parts_per_day, operation)
    
    # Display key metrics at the top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Material Utilization", f"{metrics['utilization_rate']:.1f}%")
    
    with col2:
        st.metric("Waste per Part", f"{(operation.cutting_waste):.0f} mm")
    
    with col3:
        st.metric("Cost per Part", f"${metrics['cost_per_part']:.2f}")
    
    with col4:
        st.metric("Daily Production", f"{metrics['parts_per_day']} parts")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Zeroing Analysis", "🔧 Machine Setup", "📊 Production Metrics", "💰 Cost Analysis", "📈 Timeline"])
    
    with tab1:
        st.subheader("Machine Zeroing Test Results")
        
        st.info("""
        **About Zeroing Tests:**
        These tests measure material dimensions before and after cutting to verify the machine is properly calibrated.
        A properly zeroed machine will remove exactly the specified amount of material at all measurement locations.
        """)
        
        # Tolerance selector
        tolerance = st.slider("Tolerance (meters)", min_value=0.001, max_value=0.050, value=0.010, step=0.001, 
                             format="%.3f", help="Acceptable deviation from expected cut")
        
        # Display test selector
        test_names = [test.test_name for test in sim.zeroing_tests]
        selected_test_name = st.selectbox("Select Test", test_names)
        
        # Get selected test
        selected_test = next(test for test in sim.zeroing_tests if test.test_name == selected_test_name)
        
        # Calculate status
        is_zeroed, status = selected_test.is_zeroed(tolerance)
        
        # Display overall status
        if is_zeroed:
            st.success(f"✓ Machine is properly zeroed for {selected_test_name} (all measurements within ±{tolerance:.3f}m tolerance)")
        else:
            st.error(f"✗ Machine needs adjustment for {selected_test_name} (some measurements exceed ±{tolerance:.3f}m tolerance)")
        
        # Display data table
        st.markdown("#### Measurement Data Table")
        
        expected_cut = selected_test.calculate_expected_cut()
        deltas = selected_test.calculate_deltas()
        deviations = selected_test.calculate_deviation()
        
        # Create DataFrame for display
        import pandas as pd
        
        table_data = []
        for location, values in selected_test.measurements.items():
            before = values['before']
            after = values['after']
            delta = deltas[location]
            deviation = deviations[location]
            within_tol = status[location]['within_tolerance']
            status_symbol = "✓" if within_tol else "✗"
            
            table_data.append({
                'Location': location,
                'Before (m)': f"{before:.4f}",
                'After (m)': f"{after:.4f}",
                'Removed (m)': f"{delta:.4f}",
                'Expected (m)': f"{expected_cut:.4f}",
                'Deviation (m)': f"{deviation:+.4f}",
                'Status': status_symbol
            })
        
        df = pd.DataFrame(table_data)
        
        # Style the dataframe
        def highlight_status(row):
            if row['Status'] == '✓':
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)
        
        styled_df = df.style.apply(highlight_status, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_deviation = np.mean(list(deviations.values()))
            st.metric("Average Deviation", f"{avg_deviation:+.4f} m")
        
        with col2:
            max_deviation = max(deviations.values(), key=abs)
            st.metric("Max Deviation", f"{max_deviation:+.4f} m")
        
        with col3:
            passing_count = sum(s['within_tolerance'] for s in status.values())
            total_count = len(status)
            st.metric("Locations Within Tolerance", f"{passing_count}/{total_count}")
        
        # Visualizations
        st.markdown("#### Detailed Analysis")
        fig = sim.visualize_zeroing_test(selected_test, tolerance)
        st.pyplot(fig)
        plt.close()
        
        # Comparison between tests
        if len(sim.zeroing_tests) > 1:
            st.markdown("#### Test Comparison")
            
            comparison_data = []
            for test in sim.zeroing_tests:
                test_is_zeroed, test_status = test.is_zeroed(tolerance)
                test_deviations = test.calculate_deviation()
                avg_dev = np.mean(list(test_deviations.values()))
                max_dev = max(test_deviations.values(), key=abs)
                passing = sum(s['within_tolerance'] for s in test_status.values())
                total = len(test_status)
                
                comparison_data.append({
                    'Test': test.test_name,
                    'Starting Length': f"{test.starting_length:.3f}m",
                    'Cut To': f"{test.cut_to:.3f}m",
                    'Avg Deviation': f"{avg_dev:+.4f}m",
                    'Max Deviation': f"{max_dev:+.4f}m",
                    'Within Tolerance': f"{passing}/{total}",
                    'Status': '✓ Zeroed' if test_is_zeroed else '✗ Needs Adjustment'
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Machine Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Planar Machine Specs")
            st.write(f"**Feed Width:** {machine.feed_width} mm")
            st.write(f"**Max Feed Rate:** {machine.max_feed_rate} m/min ({machine.feed_rate_mm_per_sec:.1f} mm/s)")
            st.write(f"**Material Tension:** {machine.tension_control} N")
            st.write(f"**Operating Temperature:** {machine.temperature}°C")
            st.write(f"**Number of Rollers:** {machine.num_rollers}")
            st.write(f"**Cutting System:** {'Enabled' if machine.cutting_enabled else 'Disabled'}")
        
        with col2:
            st.markdown("#### Composite Material")
            st.write(f"**Roll Width:** {roll.width} mm")
            st.write(f"**Thickness:** {roll.thickness} mm")
            st.write(f"**Density:** {roll.density} kg/m³")
            st.write(f"**Weight per meter:** {roll.weight_per_meter:.3f} kg/m")
            st.write(f"**Cost per meter:** ${roll.cost_per_meter:.2f}/m")
            st.write(f"**Roll Length:** {roll.roll_length/1000:.0f} m")
        
        st.markdown("#### Machine Visualization")
        fig = sim.visualize_machine(operation, show_feed_progress=False)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("#### Active Feed Operation")
        fig = sim.visualize_machine(operation, show_feed_progress=True)
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.subheader("Production Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Daily Production")
            st.write(f"**Parts Produced:** {metrics['parts_per_day']:,} parts")
            st.write(f"**Material Used:** {metrics['daily_material_m']:.2f} m")
            st.write(f"**Material Waste:** {metrics['daily_waste_m']:.2f} m")
            st.write(f"**Production Time:** {metrics['daily_production_hours']:.2f} hours")
            st.write(f"**Daily Cost:** ${metrics['daily_cost']:,.2f}")
            
            # Calculate throughput
            if metrics['daily_production_hours'] > 0:
                parts_per_hour = metrics['parts_per_day'] / metrics['daily_production_hours']
                st.write(f"**Throughput:** {parts_per_hour:.1f} parts/hour")
        
        with col2:
            st.markdown("#### Annual Production (250 days)")
            st.write(f"**Parts Produced:** {metrics['annual_parts']:,} parts")
            st.write(f"**Material Used:** {metrics['annual_material_m']:,.1f} m ({metrics['annual_material_m']/1000:.2f} km)")
            st.write(f"**Material Waste:** {metrics['annual_waste_m']:,.1f} m")
            st.write(f"**Annual Cost:** ${metrics['annual_cost']:,.2f}")
            st.write(f"**Waste Percentage:** {metrics['waste_rate']:.2f}%")
        
        st.markdown("#### Material Flow Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate cycle time
            cycle_time = operation.length_required / machine.feed_rate_mm_per_sec
            st.info(f"""
            **Single Part Cycle:**
            - Feed Time: {cycle_time:.1f} seconds
            - Material Used: {operation.total_material_used:.0f} mm
            - Cutting Waste: {operation.cutting_waste:.0f} mm
            - Cycle Cost: ${metrics['cost_per_part']:.2f}
            """)
        
        with col2:
            # Calculate roll usage
            parts_per_roll = int(roll.roll_length / operation.total_material_used)
            rolls_per_year = int(np.ceil(metrics['annual_parts'] / parts_per_roll))
            
            st.success(f"""
            **Roll Consumption:**
            - Parts per Roll: {parts_per_roll:,}
            - Rolls per Year: {rolls_per_year:,}
            - Roll Utilization: {(parts_per_roll * operation.total_material_used / roll.roll_length * 100):.1f}%
            """)
    
    with tab4:
        st.subheader("Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cost Breakdown")
            st.write(f"**Material Cost per Meter:** ${roll.cost_per_meter:.2f}")
            st.write(f"**Material Cost per Part:** ${metrics['cost_per_part']:.2f}")
            st.write(f"**Daily Material Cost:** ${metrics['daily_cost']:,.2f}")
            st.write(f"**Annual Material Cost:** ${metrics['annual_cost']:,.2f}")
            
            # Calculate waste cost
            waste_cost_per_part = (operation.cutting_waste / 1000) * roll.cost_per_meter
            annual_waste_cost = waste_cost_per_part * metrics['annual_parts']
            
            st.write(f"**Waste Cost per Part:** ${waste_cost_per_part:.2f}")
            st.write(f"**Annual Waste Cost:** ${annual_waste_cost:,.2f}")
        
        with col2:
            st.markdown("#### Optimization Potential")
            
            # Calculate if waste could be reduced
            if cutting_waste > 0:
                optimized_waste = max(0, cutting_waste - 5)  # Assume 5mm improvement
                optimized_total = operation.length_required + optimized_waste
                optimized_utilization = (operation.length_required / optimized_total) * 100
                
                savings_per_part = ((cutting_waste - optimized_waste) / 1000) * roll.cost_per_meter
                annual_savings = savings_per_part * metrics['annual_parts']
                
                st.info(f"""
                **Waste Reduction Scenario:**
                - Target Waste: {optimized_waste:.0f} mm (from {cutting_waste:.0f} mm)
                - New Utilization: {optimized_utilization:.1f}%
                - Savings per Part: ${savings_per_part:.2f}
                - Annual Savings: ${annual_savings:,.2f}
                """)
            else:
                st.success("✓ Already operating with zero cutting waste!")
        
        # Cost comparison chart
        st.markdown("#### Cost Distribution")
        
        material_cost = metrics['cost_per_part']
        waste_cost = waste_cost_per_part
        usable_cost = material_cost - waste_cost
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Per part cost
        costs = [usable_cost, waste_cost]
        labels = ['Usable Material', 'Waste']
        colors = ['steelblue', 'coral']
        
        ax1.pie(costs, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax1.set_title('Cost per Part')
        
        # Annual cost
        annual_usable = usable_cost * metrics['annual_parts']
        annual_waste = annual_waste_cost
        
        categories = ['Usable\nMaterial', 'Waste']
        values = [annual_usable, annual_waste]
        
        bars = ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Cost (USD)')
        ax2.set_title('Annual Cost Breakdown')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab5:
        st.subheader("Production Timeline")
        
        st.markdown("#### Daily Production Analysis")
        
        # Show timeline visualization
        fig = sim.plot_production_timeline(parts_per_day, operation)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("#### Multi-Day Projection")
        days = st.slider("Select number of days to project", 1, 30, 7)
        
        projection_data = []
        for day in range(1, days + 1):
            cumulative_parts = metrics['parts_per_day'] * day
            cumulative_material = metrics['daily_material_m'] * day
            cumulative_cost = metrics['daily_cost'] * day
            cumulative_waste = metrics['daily_waste_m'] * day
            
            projection_data.append({
                'Day': day,
                'Parts': cumulative_parts,
                'Material (m)': cumulative_material,
                'Cost': cumulative_cost,
                'Waste (m)': cumulative_waste
            })
        
        # Display projection chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        days_list = [d['Day'] for d in projection_data]
        parts = [d['Parts'] for d in projection_data]
        material = [d['Material (m)'] for d in projection_data]
        costs = [d['Cost'] for d in projection_data]
        waste = [d['Waste (m)'] for d in projection_data]
        
        ax1.plot(days_list, parts, marker='o', linewidth=2, color='green')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Cumulative Parts')
        ax1.set_title('Parts Production')
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(days_list, parts, alpha=0.3, color='green')
        
        ax2.plot(days_list, material, marker='s', linewidth=2, color='steelblue')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Cumulative Material (m)')
        ax2.set_title('Material Consumption')
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(days_list, material, alpha=0.3, color='steelblue')
        
        ax3.plot(days_list, costs, marker='^', linewidth=2, color='purple')
        ax3.set_xlabel('Day')
        ax3.set_ylabel('Cumulative Cost (USD)')
        ax3.set_title('Cost Accumulation')
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(days_list, costs, alpha=0.3, color='purple')
        
        ax4.plot(days_list, waste, marker='d', linewidth=2, color='coral')
        ax4.set_xlabel('Day')
        ax4.set_ylabel('Cumulative Waste (m)')
        ax4.set_title('Waste Accumulation')
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(days_list, waste, alpha=0.3, color='coral')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Footer
    st.markdown("---")
    st.markdown("*Planar Composite Feed Machine Simulation*")


def run_declam_simulation():
    """Run the declam forming process simulation."""
    st.markdown("### Declam Forming Process - Material Utilization Analysis")
    
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
