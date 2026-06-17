# manu-simu

Composite Manufacturing Simulation for Declam Forming Process

## Overview

This application simulates the composite manufacturing process for applying decorative laminate (declam) to contoured parts using a heated forming table. It analyzes material utilization, waste reduction opportunities, and cost optimization through nesting strategies.

## Features

- **Material Utilization Analysis**: Calculate waste percentages and utilization rates
- **Cost Analysis**: Track material costs per part and annual projections
- **Nesting Optimization**: Simulate multiple parts per sheet to reduce waste
- **Interactive Visualization**: View current and optimized layouts
- **Financial Projections**: Multi-year cost and waste projections

## Installation

1. Install required packages:
```cmd
pip install -r requirements.txt
```

Or install individually:
```cmd
pip install streamlit numpy matplotlib
```

## Running the App

Start the Streamlit web app:
```cmd
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

1. Configure parameters in the left sidebar:
   - **Declam Sheet**: Width, height, and cost per m²
   - **Part Dimensions**: Width, length, and curvature radius
   - **Forming Table**: Dimensions, temperature, and pressure
   - **Production Volume**: Annual production quantity

2. Explore the tabs:
   - **Current State**: View current setup and utilization metrics
   - **Optimization**: See nesting optimization potential
   - **Financial Analysis**: Compare costs between current and optimized approaches
   - **Annual Projections**: View multi-year projections

## Process Details

The simulation models the declam forming process where:
- Declam sheets are placed on a heated forming table
- Pressure and heat contour the declam to the part's curved surface
- The table is sized for the full sheet dimension
- Wrinkle prevention is achieved through controlled pressure distribution

## Default Configuration

- **Sheet Size**: 1220mm x 2440mm (standard 4x8 ft)
- **Part Size**: 300mm x 400mm with 150mm curvature radius
- **Forming Table**: 1220mm x 2440mm
- **Temperature**: 80°C
- **Pressure**: 0.5 MPa
- **Annual Production**: 1000 parts/year

## License

MIT License - See LICENSE file for details
