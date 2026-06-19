# 2D Incompressible Navier–Stokes CFD Solver

## Overview

A custom Computational Fluid Dynamics (CFD) solver developed in Python for the simulation of two-dimensional incompressible flow around immersed geometries including cylinders and NACA airfoils.

The solver combines finite-volume discretisation, higher-order convection schemes, pressure-velocity coupling, and turbulence modelling to investigate aerodynamic performance, flow separation, and stall behaviour across a wide range of Reynolds numbers.

---

## Features

### Numerical Methods

- Finite-volume discretisation of the incompressible Navier–Stokes equations
- MUSCL reconstruction with minmod limiter
- Rusanov (Local Lax-Friedrichs) flux scheme
- Adaptive CFL-based timestep control
- Pressure Poisson equation solver
- Sparse matrix factorisation for efficient pressure solution

### Turbulence Modelling

Implemented the Spalart–Allmaras one-equation turbulence model including:

- Transport equation for modified turbulent viscosity (`nuTilde`)
- Wall-distance computation using Euclidean distance transforms
- Production and destruction source terms
- Cross-diffusion correction term
- Eddy-viscosity formulation
- Turbulent viscosity coupling into momentum equations
- Wall boundary condition enforcement (`nuTilde = 0`)
- Freestream turbulence specification

### Geometry Support

- Circular cylinder
- NACA 0012
- NACA 4412
- Angle of attack support through body rotation
- Immersed-boundary mask generation

### Aerodynamic Analysis

#### Force Analysis

- Lift coefficient (CL)
- Drag coefficient (CD)
- Pressure-force integration
- Time-history force monitoring

#### Flow Visualisation

- Velocity magnitude contours
- Pressure contours
- Streamlines
- Velocity gradients
- Pressure gradients
- Vorticity fields
- Turbulent viscosity fields

#### Aerodynamic Performance Metrics

- Pressure coefficient distributions (Cp)
- Lift curves
- Drag polars
- Separation-point identification
- Stall onset detection

---

## Verification and Validation

### Verification Studies

- Grid-independence studies
- Temporal convergence assessment
- Force coefficient convergence monitoring
- Pressure solver verification
- MUSCL reconstruction verification

### Benchmark Cases

- Lid-driven cavity flow
- Cylinder flow simulations
- Published aerodynamic force coefficient comparisons
- Published pressure distribution comparisons

---

## Aerodynamic Stall Investigation

### Airfoils Studied

- NACA 0012
- NACA 2412
- NACA 4412

### Test Conditions

#### Reynolds Numbers

- 5 × 10⁴
- 5 × 10⁵
- 7.5 × 10⁵
- 2.5 × 10⁶

#### Angle of Attack Range

- 0° to 20°

### Analysis Performed

- Lift coefficient versus angle of attack
- Drag coefficient versus angle of attack
- Pressure coefficient distributions
- Turbulent viscosity field analysis
- Wake structure visualisation
- Flow separation tracking
- Streamline analysis
- Stall onset identification

---

## Key Engineering Outcomes

- Developed a finite-volume Navier–Stokes solver from first principles
- Implemented MUSCL reconstruction and Rusanov flux convection schemes
- Developed a pressure-correction algorithm with sparse-matrix acceleration
- Implemented the Spalart–Allmaras RANS turbulence model
- Investigated turbulent separation and aerodynamic stall
- Performed verification and validation against benchmark and published aerodynamic data
- Generated aerodynamic force, pressure, turbulence, and separation analyses comparable to industry CFD workflows

---

## Future Development

- Additional turbulence models
- Airfoil coordinate-file import capability
- Viscous wall-force integration
- Extended validation database
- Higher Reynolds number studies
- Enhanced post-processing and aerodynamic analysis tools
