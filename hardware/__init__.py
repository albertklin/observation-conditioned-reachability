"""OCR Hardware Deployment Package.

This package provides the tools for deploying the OCR safety filter
on hardware. It includes:

- interfaces/: Abstract base classes for hardware components
- sim/: Simulated implementations for testing
- configs/: YAML configuration files
- run.py: Main entry point for running the safety filter

To deploy on your hardware:
1. Implement the interfaces for your specific hardware
2. Copy and modify configs/default.yaml
3. Run hardware/run.py with your config
"""
