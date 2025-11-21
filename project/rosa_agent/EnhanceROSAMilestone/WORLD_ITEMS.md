# Custom Items in sim_house.sdf.xacro World

Based on the world file analysis, here are the custom items placed in the simulation:

## Detection Platforms

### Platform 1
- **Location**: `x=-1.63, y=0.40, z=0.05`
- **Rotation**: `1.57` radians (90 degrees)
- **Size**: `1.2m x 0.3m x 0.1m` (gray platform)
- **Items on Platform**:
  1. **Banana for Scale**
     - Position: `x=-1.61, y=0.73, z=0.09`
     - Source: `https://fuel.gazebosim.org/1.0/mjcarroll/models/Banana for Scale`
  
  2. **JanSport Backpack Red**
     - Position: `x=-1.60, y=-0.05, z=0.09`
     - Source: `https://fuel.gazebosim.org/1.0/OpenRobotics/models/JanSport Backpack Red`

### Platform 2
- **Location**: `x=1.65, y=2.59, z=0.05`
- **Rotation**: `1.57` radians (90 degrees)
- **Size**: `1.2m x 0.3m x 0.1m` (gray platform)
- **Items on Platform**:
  1. **Eat to Live Book**
     - Position: `x=1.65, y=2.74, z=0.15`
     - Rotation: `1.57` radians (90 degrees - rotated)
     - Source: `https://fuel.gazebosim.org/1.0/GoogleResearch/models/Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book`
  
  2. **Water Bottle**
     - Position: `x=1.65, y=2.44, z=0.15`
     - Source: `https://fuel.gazebosim.org/1.0/iche033/models/Water Bottle`

### Platform 3
- **Location**: `x=5.19, y=2.80, z=0.05`
- **Rotation**: `1.57` radians (90 degrees)
- **Size**: `1.2m x 0.3m x 0.1m` (gray platform)
- **Items on Platform**:
  1. **Birthday Cake**
     - Position: `x=5.19, y=2.95, z=0.15`
     - Source: `https://fuel.gazebosim.org/1.0/chapulina/models/Birthday cake`
  
  2. **RoboCup 3D Simulation Ball**
     - Position: `x=5.19, y=2.65, z=0.15`
     - Source: `https://fuel.gazebosim.org/1.0/OpenRobotics/models/RoboCup 3D Simulation Ball`

## Summary

- **3 Detection Platforms** (gray rectangular platforms, 1.2m x 0.3m)
- **6 Objects** placed on platforms:
  - Banana for Scale
  - JanSport Backpack Red
  - Eat to Live Book
  - Water Bottle
  - Birthday Cake
  - RoboCup 3D Simulation Ball

All platforms are static and positioned at `z=0.05` (5cm above ground), with items placed at `z=0.09-0.15` (on top of platforms).

## Map Coordinates Reference

The world uses a coordinate system where:
- The `sim_house` model is positioned at `x=1.48679, y=1.14707`
- Platforms are positioned relative to this coordinate system
- All coordinates are in meters

## Notes

- All items are loaded from Gazebo Fuel (online model repository)
- Platforms are static (non-movable) objects
- Items are placed on top of platforms for detection/navigation tasks
- The world also includes the standard `sim_house` model with walls forming a maze-like structure


