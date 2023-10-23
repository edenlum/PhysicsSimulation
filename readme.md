# Fluid Simulation
This is a fluid simulation written in Python usng the Pygame library. The simulation uses a grid of density cells, and the velocities on the faces of the cells

There are options to add gravity, incompressibility, smoke, walls, wind, set up a wind tunnel and push particles away.

## Installation
pip install -r requirements.txt

## Usage
python sim.py

Use buttons to change modes. 
Adding wind will create a constant wind to the right from the cell you click on.
Adding walls will create gray cells that have 0 velocity on their faces.
Added smoke will be shown in the green channel. 
Density is shown in the blue and red channels, representing high and low density respectively.
Space will pause the simulation.
