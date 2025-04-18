# Racing Car AI with NEAT 🏎️

## Project Overview

This project implements a racing car game where an AI, trained using the NeuroEvolution of Augmenting Topologies (NEAT) algorithm, learns to navigate a track. The AI controls a car, making decisions to rotate and move forward to avoid track borders and reach the finish line. The game is built using Pygame for rendering and NEAT-Python for evolving neural networks.

## Features

- **Game Mechanics**: A car navigates a predefined track, avoiding borders and aiming for the finish line.
- **AI Training**: Utilizes the NEAT algorithm to evolve neural networks that control the car's movements.
- **Graphics**: Uses Pygame to render the car, track, grass background, and finish line with smooth rotation animations.
- **Collision Detection**: Implements mask-based collision detection for track borders and the finish line (partially implemented).
- **Fitness Evaluation**: The AI's fitness is based on survival time, with potential bonuses for reaching the finish line and penalties for collisions (partially implemented).

## Requirements

- Python 3.x
- Pygame
- NEAT-Python
- Image assets (car, track, track border, finish line, grass) located in the `imgs` directory

## Installation

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:

   ```bash
   pip install pygame neat-python
   ```

3. **Prepare Assets**: Ensure the following image files are in the `imgs` directory:

   - `RedCar.png` (red car image, unused in current implementation)
   - `WhiteCar.png` (white car image, used for the AI-controlled car)
   - `track.png` (track image)
   - `track-border.png` (track border for collision detection)
   - `finish.png` (finish line image)
   - `grass.jpg` (background grass image)

4. **Configure NEAT**: Ensure the `config_feedforward.txt` file is in the project directory. This file defines the NEAT algorithm's parameters, such as population size and mutation rates.

## Usage

1. **Run the Game**:

   ```bash
   python racing_car_ai.py
   ```

   The script will load the NEAT configuration, initialize the population, and start training the AI. The game window will display multiple cars (each controlled by a neural network) attempting to navigate the track.

2. **Training Process**:

   - The NEAT algorithm evolves neural networks over generations (number of generations configurable in the script).
   - Each generation, cars are evaluated based on their fitness (currently based on survival time, with collision and finish line logic partially implemented).
   - The best-performing neural network is retained, and the population evolves through mutation and crossover.

3. **Controls**:

   - No manual controls are available, as the AI fully controls the cars.
   - Close the Pygame window to stop the simulation.

## File Structure

```
RacingCar/
├── imgs/
│   ├── track.png
│   ├── track-border.png
│   ├── finish.png
│   ├── grass.jpg
├── RedCar.png
├── WhiteCar.png
├── main.py
└── config_feedforward.txt
```

## How It Works

- **Properties Class**: Base class managing the car's properties, including velocity, rotation, acceleration, and movement physics. It handles rotation, forward movement, slowing down, and collision detection.
- **Car Class**: Inherits from `Properties`, specifying the car's image (`WhiteCar.png`) and starting position.
- **NEAT Integration**: Each car is controlled by a feedforward neural network. Inputs are placeholders for sensor data (e.g., left, right, in_front, left_45, right_45), but the logic is incomplete. Outputs determine car actions (currently unimplemented).
- **Collision Detection**: Uses Pygame's mask-based collision to detect overlaps with the track border and finish line (commented out in the code).
- **Fitness Function**: Cars gain fitness for surviving longer, with planned penalties for collisions and bonuses for reaching the finish line (not fully implemented).

## Notes

- The image paths in the code (e.g., `E:\Python\ML\Game\RacingCar\imgs\`) are hardcoded. Update them to use relative paths (e.g., `os.path.join('imgs', 'WhiteCar.png')`) for portability.
- The AI decision-making logic (`output` conditions in `main`) is incomplete, as the neural network outputs are not mapped to actions (e.g., rotate left, right, or move forward).
- Collision detection and finish line logic are commented out, so the AI currently does not respond to track borders or the finish line.
- The game runs at 60 FPS, but performance may vary with a large population size.

## Future Improvements

- Complete the AI decision-making logic by mapping neural network outputs to car actions (e.g., rotate left, rotate right, move forward).
- Uncomment and refine the collision detection and finish line logic to penalize crashes and reward reaching the finish line.
- Add sensor inputs (e.g., raycasting) to provide the AI with meaningful data about the track and obstacles.
- Implement a manual play mode for human players.
- Add a graphical interface to visualize training progress and neural network performance.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Inspired by classic racing games.
- Built using the NEAT-Python library and Pygame.
