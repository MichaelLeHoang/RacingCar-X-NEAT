# Racing Car X NEAT

An interactive Pygame application for training NEAT-controlled cars, racing a ten-level campaign, managing saved models, and building reusable custom tracks.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python RacingCar.py
```

The window is resizable and uses a logical 1280×800 layout.

## Main modes

- **Get Started** — drag a saved model from the inventory onto an unlocked campaign level. Completing a level unlocks the next one globally.
- **Train** — configure one or more named car models and select unlocked campaign/custom tracks, then run generations continuously until Stop is pressed or a genome finishes.
- **Custom Tracks** — place snapping straight, corner, and start/finish pieces extracted from the original `track.png`. Components are reused at every angle by rotating them at runtime.

First Lap in both Campaign and Train uses the exact previous-version bitmap track, border, finish line, scale, and spawn geometry.

## Track editor controls

- Left click places or selects a piece.
- Right click deletes a piece.
- **R** rotates the selected or next piece clockwise by 90 degrees.
- **Delete/Backspace** removes the selected piece.
- **Ctrl/Cmd+Z** undoes the last edit.
- The pill-shaped Rotate, Delete, Clear, Undo, Test, and Save controls provide mouse alternatives.

A saved track must contain exactly one start/finish piece, at least eight pieces, matched joins, and one connected non-branching loop. Checkpoints are generated automatically.

## Models and storage

- Champions are saved as versioned, non-executable `.rcmodel` JSON files.
- Custom tracks are saved as versioned `.rctrack` JSON files.
- Get Started and the saved-model inventory read from the same model collection. The inventory count and pagination expose every saved model.
- Select a model and use **Rename** or **Delete**; right-clicking a card is also a rename shortcut. Deletion asks for confirmation.
- Hover a model and press **Export** to copy it to the exports folder.
- Place `.rcmodel` or `.rctrack` files in the imports folder and press **Import**.
- Local data defaults to `~/.racing_car_x_neat`; set `RACING_DATA_DIR` to override it.

All car sprites are normalized to 32×64 with transparent backgrounds. Each car type exposes saved performance stats for maximum speed, acceleration, and turning while controllers keep the same five normalized inputs and four actions.

## Continuous training

- **Start Training** automatically advances through generations and rotates across selected tracks.
- Training completes immediately when the first genome returns to and overlaps the finish-line pixel mask; that finisher becomes the saveable best model.
- **Stop Training** immediately ends the partial generation and keeps its best genome available for saving.
- Finishing or stopping opens a confirmation panel with **Save Best Model** and **Not Now** actions.
- Space pauses or resumes without advancing simulation time.
- The 1×, 2×, 4×, and Max pills change execution speed without changing simulation behavior.
- Training stops automatically when the champion validates on all selected tracks; it can then be saved or restarted.

Use the **+** button to create up to six independent training-model configurations. Each tab retains its own name, car type, stats, selected tracks, population, generation, and current champion.

The Train screen reports live generation, best fitness, selected and passed tracks, run status, and car performance. Saved-model inventory cards retain these training results alongside speed, acceleration, turning, and campaign wins.

Track selection uses cached snapshots of the actual legacy, campaign, or custom track instead of text-only buttons.

When dragging an inventory model, the car and model name follow the pointer. Unlocked levels highlight cyan, locked levels highlight red, and invalid drops animate back to inventory.

## Race rules

A car must leave the starting area to arm its lap and then overlap the actual finish-line pixel mask. This prevents an immediate spawn-area completion without relying on approximate bitmap checkpoint positions. Collision, timeout, or failing to move three pixels for five seconds ends the attempt. Completing a campaign level opens a congratulations panel with **Next Level** and **Main Menu** actions; a failed attempt can be handed directly to Train as the seed for a new population.

## Tests

```bash
python -m unittest discover -s tests -v
```

Headless smoke testing is supported with SDL's dummy video/audio drivers.
