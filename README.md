# Racing Car X NEAT

Racing Car X NEAT is an interactive top-down Pygame racing game built around
neat-python. Train autonomous cars, save their genomes, race a ten-level
campaign, and build tracks from the same semantic components used by collision,
progress tracking, campaign content, and procedural generation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python RacingCar.py
```

The application uses a resizable window with a 1280×800 logical canvas. SDL's
dummy video and audio drivers remain supported for headless tests.

## Game modes

### Play

Play opens Level Selection. Choose one of the available level cards before
selecting a car. Locked cards are dimmed; completed cards show their best time.

The selected circuit then opens in race preparation. Drag a saved model from
the in-race drawer onto the dashed **DROP CAR TO START** area. Browsing and
dragging do not create a simulation car or start the timer. A valid drop starts
a short READY → GO countdown; race time starts at GO.

A Play lap completes when the car returns to and overlaps the actual finish
mask after first leaving the start area. Its initial placement cannot trigger a
win. Ordered gates remain part of training fitness and champion validation. Completion, collision,
stalling, and timeout permanently freeze the attempt time. Results offer the
appropriate Next Level, Retry, Train This Car, and Level Select actions.

Level 1 keeps the repository's original `track.png`, `track-border.png`, finish
line, scale, placement, and spawn direction. Levels 2–10 are stable checked-in
semantic tracks with increasing geometry-derived difficulty.

### Train

Training continues to use NEAT population, species, reproduction, and JSON
genome serialization. It runs as bounded simulation steps from the Pygame main
loop, so Start, Pause, Resume, Stop, 1×, 2×, 4×, and Max remain responsive.

Three distinct modes are available:

- **Original Track** evaluates and validates on the exact legacy bitmap track.
- **Custom Tracks** evaluates every selected valid custom track and requires the
  champion to pass all of them.
- **Random Curriculum** rotates through deterministic generated training seeds
  and difficulty bands. Validation uses three separate held-out seeds.

Fitness rewards ordered path progress, first-time gates, controlled road speed,
lap completion, and faster finishes. It penalizes reverse progress, collisions,
spinning without progress, stalls, timeouts, and segment oscillation. One track
finish records a candidate but does not stop the generation or validate a
multi-track model.

Suite scoring is completion-first: the number of required tracks completed
dominates fitness, completing every track adds a further suite bonus, and
normalized lap time ranks genomes once their completion coverage is equal.
NEAT's single-number early termination is disabled; only the declared validation
suite can produce a Validated Champion.

Original Track completion uses the actual legacy finish-line bitmap mask after
the car leaves the start area. Its approximate progress gates still contribute
fitness, but missing one cannot cause a visibly completed lap to run forever.

**Save Current Best** updates the current lineage record as a Draft instead of
adding another inventory car. A model becomes Validated only after its entire
declared validation suite passes. Stopping discards partial-generation
fitness and preserves the latest completed champion. Multiple training profiles
retain independent names, skins, modes, seeds, suites, populations, and results.
Pressing **Stop Training** immediately opens a save prompt for that completed
champion. Random Curriculum offers Save Draft and enables Save Validated only
after all held-out validation tracks pass.

Play and Train read the same paginated saved-model inventory. A saved genome can
be renamed, imported, exported, deleted, or continued from Train. Continuing
restores its generation, skin, and declared track suite; choosing **Train This
Car** after a campaign failure preselects that exact failed level as the suite.
Training saves share a lineage identifier. Save Current Best replaces that
lineage's Draft, and Save Validated Champion promotes the same inventory record,
so neither action creates duplicate cars. Restored genomes also reserve their
historical NEAT innovation range before mutation and crossover.

### Track Builder

The builder exposes Select, Straight, Corner, Start/Finish, Rotate, Delete,
Undo, Redo, Clear, Test, Save, Import, Export, and Saved Tracks controls.
Pieces snap to a 14×10 grid. Matching ports are green; open or mismatched ports
and affected cells are red. Incomplete tracks are allowed while editing, but
Save and Test remain disabled until validation passes.

A saveable track requires:

- Exactly one oriented Start/Finish piece.
- At least 12 pieces and four corners.
- Reciprocal ports, one connected component, and one non-branching closed loop.
- No duplicate, out-of-bounds, crossing, sub-loop, or mask-merging pieces.
- A straight approach and road-contained car drop area. Parallel lanes may sit
  in neighboring cells when their canonical road/collision masks stay separate;
  the builder reports that arrangement as a warning rather than rejecting it.

Test opens normal race preparation so the test model is explicitly selected;
testing never saves the editor snapshot. Invalid v1 tracks appear in Saved
Tracks as **Needs repair** and cannot enter racing or training until fixed.

## Canonical track system

Component tracks use 64×64 semantic tiles with a 44-pixel asphalt width and
center-aligned directional ports. Visual artwork, road masks, curb masks,
collision masks, finish masks, drop zones, and editor previews all derive from
the same geometry. Runtime play never crops arbitrary pieces from `track.png`.

Checked-in assets under `imgs/tiles/` are reproducible with:

```bash
python tools/build_track_assets.py
```

The generator returns `TrackDefinition` records rather than image files. It is
seeded, bounded by attempt and search-node limits, validates through the same
builder validator, and uses validated deterministic fallbacks. Stable campaign
content can be rebuilt with:

```bash
python tools/build_campaign_content.py
```

## Models, tracks, and compatibility

Local data remains under `~/.racing_car_x_neat` unless `RACING_DATA_DIR` is set.
The existing `models/`, `tracks/`, `imports/`, `exports/`, and `progress.json`
locations are preserved. Writes use atomic temporary-file replacement.

`.rcmodel` and `.rctrack` files are non-executable JSON. Schema-v1 models and
tracks load through backward-compatible in-memory migration; the next save
writes schema v2. Invalid imports are reported and left untouched.

The controller contract is intentionally unchanged:

- Controller version: `five-sensor-v1`
- Inputs: five normalized distances—left, right, forward, forward-left, and
  forward-right.
- Outputs: left+accelerate, right+accelerate, straight+accelerate, and coast.

Changing this contract requires a new explicit controller version and migration
strategy. Existing saved genomes are not silently reinterpreted.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers canonical seams and masks, structured topology diagnostics,
hundreds of generated seeds, stable campaign geometry, preparation/drop flow,
ordered lap completion, frozen timers, progress-aware multi-track fitness,
held-out validation, editor history, and v1/v2 persistence.
