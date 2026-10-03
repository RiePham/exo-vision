# Exo Vision

A webcam looks at a table, finds cups, bottles and phones, and tells you where they are.

This is the first step toward a robot arm that can pick things up for a person.

## Why

Before a robot arm can grab something, it needs to know two things:

1. **What** the thing is.
2. **Where** the thing is.

This project gives the arm eyes for both.

## What it does

1. **It looks.** A webcam sits on the table and watches what is in front of it.
2. **It names things.** It finds cups, bottles and phones and draws a box around each one.
3. **It finds where they are.** The webcam is the starting point, (0, 0). For each object, it works out:
   - **Y**: how far ahead of the webcam it is
   - **X**: how far left (-) or right (+) of the webcam it is
4. **It asks.** For example: *"Grab the cup at X = +4 cm, Y = 38 cm? yes / no"*
5. **You answer.**
   - **Yes:** it locks on to that object, keeps updating its position and says how to hold it. A cup or bottle needs a full-hand grip. A phone needs a finger pinch.
   - **No:** it skips that object and asks about the next one.

The window shows two things side by side: the camera view, and a map of the table seen from above.

```
              Y (forward, cm)
                 ^
                 |        • bottle
         • cup   |
                 |
  -X  -----------+-----------  +X
             webcam (0, 0)
```

## How it knows what the thing is

It uses **YOLOv8**, a ready-made program that has already learned from more than 100,000 photos what cups, bottles and phones look like. Because it saw them big and small, near and far, it can still spot them when they move away.

## How it knows how far away the thing is

A webcam takes flat pictures, so it cannot measure distance by itself. We use a simple rule:

> **Things look smaller when they are farther away.**

We know a cup is about 9.5 cm tall. If the cup looks big on the screen, it is close. If it looks small, it is far. A bit of math turns "how big it looks" into centimetres.

| How tall the cup looks on screen | How far away it is |
|---|---|
| 175 pixels | about 30 cm |
| 105 pixels | about 50 cm |
| 52 pixels  | about 100 cm |

### Calibration (one time only)

Every webcam zooms a little differently. To set it up for your webcam:

1. Put **one** cup exactly **30 cm** in front of the webcam.
2. Press **c**.
3. The program prints a number, for example `--focal 553`. Use it the next time you run the program, so you don't need to set it up again.

## Try it

You need Python 3.10 or newer and a webcam.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python grasp_assist.py
```

The first run downloads the YOLOv8 model file (about 6 MB) by itself.

| Key | What it does |
|---|---|
| y | Yes, grab this object |
| n | No, skip it |
| r | Start over |
| c | Calibrate (see above) |
| q | Quit |

If you already calibrated: `python grasp_assist.py --focal 553`

## Files

| File | What it is |
|---|---|
| `grasp_assist.py` | The main program: finds objects, works out where they are, asks the user |
| `detect_grasp.py` | The first version: finds objects and suggests a grip, no position yet |
| `requirements.txt` | The Python packages to install |

## What works now, and what doesn't yet

**Works now**
- Finds and names cups, bottles and phones in real time.
- Gives each object a position on the table, with the webcam at (0, 0).
- Asks the user and remembers the answer.
- Suggests a grip for each object.

**Not yet**
- The distance is a guess based on the usual size of each object. A cup much bigger than 9.5 cm will look closer than it really is.
- A phone lying flat is harder to measure.
- With one webcam there is no true depth, only an estimate.

## What comes next

1. **Better positions.** Put 4 small printed markers on the table so the program can measure the table directly, instead of guessing from object size.
2. **Talk to the robot arm.** Turn the webcam's (X, Y) into a position the robot arm understands.
3. **Grab for real.** Send the position and grip to the arm when the user says yes. The spot for this is ready in the code: `on_grab()` in `grasp_assist.py`.
4. **Test with people.** Try it with real users at a real table.

Videos:



https://github.com/user-attachments/assets/9dbfff46-677e-48c9-8f90-c7bc2a33ed4b





