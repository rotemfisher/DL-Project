# Digital to Analog Clock Converter - Project Guidebook

## 1. Project Overview
The goal of this project is to build a Deep Learning model that can "read" a digital clock and "draw" the corresponding analog clock. This is a Computer Vision task that involves translating one visual representation of time (digits) into another (geometric angles of hands), or potentially mapping the style of one clock to the time of another.

**Core Challenge:** We must train our own neural networks (using PyTorch) and are limited to a training set of 400 unique time combinations.

## 2. Work Log & Architectural Decisions

### Phase 1: Initialization & Data Strategy

#### Step 1: Project Scaffolding
**What we did:** Created a standard machine learning project structure:
- `src/`: For all source code (scripts, model definitions).
- `data/`: Split into `train` and `test` to keep our datasets organized.
- `models/`: A designated place to save trained model weights (`.pth` files).

**Why:** Keeping data separate from code and having a dedicated place for artifacts (models) prevents clutter and makes the project reproducible.

#### Step 2: Environment Setup
**What we did:** Created `requirements.txt` listing `torch`, `torchvision`, `pillow`, and `numpy`.
**Why:** 
- **PyTorch (`torch`, `torchvision`):** The mandatory framework for this project.
- **Pillow (`PIL`):** The Python Imaging Library, essential for programmatic image manipulation and drawing.
- **NumPy:** The standard for numerical operations, useful for array manipulations before feeding data to the network.

#### Step 3: Synthetic Data Generation
**What we did:** Implemented `src/data_generator.py`.
**Why:** 
The project requirements suggest starting "small" and mentions the difficulty of finding thousands of perfectly labeled clock pairs. Instead of searching the web, we built a **Generator**.

**Advantages of this approach:**
1.  **Infinite Data:** We can generate as many examples as we need on demand.
2.  **Perfect Labels:** Since we generate the image from a time value (H, M, S), we know *exactly* what time the image represents. There is no risk of mislabeling.
3.  **Control:** We can tweak the difficulty (e.g., change colors, fonts, hand thickness) by modifying the code.

**Technical Implementation Details:**
- **Digital Clock:** We use `ImageDraw.text` to render the time string. We calculate the bounding box to center the text perfectly.
- **Analog Clock:** We use **Trigonometry** to calculate the hand positions.
    - *Formula:* $x = center_x + length \times \cos(angle)$
    - Angles are adjusted because 0 degrees in math is usually "3 o'clock", but on a clock, 0 is "12 o'clock" (requires a -90 degree offset).
    - We implemented "continuous" movement for the hour and minute hands (e.g., at 10:30, the hour hand isn't exactly on 10, it's halfway between 10 and 11). This adds realism.

#### Step 4: Data Loading (PyTorch Dataset)
**What we did:** Created `src/dataset.py` with a custom `ClockDataset` class.
**Why:** 
PyTorch requires a standard way to iterate through data. This script:
1.  **Pairs Images:** It automatically finds matching pairs of digital and analog clocks based on the filename strings.
2.  **Pre-processing:** It converts images to RGB and applies standard transforms.
3.  **Normalization:** We normalize pixel values to the `[-1, 1]` range (using `0.5` mean/std), which is a common practice for Generative Adversarial Networks (GANs) and image-to-image translation models to ensure training stability.

#### Step 5: Model Architecture (U-Net Translator)
**What we did:** Implemented a U-Net architecture in `src/model.py`.
**Why:** 
The U-Net architecture is highly effective for image-to-image translation.
- **Encoder-Decoder:** The network first compresses the digital clock image into a low-dimensional representation (capturing "what" time it is) and then expands it back into a clock face.
- **Skip Connections:** This is the "secret sauce". By concatenating features from the encoder directly to the decoder, the model can preserve the sharp details of the original canvas, which is crucial for drawing precise, thin hands on the analog clock.
- **Tanh Activation:** The final layer uses `Tanh` to ensure the output pixels are in the same `[-1, 1]` range as our normalized training data.

#### Step 6: Workflow Migration (Jupyter Notebook)
**What we did:** Created `Project_Walkthrough.ipynb` and updated requirements to include `jupyter`, `matplotlib`, and `tqdm`.
**Why:** 
We shifted the training and visualization logic from a script (`train.py`) to an interactive Notebook.
- **Storytelling:** This allows us to present the project as a cohesive narrative, showing code alongside its visual output.
- **Immediate Feedback:** We can see the generated clocks evolve in real-time as the model trains.
- **Clean Architecture:** We kept the heavy lifting (Dataset, Model) in `src/` to ensure the notebook remains readable and focused on high-level logic.

### Next Steps
1.  **Run the Notebook:** Execute the cells in `Project_Walkthrough.ipynb` to train the model.
2.  **Analyze Results:** Check if the U-Net is successfully learning to draw the hands.
3.  **Iterate:** If the results are blurry or incorrect, we might need to adjust the loss function (e.g., add Perceptual Loss) or the model capacity.



