# SignSense Quick Reference

## Static Signs (MLP) - Single Poses

```bash
# 1. Record hand poses for letters
python -m ml.record_landmarks

# 2. Train model
python -m ml.train

# 3. Run app
python main.py  → Play Mode → Letters shown as they're detected
```

**Controls during recording:**
- Press `A-Z` to change letter label
- Perform the letter pose (hold consistently)
- Press `S` to save and exit
- Press `ESC` to exit without saving

**Result:** `ml/models/sign_mlp.pt` (MLP model for A-D, etc.)

---

## Dynamic Signs (LSTM) - Multi-Stage Movements

```bash
# 1. Record J sign with stage boundaries
python -m ml.dynamic_recorder

# 2. Train LSTM model
python -m ml.dynamic_train J

# 3. Record Z sign with stage boundaries
python -m ml.dynamic_recorder

# 4. Train LSTM model  
python -m ml.dynamic_train Z

# 5. Run app
python main.py  → Debug Mode → See stages progress as you perform J or Z
```

**Controls during recording:**
- Press `S` → Enter sign name (e.g., "J")
- Press `0-9` → Set stage number
- Press `SPACE` → Start/stop recording current stage
- Press `N` → Move to next stage
- Press `Q` → Save all stages and exit

**Result:** 
- `ml/models/dynamic_J.pt` (LSTM model for J)
- `ml/models/dynamic_Z.pt` (LSTM model for Z)

---

## File Structure

```
ml/
├── data/
│   ├── landmarks.npy         ← Static pose data (per letter)
│   └── dynamic/
│       ├── J/
│       │   ├── J_s0_0.npy   ← Stage 0, sequence 0
│       │   ├── J_s0_1.npy   ← Stage 0, sequence 1
│       │   ├── J_s1_0.npy   ← Stage 1, sequence 0
│       │   └── metadata.json
│       └── Z/
│           └── ...
├── models/
│   ├── sign_mlp.pt          ← Static sign model
│   ├── dynamic_J.pt         ← J dynamic model
│   └── dynamic_Z.pt         ← Z dynamic model
├── record_landmarks.py      ← Record static poses
├── dynamic_recorder.py      ← Record dynamic sequences
├── train.py                 ← Train static signs
├── dynamic_train.py         ← Train dynamic signs
├── model.py                 ← DynamicSignMLP (CNN) architecture
└── dynamic_model.py         ← DynamicSignLSTM architecture
```

---

## Model Architecture

### Static Signs (MLP)
```
Input (63 dims: 21 landmarks × 3 coords)
    ↓
Linear(63 → 256) + Batch Norm + ReLU + Dropout(0.3)
    ↓
Linear(256 → 128) + Batch Norm + ReLU + Dropout(0.2)
    ↓
Linear(128 → 64) + Batch Norm + ReLU
    ↓
Output (num_letters logits)
```

### Dynamic Signs (LSTM)
```
Sequence of frames (variable length)
    ↓
LSTM (63 → 128 hidden, 2 layers)
    ↓
Stage Head: Linear(128 → num_stages)    [predict current stage]
Transition Head: Linear(128 → 1)        [detect stage transition]
    ↓
Output (stage logits + transition confidence)
```

---

## Training Parameters

### Static Signs
- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** Until convergence (typically 50-100)
- **Batch size:** 32
- **Early stopping:** patience=15

### Dynamic Signs  
- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss (per frame)
- **Epochs:** Until convergence (typically 30-60)
- **Batch size:** 4
- **Train/Val split:** 80/20
- **Early stopping:** patience=15

---

## Common Commands

```bash
# Record static signs
python -m ml.record_landmarks

# Train static signs
python -m ml.train

# Record dynamic sign (e.g., J)
python -m ml.dynamic_recorder

# Train dynamic sign
python -m ml.dynamic_train J

# Test recognition
python main.py

# Debug mode (see all scores)
# (Launch from main menu)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named torch" | `pip install torch` |
| Low accuracy (<80%) | Record more training data |
| Model overfits | Add dropout, collect more varied data |
| Slow training | Use GPU or reduce model size |
| "Model not found" | Train model first |
| Dynamic signs don't work | Record sequences and train LSTM |

---

## Tips for Better Accuracy

### Static Signs
✓ Record poses from multiple angles  
✓ Vary distances from camera  
✓ Hold each pose consistently (~1 sec)  
✓ Try different lighting conditions  
✓ Use both left and right hands  

### Dynamic Signs
✓ Clearly define stage boundaries  
✓ Hold each stage for ~1 second  
✓ Perform movements at consistent speed  
✓ Record 5-10 complete sequences per sign  
✓ Be consistent with hand orientation  

---

**Need more help?** See [MODEL_GUIDE.md](MODEL_GUIDE.md) for detailed documentation.
