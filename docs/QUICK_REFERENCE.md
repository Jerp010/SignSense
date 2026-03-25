# SignSense Quick Reference

## Essential Commands

```bash
# Activate virtual environment
cd signsense && .venv\Scripts\activate

# Run application
python -m signsense.main

# Record static signs
python -m signsense.ml.record_landmarks

# Record dynamic signs
python -m signsense.ml.dynamic_recorder

# Train static model
python -m signsense.ml.train

# Train dynamic model
python -m signsense.ml.dynamic_train <SIGN>
```

## Recording Controls

### Static Signs
| Key | Action |
|-----|--------|
| `T` | Type custom label |
| `A-Z` | Select letter |
| `SPACE` | Start/stop recording |
| `[` | Save and exit |

### Dynamic Signs
| Key | Action |
|-----|--------|
| `T` | Type gesture name |
| `A-Z` | Select letter |
| `S` | Toggle simple/complex |
| `0-9` | Set stage |
| `SPACE` | Start/stop |
| `P` | Previous stage |
| `N` | Next stage |
| `Q` | Save and exit |

## File Structure

```
signsense/
├── config/
│   ├── dynamic_signs.csv    # Dynamic signs registry
│   └── dynamic_signs.yaml  # Sign definitions
├── ml/
│   ├── data/
│   │   ├── landmarks.csv   # Static pose data
│   │   └── dynamic/        # Dynamic sequence data
│   └── models/
│       ├── sign_mlp.pt     # Static model
│       └── dynamic_*.pt    # Dynamic models
└── signs/
    └── sign_registry.py    # Main sign registry
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named X" | Activate virtual environment |
| Low accuracy | Record more data |
| "Model not found" | Train model first |
| Dynamic signs don't work | Record complete sequences |
