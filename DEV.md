

### Adding a new model class

In this section, you shall find steps on how to add a new model (e.g. method of self-supervised video representation learning) to evaluation (fine-tune) on the EPIC-KITCHENS dataset.

**1. Add a config**: See example `configs/EPIC-KITCHENS/GDT/base.yaml`

**2. Add relevant keys in config (optional)**: For instance, I need `cfg.MODEL.CKPT` to be present to load model weights for GDT. Thus, I need that key and I added the following snippet to `slowfast/config/defaults.py`:
```python
# Model checkpoint path (optional)
_C.MODEL.CKPT = None
```

**3. Add the model code**: Inside `slowfast/models/gdt/*.py`, add your model code and add relevant tests.
