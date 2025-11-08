# Test: render_val_accuracy.svg

**Goal:** ensure the SVG chart renders.

### Steps
1) Verify your Gist CSV has at least a few rows with a `val_accuracy` value (can be blank for some rows).
2) Push a commit to run CI.
3) In the run Summary, look for an embedded image named `val_accuracy`.

### Expected
- `val_accuracy.svg` is attached in the Summary tab and updates over time.
