# Manual Inpaint Selection Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a new manual rectangle replace the pending inpaint preview while preserving undo/redo for that replacement.

**Architecture:** Keep pending-selection state owned by `DrawingPanel`. Add a draw-stack command that snapshots the old and new pending states and asks the panel to restore either one; actual image pixels continue to use `InpaintUndoCommand`.

**Tech Stack:** Python, Qt/PySide through qtpy, NumPy, unittest

## Global Constraints

- Replacing a pending preview must not alter previously applied inpaint pixels.
- Undo restores the previous preview; redo restores the newer preview.
- Applying inpaint clears the preview and remains independently undoable.
- Automatic rectangle mode and right-button restore behavior remain unchanged.

---

### Task 1: Undoable pending rectangle replacement

**Files:**
- Modify: `tests/test_paint_mode.py`
- Modify: `ui/drawing_commands.py`
- Modify: `ui/drawingpanel.py`

**Interfaces:**
- Consumes: `Canvas.push_undo_command(command)`, `DrawingPanel.rect_inpaint_dict`, `DrawingPanel.inpaint_mask_item`
- Produces: `RectInpaintSelectionCommand(panel, new_state)`, `DrawingPanel.captureRectInpaintSelection()`, `DrawingPanel.restoreRectInpaintSelection(state)`

- [ ] **Step 1: Write failing interaction tests**

Add tests using a real `Canvas` and `DrawingPanel` with a deterministic rectangle-mask function. Verify the first manual selection leaves `ImageEditMode.RectTool` active, the second selection changes the visible rectangle, undo restores the first selection and preview position, and redo restores the second.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_paint_mode.PaintModeTest.test_manual_rectangle_selection_replaces_preview_with_undo`

Expected: failure because manual selection currently leaves `ImageEditMode.NONE` and no selection replacement command exists.

- [ ] **Step 3: Implement snapshot restoration and the undo command**

Add a command that captures the current pending selection and swaps it with a copied new state:

```python
class RectInpaintSelectionCommand(QUndoCommand):
    def __init__(self, panel, new_state):
        super().__init__()
        self.panel = panel
        self.old_state = panel.captureRectInpaintSelection()
        self.new_state = new_state

    def redo(self):
        self.panel.restoreRectInpaintSelection(self.new_state)

    def undo(self):
        self.panel.restoreRectInpaintSelection(self.old_state)
```

`DrawingPanel` must deep-copy NumPy-backed payloads, copy the `QPixmap`, restore the preview parent and position, and restore rectangle mode. The first pending selection can be installed directly; subsequent selections are pushed through the command.

- [ ] **Step 4: Verify GREEN and image undo behavior**

Run the focused manual-selection tests, then the existing `InpaintUndoCommand` tests. Confirm applying a pending selection clears its preview and undo restores the image and mask pixels.

- [ ] **Step 5: Run full verification**

Run: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -p 'test_*.py'`

Expected: all tests pass with zero failures.
