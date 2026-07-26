# Manual Inpaint Selection Replacement Design

## Goal

When rectangle inpainting is in manual mode, drawing a new valid rectangle replaces the pending selection preview. The previous preview is not applied to the image. Undo restores the previous pending preview, and redo restores the newer preview.

## Behavior

- Completing the first valid manual rectangle displays its mask preview and immediately leaves the rectangle tool ready for another drag.
- Completing another valid manual rectangle replaces the existing preview with the new rectangle and mask.
- Undo after replacement restores the previous rectangle, raw mask, processed mask, preview pixmap, and preview position.
- Redo restores the newer selection.
- An invalid rectangle keeps the existing preview unchanged.
- Pressing Inpaint applies only the currently visible selection through the existing image undo command, then removes the preview.
- Previously applied inpaint results are not removed when a pending selection is replaced.
- Automatic rectangle inpainting and right-button restore behavior remain unchanged.

## Implementation

Represent the pending manual rectangle as a snapshot containing its inpaint payload, raw mask, processed preview pixmap, and position. A dedicated draw-stack command swaps between the old and new snapshots. `DrawingPanel` owns snapshot capture and restoration because it owns the pending selection UI.

After a manual selection command is pushed, rectangle mode is restored so the next drag can begin without pressing Delete or reselecting the tool. The existing `InpaintUndoCommand` continues to own pixel and mask changes after the user presses Inpaint.

## Testing

- A real `DrawingPanel` and `Canvas` test verifies that a second manual rectangle replaces the preview and leaves rectangle mode active.
- Undo and redo are verified against the observable pending rectangle, mask, preview position, and scene visibility.
- A test verifies that applying the selected region still clears the preview and that the existing pixel operation remains undoable.
