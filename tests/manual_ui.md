# Manual UI script

Phase 4 verify. Work down the list on a real desktop and tick every line.

Start the window:

```
.venv\Scripts\python.exe -m ui.app
```

Some of these checks also run automatically in `tests/test_ui.py`. That file
drives the real window through real worker processes. The lines below marked
`(also automated)` are covered there. Run them by hand anyway, because the
automated version never draws the window on a screen.

## Choosing the input

- [x] Drop a folder of videos on the drop zone. The count and total size appear,
      for example "12 videos, 8.4 GB". (also automated)
- [x] Drop three video files at once. The count says 3 videos.
- [x] Drop a folder that holds no videos. A message says the folder holds no
      videos and says what to do.
- [x] Press Ctrl+O. The file dialog opens.
- [x] Press Ctrl+Shift+O. The folder dialog opens.
- [x] Drag a folder over the drop zone. The dashed border changes while the
      pointer is over it.

## Choosing the output folder

- [x] Choose an output folder. The path appears in the field.
- [x] The app suggests a folder next to the source the first time.
- [x] Start turns on once a video and an output folder both exist. (also automated)
- [x] Before then, Start is off and the line beside it says what is missing.
      (also automated)

## Running

- [x] Press Start. Rows move from Waiting to Detecting to Writing to Checking
      to Done. (also automated)
- [x] Drag the window while the batch runs. The window keeps drawing.
- [x] The overall bar and the count move as files finish. (also automated)
- [x] The time estimate says "Estimating time left." until the first file
      finishes, then gives a time.
- [x] The Start button reads Stop while the batch runs. (also automated)
- [x] Choose files, Choose a folder and the advanced controls are off while the
      batch runs.

## The check and the gate

- [ ] Open Advanced settings. "Check each copy and hold back any that still
      shows something" is ticked when the window is opened for the first time.
- [ ] Run a batch with it ticked. Rows show Checking after Writing, and the
      run takes about twice as long. (also automated)
- [ ] The summary says the check found nothing, on footage that has nothing
      left in it. (also automated)
- [ ] Run a batch of the real sample footage with it ticked. Copies are held
      back, the summary names the kinds, and the files are in a quarantine
      folder beside the output folder with their records.
- [ ] Untick it and run again. No row reads Checking and the summary says
      nothing about the check. (also automated)
- [ ] Untick it, close the window, open it again. It is still unticked.
      (also automated)

## Stopping

- [x] Press Stop. A dialog asks first. (also automated for the stop itself)
- [x] Choose Keep going. The batch carries on.
- [x] Press Stop again and confirm. Remaining rows read Stopped. (also automated)
- [x] Look in the output folder. No half written file is there. (also automated)
- [x] Press Escape while the batch runs. The same dialog appears.
- [x] Close the window while the batch runs. A dialog asks first.

## Finishing

- [x] The summary line reads "N videos done. N failed." (also automated)
- [x] Press Open output folder. The folder opens in Explorer.
- [x] Run the same batch again. Every row reads Already done, and the summary
      says how many were already done. (also automated)
- [x] Tick Replace existing files, then run again. The rows process again.

## Failures

- [x] Put a file named `broken.mp4` holding text in the source folder. Its row
      reads Failed and the batch continues to the next file.
- [x] Rest the pointer on the Failed row. The tooltip gives the reason.

## Appearance and keyboard

- [x] Switch Windows to light appearance. Text and the dashed border stay
      readable.
- [x] Switch Windows to dark appearance. Text and the dashed border stay
      readable.
- [x] Press Tab through every control. Each one shows a focus ring.
- [x] Open Advanced settings. Every control has one line of help under it.
- [x] Make the window narrow and short. Nothing overlaps. The page scrolls.
- [x] Make the window large. The drop zone and the progress list grow. The
      buttons stay their size.
- [x] Read every label. Every one is sentence case.

## Design review

The `apple-design` skill is not installed in this environment, so its rule list
from Section 7 was applied by hand against rendered images of every state. See
`docs/design_review.md`. Zero Critical and zero High items are open.
