# Design review

Phase 4 asks for a review of the finished UI against the `apple-design` skill.
That skill is not installed in this environment, so the rule list in Section 7 of
the build plan was applied by hand.

Method: every state was rendered to an image with the real Windows font and
palette, in light and dark appearance, at two window sizes. The images were read
and checked against the rules. The images are not committed, because they are
easy to regenerate and add nothing to the repo.

## Findings

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | Critical | Advanced settings overlapped. Opening the section on a 760 px window squeezed the wrapped help labels until they drew on top of each other and hid the "Detector" heading. | The whole page moved into one scroll area, so no widget is squeezed below the height it needs. The progress list lost its own scroll area, because a scrolling region inside a scrolling region traps the wheel. |
| 2 | Critical | The drop zone had no visible edge in dark appearance. The border took its colour from the palette mid role, which is nearly the window colour on a dark theme. The dashed edge is the only thing that says the region takes a drop. | The border now comes from the window text colour at 130 alpha, so it holds in both appearances. |
| 3 | High | A waiting progress bar drew as a hairline and read as a divider rule, not as a bar at zero. | Both bar kinds now carry a drawn groove, from the text colour at 110 alpha, and a filled chunk in the accent colour. |
| 4 | High | The advanced settings toggle painted itself in the platform highlight when open. That put a second accent on screen, against the one primary action rule. | The toggle is flat in every state and underlines on hover. |
| 5 | Medium | The frame step control read "1 frames". | The suffix went away. The label reads "Detect on every Nth frame" and the control shows a number. |
| 6 | Medium | The focus ring on the primary button was a border that only existed while focused, so the label shifted when focus arrived. | The border is always present and transparent. Focus changes its colour only. |
| 7 | Medium | After a stopped run the overall line read "4 of 4 done" and the bar sat at full, because it counted every row that came back rather than every row that produced a copy. | The count excludes stopped rows. The estimate is replaced by a final count when the run ends, so it cannot go stale. |
| 8 | Medium | A second run over the same output folder reported "0 videos done. 0 failed." while every row read Already done. That reads as a total failure. | The summary adds a sentence naming how many were already done. The reference string from the plan did not change. |

Items 7 and 8 were found by driving the real window through a real batch, not by
reading images.

## Rules checked, and where each one is met

| Rule | Where |
|---|---|
| Clickable controls at least 24 px, primary 32 px or more | `CONTROL_HEIGHT` is 28, `PRIMARY_HEIGHT` is 36 |
| Body text at 4.5:1 in both palettes | All text uses the palette text colour. Help text is one point smaller, never fainter |
| No information carried by colour alone | Every status draws a mark and a word. `tests/test_strings.py` checks that every status word has a mark |
| Sentence case everywhere | `tests/test_strings.py::test_sentence_case` |
| One primary action | Only "Blur faces" carries the accent. Finding 4 fixed the second one |
| Empty, loading, error and success states designed | Images 01, 04, 05 and the failure row |
| Destructive actions confirm first | Stop asks. Closing while running asks. Replace existing files is an unticked box |
| The window resizes | The drop zone and the progress list carry the stretch. Buttons are fixed |
| Keyboard reachable with a visible focus ring | Tab order is the layout order. The primary button keeps a focus border. Every icon only control has an accessible name |
| Spacing on an 8 px grid | `GRID` 8, `MARGIN` 16, `SECTION` 24 |

## Open items

None at Critical or High.

One item stays open at Low. The help text under each radio button starts a few
pixels to the left of the button label, because the help label sits in the
column rather than under the label. It reads correctly and groups correctly.
