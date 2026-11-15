# Offgrid: Five-Minute Presenter Notes

Use these as cues, not a manuscript to memorize. The audience should remember:
removing GPS does not remove visual location clues.

## Before Starting

Open http://127.0.0.1:8765/, confirm Live inference, click Start over, and
turn on Present. Start on the first forest-road example with its estimate hidden.
The three example thumbnails remain visible in this version of Present mode.

The story follows the audience's decision: inspect a photo, see its recorded
estimate, choose a detail to hide, test that edit, and judge the result against
the system's failures. Do not invent a customer or a personal incident.

## 0:00-0:25 | Give Them A Question

Show the photo and the metadata line. Keep the map result hidden.

> This photo has no GPS data. How much do you think that hides?
>
> Just looking at the road and the trees, where would you put it?

Take one quick answer. Do not turn this into a guessing-game round.

> We built Offgrid to test what a location model can infer from the picture,
> and what happens when we change the picture.

## 0:25-1:15 | Show The First Answer

Click Reveal estimate, pause, then Reveal known location.

> This is a saved evaluation result. The model puts the photo about 6.6
> kilometres from where it was taken.
>
> There's no address here, but there is geographic information in the pixels.
>
> Now let's test a change you'd actually make before sharing it.

Let the audience look at the map. Do not read the latitude and longitude aloud.

## 1:15-2:35 | Let Someone Choose The Edit

> What would you cover?

Take one suggestion. If nobody answers promptly, use Cover, Cover lower center,
Apply cover. Click Analyze edited photo.

> This part runs live on our GPU. It analyzes the original again, then the
> edited picture with the same model settings, so we're comparing the two
> versions fairly.

Pause while it runs. Do not fill every second with explanation.

When both live results appear:

> The estimate moved [read the displayed distance].
>
> That tells us how much this edit changed this model's answer. It doesn't
> establish that the photo is private.

Read today's result. The rehearsed cover moved the estimate 1,136.3 km, but
never promise that result for another rectangle.

## 2:35-3:05 | Show Where It Fails

Select the second example, Under open skies. Reveal estimate and known location.

> This example misses by about 1,617 kilometres.
>
> These are the first three evaluation examples, including the misses. A
> convincing-looking pin on a map can still be very wrong.

## 3:05-4:10 | Explain The Work

Open The evidence.

> Underneath, three existing vision models retrieve similar street images.
> We calibrated how their results are combined into a location estimate.
>
> We tested the locked change on 2,000 fresh evaluation sequences. GeoScore,
> a distance-based geolocation metric, rose from about 3,963 to 4,082.
> The paired 95-percent interval for the gain stays above zero.
>
> Accuracy within 25 kilometres improved. Accuracy within one kilometre
> declined, so the improvement doesn't hold at every distance.
>
> This is a development-subset result. We didn't train the encoders or claim
> a new official benchmark.

Point to the evidence rather than reciting every table entry.

## 4:10-4:45 | Make The Product Concrete

Close the dialog and return to the photo.

> You can paste a photo, drop one onto the page, or choose a file. Then cover
> or crop something and test it again. Each revealed coordinate opens in
> Google Maps if you want to inspect the area.
>
> For your own uploads, we don't know the true location. We can show a
> prediction and how it changes, but we can't calculate its error.

Describe these options; do not start a second upload.

## 4:45-4:55 | Finish

> Offgrid builds on Chiikabu Labs' Chipoint and existing open research.
> Our work is the calibration, evaluation, and interactive experiment.
>
> Removing GPS does not remove visual location clues.

Stop. The final five seconds are buffer.

## If The Clock Or GPU Slips

- At 20 seconds waiting, use Continue to evidence. Say: "That run hasn't
  finished. I'll show you what we've measured instead." Never narrate a
  rehearsal result as today's live output.
- Past 2:35 without a completed comparison: cut further editing and use the
  second recorded example.
- Past 3:30 before opening evidence: drop the product-options section. Keep
  the measured gain, the one-kilometre decline, attribution, and the close.
- Past 4:30: stop adding material and close.

## How This Version Uses The Skills

- `storyboard`: the audience makes a decision and tests it; the interface
  controls support that story instead of becoming the story.
- `tech-talk-outline`: one takeaway, timed sections, optional material and
  explicit cuts. Its timing guidance is a planning aid, not a measured
  guarantee that this speaker will finish on time.
- `humanizer`: plain spoken sentences, fewer staged transitions, less
  numerical recital, and limitations stated where they affect interpretation.

Sources: `blader/humanizer`, `deanpeters/product-manager-skills`, and
`samber/developer-relations-skills`. These are community-authored instructions,
not evidence that a talk works. Time one complete spoken rehearsal before use.
