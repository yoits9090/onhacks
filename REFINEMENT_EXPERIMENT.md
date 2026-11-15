# User-Requested Refinement: V3

Declared after saving the accepted v2 checkpoint, before any archive-02
evaluation predictions. V2 remains deployed and immutable.

The user explicitly asked for another attempt after v2 was verified.
This extends the earlier two-attempt plan; it does not retroactively
change those experiments or reuse their evaluation labels for selection.

## Fixed Search And Reference

The reference is the accepted v2 checkpoint: SO400M=1, GOPT=2, DINOv3=0.5,
depth=5. It is NOT the weaker original equal-weight/top-10 baseline.

Use only the original 2,000 tuning sequences and their immutable retrievals.
The v2 winner was on two weight-grid boundaries, motivating this local grid:

- SO400M remains fixed at 1.
- GOPT weights: {1.5, 2, 3, 4}.
- DINOv3 weights: {0.25, 0.5, 0.75}; no tower is dropped.
- Depths: {3, 5, 7, 10}.
- Exactly 48 configurations; include the accepted checkpoint as control.
- Reuse already frozen tuning metrics for identical configurations.
- Eligible configurations cannot reduce tuning Acc@25km versus v2.
- Maximize tuning mean GeoScore; ties prefer v2, then enumeration order.
- Lock the choice before preparing the new evaluation manifest.
- Stop without a fresh test if v2 wins tuning.

All preprocessing, model weights, galleries, retrieval depth, fusion
normalization, cluster radii, seed selection and fourth-power spherical
consensus are unchanged. This is calibration, not encoder fine-tuning.

## Fresh Comparison

Use pinned OSV-5M archive 02.zip and test.csv. Exclude ALL sequences from
both previous manifests: 4,000 from v1 and 2,000 fresh sequences from v2.
Retain only the original globally hashed evaluation partition. Select one
image per sequence using the frozen image hash, rank with the v3 sample
salt, and take 4,000 sequences if available, otherwise 2,000. Below 2,000,
stop. Cohort size is determined by availability, never measured outcomes.

Cache all three towers once; compare only the locked candidate against
the saved v2 configuration. Report all standard metrics and the ordinary
paired 95% sequence-bootstrap interval.

For acceptance, require a positive GeoScore difference, no Acc@25km decline,
and the LOWER bound of a **98.3333% paired interval above zero**, using
Bonferroni alpha=0.05/3 across the three staged attempts. Use the 10,000
replicates and fixed seed in refinement_config.json. The earlier accepted
v2 report remains unchanged.

No fourth attempt is planned. Keep v2 if the stricter gate fails.
Write artifacts/experiment-v3/ separately; never overwrite v2 artifacts
or checkpoints/fusion-v2/. Stop computation no later than 16:00 EST;
provider billing remains a separate manual lifecycle action.
