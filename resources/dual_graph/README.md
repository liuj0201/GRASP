# External dictionary resource

`cmudict.dict` and `CMUDICT_LICENSE` are unmodified public files downloaded on
2026-09-05 from the [CMUdict repository](https://github.com/cmusphinx/cmudict),
`master/cmudict.dict` and `master/LICENSE`. CMUdict is maintained by Carnegie
Mellon University and redistributed here under its included license. No hashing
or mutable on-demand dictionary download is performed by the experiment.

The frozen acceptable-pronunciation policy is `configs/dual_acceptable_v1.json`.
Source stress markers and variants are retained in each generated word block;
the acoustic model uses the project's 39-phone ARPABET inventory, without stress.
Only dictionary-attested same-word AH0/IH0 alternatives, at a single fixed
within-word position, are added. This is not a blanket weak-vowel rule.

This conservative first version does not infer lexical meaning or part of speech.
Known homographs and words with unsupported length/context/segment variants are
marked non-diagnostic. Their unsupported variants are reported, not silently
accepted or automatically labelled patient errors. It therefore does not yet
implement the plan's full length-changing normal-variant blocks.
