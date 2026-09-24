from da_cf_gop.dual_audit import PROVENANCE, closure_audit
from da_cf_gop.phonology import PhnSegment


def segment(i, start, end, label):
    return PhnSegment(i, start, end, label, None, None)


def test_only_contiguous_matching_release_can_pair():
    segments = [segment(0, 0, 20, "tcl"), segment(1, 20, 30, "t"),
                segment(2, 30, 40, "pcl"), segment(3, 40, 50, "b"),
                segment(4, 50, 60, "kcl"), segment(5, 61, 70, "k"),
                segment(6, 70, 80, "dcl")]
    result = closure_audit(segments)
    assert [row["contiguous_matching_release"] for row in result] == [True, False, False, False]
    assert [row["raw_index"] for row in result] == [0, 2, 4, 6]


def test_closure_is_not_deleted_or_changed_by_audit():
    segments = [segment(0, 0, 20, "vcl"), segment(1, 20, 30, "v")]
    original = list(segments)
    result = closure_audit(segments)
    assert segments == original
    assert result[0]["status"] == "unpaired_closure_requires_review"


def test_provenance_distinguishes_corpus_manual_evidence_from_per_file_review():
    assert PROVENANCE["status"] == "corpus_level_manual_error_annotation_supported"
    assert PROVENANCE["per_file_independent_annotation_audit"] == "not_available"
    assert any("rudzicz_nips10" in source["url"] for source in PROVENANCE["sources"])
