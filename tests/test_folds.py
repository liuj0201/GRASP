from da_cf_gop.folds import nested_loso_folds, training_artifact_hash


def test_nested_loso_is_speaker_disjoint():
    folds = nested_loso_folds("primary5", ["F03", "F04", "M01", "M04", "M05"], ["MC01", "MC02"])
    assert len(folds) == 5
    for fold in folds:
        assert fold.held_out_patient not in fold.training_patients
        assert len(fold.inner_folds) == 4
        for inner in fold.inner_folds:
            assert inner.held_out_patient not in inner.training_patients
            assert fold.held_out_patient not in inner.training_patients


def test_training_hash_does_not_have_a_test_input_channel():
    base = dict(stage="prior", config_sha256="a" * 64, checkpoint_sha256="b" * 64)
    first = training_artifact_hash(training_rows=[{"speaker": "F03", "gold": 0}], **base)
    second = training_artifact_hash(training_rows=[{"gold": 0, "speaker": "F03"}], **base)
    changed = training_artifact_hash(training_rows=[{"speaker": "F03", "gold": 1}], **base)
    assert first == second
    assert first != changed
