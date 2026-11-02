from signrec import config
from signrec.classes import GROUP_LABELS, class_names_vi, load_classes, num_classes


def test_82_classes_with_contiguous_ids():
    classes = load_classes()
    assert num_classes() == 82
    assert [c["id"] for c in classes] == list(range(82))


def test_every_class_has_names_and_a_known_group():
    for c in load_classes():
        assert c["name_vi"].strip() and c["name_en"].strip()
        assert c["group"] in GROUP_LABELS
    assert len(class_names_vi()) == 82


def test_every_class_has_a_sample_image_for_the_webapp():
    missing = [c["id"] for c in load_classes() if not (config.SAMPLES_DIR / f"{c['id']}.jpg").exists()]
    assert missing == []
