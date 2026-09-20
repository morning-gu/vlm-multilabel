import json

import pytest

from vlm_multilabel.constants.base import load_config
from vlm_multilabel.commands.convert_chestxray14 import main, to_record


@pytest.fixture()
def config():
    return load_config("chestxray14")


def _render(config):
    return config.render_prompt()


def test_normalize_pleural_thickening(config):
    record = to_record(
        config,
        "image.png",
        "Atelectasis|Pleural_Thickening",
        _render(config),
    )
    assert record is not None
    assert record["primary_code"] == "AT"
    assert record["label_vector"][0] == 1
    assert record["label_vector"][12] == 1
    assert record["label_vector"][13] == 0


def test_no_positive_is_skipped_by_default(config):
    assert to_record(config, "image.png", "No Finding", _render(config)) is None


def test_official_test_can_retain_no_positive(config):
    record = to_record(
        config,
        "image.png",
        "No Finding",
        _render(config),
        allow_no_positive=True,
    )
    assert record is not None
    assert record["primary_code"] is None
    assert record["label_vector"] == [0] * config.num_classes
    assert [message["role"] for message in record["messages"]] == ["system", "user"]


def test_unknown_label_raises(config):
    with pytest.raises(ValueError, match="Unknown ChestX-ray14"):
        to_record(config, "image.png", "Not A Disease", _render(config))


def test_official_test_split_is_complete_and_train_is_positive_only(
    tmp_path, monkeypatch, capsys, config
):
    images = tmp_path / "images"
    images.mkdir()
    rows = [
        ("train01.png", "Atelectasis"),
        ("train00.png", "No Finding"),
        ("test01.png", "Pleural_Thickening"),
        ("test00.png", "No Finding"),
        ("test02.png", "Hernia"),
    ]
    for name, _ in rows:
        (images / name).write_bytes(b"png")

    csv_path = tmp_path / "Data_Entry_2017.csv"
    with csv_path.open("w", encoding="utf-8") as stream:
        stream.write("Image Index,Finding Labels\n")
        for name, labels in rows:
            stream.write(f"{name},{labels}\n")
    (tmp_path / "train_val_list.txt").write_text(
        "train01.png\ntrain00.png\n", encoding="utf-8"
    )
    (tmp_path / "test_list.txt").write_text(
        "test01.png\ntest00.png\ntest02.png\n", encoding="utf-8"
    )

    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert_chestxray14_to_swift",
            "--csv", str(csv_path),
            "--image-root", str(images),
            "--split-dir", str(tmp_path),
            "--out", str(out),
        ],
    )
    main()

    train = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    test = [json.loads(line) for line in (out / "test.jsonl").read_text().splitlines()]
    assert len(train) == 1 and train[0]["images"][0].endswith("train01.png")
    assert not (out / "val.jsonl").exists()
    assert len(test) == 3
    zero = next(record for record in test if record["primary_code"] is None)
    assert zero["label_vector"] == [0] * config.num_classes
    assert sum(record["label_vector"][12] == 1 for record in test) == 1
    assert sum(record["label_vector"][13] == 1 for record in test) == 1
    assert "test=3" in capsys.readouterr().out
