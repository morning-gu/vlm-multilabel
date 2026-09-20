import pytest

from vlm_multilabel.constants.base import load_config
from vlm_multilabel.data.conversion import (
    build_record,
    choose_primary_label,
    first_positive_code,
    validate_first_positive_records,
)


@pytest.mark.parametrize(
    "dataset,num_classes", [("voc", 20), ("chestxray14", 14), ("celeba", 40)]
)
def test_load_config_derivates_num_classes(dataset, num_classes):
    config = load_config(dataset)
    assert config.num_classes == num_classes
    assert len(config.label_codes) == num_classes
    assert config.render_prompt()


def test_parse_output_returns_none_for_unknown_label():
    config = load_config("voc")
    assert config.parse_output("The object is unknown") == (None, "")


def test_parse_output_extracts_code_and_explanation():
    config = load_config("voc")
    code, explanation = config.parse_output(
        "<think>reasoning</think><explanation>a person</explanation> ps"
    )
    assert code == "ps"
    assert explanation == "a person"


def test_build_record_selects_first_positive_in_config_order():
    config = load_config("voc")
    record = build_record(
        config,
        "image.jpg",
        ["cat", "person"],
        config.render_prompt(),
        config.user_prompt,
    )
    assert record["primary_code"] == "ps"
    assert record["label_vector"][0] == 1
    assert record["label_vector"][2] == 1
    assert sum(record["label_vector"]) == 2


def test_choose_primary_label_is_first_and_rejects_empty():
    assert choose_primary_label(["person", "cat"]) == "person"
    with pytest.raises(ValueError):
        choose_primary_label([])


def test_first_positive_code_uses_config_order():
    config = load_config("voc")
    label_vector = [0] * config.num_classes
    label_vector[2] = 1
    label_vector[0] = 1
    assert first_positive_code(config, label_vector) == "ps"


def test_validate_first_positive_records_accepts_and_rejects():
    config = load_config("voc")
    label_vector = [0] * config.num_classes
    label_vector[0] = 1
    valid = {"label_vector": label_vector, "primary_code": "ps"}
    assert validate_first_positive_records(config, [valid]) == 1

    invalid = {"label_vector": label_vector, "primary_code": "ct"}
    with pytest.raises(ValueError, match="first positive is 'ps'"):
        validate_first_positive_records(config, [invalid])


@pytest.mark.parametrize("dataset", ["voc", "chestxray14", "celeba"])
def test_prompts_ask_for_one_correct_category(dataset):
    config = load_config(dataset)
    rendered = config.render_prompt().lower()
    user_prompt = config.user_prompt.lower()
    assert "dominant" not in rendered
    assert "dominant" not in user_prompt
    assert "one correct" in rendered
    assert "one correct" in user_prompt
