import torch

from vlm_multilabel.losses.template_patch import patch_template_for_label_vector


class FakeTemplate:
    use_model = False

    def encode(self, inputs, **kwargs):
        return dict(inputs)

    def data_collator(self, batch, **kwargs):
        return {key: [item.get(key) for item in batch] for key in batch[0]}


def test_template_patch_is_idempotent_and_preserves_label_vector():
    template = FakeTemplate()
    patch_template_for_label_vector(template)
    patched_encode = template.encode
    patch_template_for_label_vector(template)
    assert template.encode is patched_encode

    record = {"input_ids": [1], "label_vector": [1, 0]}
    assert template.encode(record)["label_vector"] == [1, 0]
    batch = template.data_collator([record])
    assert torch.equal(batch["label_vector"], torch.tensor([[1, 0]], dtype=torch.float32))
