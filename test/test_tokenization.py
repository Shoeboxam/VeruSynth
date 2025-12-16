import pytest
from transformers import AutoTokenizer

from fine_tune.train import tokenize_with_labels  # adjust import
from fine_tune.train import BASE_MODEL            # or inline model name


@pytest.mark.parametrize("cutoff_len", [128])
def test_tokenize_with_labels_masks_prompt_correctly(cutoff_len):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = "INSTRUCTIONS:\n\nHere is some code."
    target_json = '{\n  "annotations": []\n}'
    full_text = prompt + "\n\n" + target_json

    example = {"text": full_text}
    out = tokenize_with_labels(example, tokenizer=tokenizer, cutoff_len=cutoff_len)

    input_ids = out["input_ids"]
    labels = out["labels"]
    attention_mask = out["attention_mask"]

    # Decode once for sanity
    decoded = tokenizer.decode([tid for tid, m in zip(input_ids, attention_mask) if m == 1])

    # Verify that decoded text ends with our JSON (no truncation here)
    assert decoded.strip().endswith(target_json.strip())

    # Compute the prompt length in tokens the same way as the function
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
        return_tensors=None,
    )["input_ids"]
    prompt_len = sum(1 for t in prompt_tokens if t != tokenizer.pad_token_id)

    # 1) Labels before prompt_len should be -100
    for i in range(prompt_len):
        if attention_mask[i] == 1:  # ignore any leading pads, just in case
            assert labels[i] == -100

    # 2) For the target region, labels should match input_ids (where mask == 1)
    for i in range(prompt_len, len(input_ids)):
        if attention_mask[i] == 1:
            assert labels[i] == input_ids[i] or labels[i] == -100  # last pad area
