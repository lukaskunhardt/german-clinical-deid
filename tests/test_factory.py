"""Check factory.py against the original research code: same case sheets and same prompts, no API calls.

  python -I tests/test_factory.py <research repo> <raw records of one batch> <seed>

The raw records hold the case sheets the original code drew. Only sheets of the last revision (seed 21)
are expected to match; earlier batches of the corpus came from earlier revisions of the sheet.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import factory  # noqa: E402


def main(repo, raw_path, seed):
    sys.path.insert(0, os.path.join(repo, "fabrik"))
    import fabrik as old  # noqa: E402
    lists = factory.load_lists()
    system_old = open(os.path.join(repo, "fabrik/de/prompt_ohne.md"), encoding="utf-8").read() + "\n" + \
        open(os.path.join(repo, "fabrik/de/prompt_v2.md"), encoding="utf-8").read()
    assert system_old == open(os.path.join(factory.DATA, "system_prompt.md"), encoding="utf-8").read(), "system prompt differs"
    n = same_case = same_prompt = 0
    for line in open(raw_path, encoding="utf-8"):
        shipped = json.loads(line)["case"]
        i = int(shipped["doc_id"].split("-")[2])
        new = factory.make_case(i, seed, *lists)
        for k in ("fallkern", "fallkern_atc", "fallkern_meta"):  # the clinical core comes from an LLM call; reuse the shipped one
            if k in shipped:
                new[k] = shipped[k]
        n += 1
        same_case += json.dumps(new, ensure_ascii=False) == json.dumps(shipped, ensure_ascii=False)  # key order matters for the prompt
        same_prompt += factory.user_prompt(new) == old.user_prompt(shipped)
    print(f"{n} sheets: {same_case} identical case sheets, {same_prompt} identical user prompts; system prompt identical")
    sys.exit(0 if n == same_case == same_prompt else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
