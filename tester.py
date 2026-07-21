"""
test_cli.py

Interactive command-line tester for tag_language().
Drop this file into your pinoybot/ folder (next to pinoybot.py) and run:

    python test_cli.py

Type a sentence, hit Enter, see the tags. Type 'quit' or 'exit' to stop.
"""

import re
from pinoybot import tag_language

# Very basic tokenizer: splits off punctuation as its own token,
# keeps hyphenated words (e.g. "nag-march") and words together.
# This is NOT the same tokenizer used to build the dataset -- it's just
# good enough for quick manual testing from the command line.
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*|[^\sA-Za-z0-9]")


def tokenize(sentence: str):
    return TOKEN_PATTERN.findall(sentence)


def main():
    print("PinoyBot interactive tester. Type a sentence and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            sentence = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not sentence:
            continue
        if sentence.lower() in ("quit", "exit"):
            print("Bye!")
            break

        tokens = tokenize(sentence)
        if not tokens:
            print("(no tokens found)")
            continue

        tags = tag_language(tokens)

        if len(tags) != len(tokens):
            print(f"WARNING: got {len(tags)} tags for {len(tokens)} tokens -- "
                  f"tag_language() should return one tag per token.")

        for tok, tag in zip(tokens, tags):
            print(f"  {tok!r:20s} -> {tag}")
        print()


if __name__ == "__main__":
    main()