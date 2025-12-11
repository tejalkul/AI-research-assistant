from app import answer_question

tests = [
    ("Which paper introduces method Z, and what datasets were used?", ["Z", "dataset"]),
    ("List the limitations of approach A stated in the conclusion.", ["limitation", "conclusion"]),
    ("How does model B reduce compute compared to baseline?", ["compute", "baseline"])
]

def contains_any(text, keywords):
    t = text.lower()
    return any(k.lower() in t for k in keywords)

if __name__ == "__main__":
    for q, kws in tests:
        out = answer_question(q)
        ans = out["answer"]
        ok = contains_any(ans, kws)
        print("\nQ:", q)
        print("OK?" , ok)
        print("Ans:", ans[:400], "...")
        print("Sources:", out["sources"])
