import random
import re


def split_into_words(text: str) -> list[str]:
  # Split on words and keep punctuation
  tokens = re.findall(r"\w+|[.,!?]", text.lower())
  return tokens


class WordCodebook:

  def __init__(self):
    self.id_to_word = []
    self.word_to_id = {}
    self.vocab_size = 0

  def build_from_words(self, words: list[str]):
    for w in words:
      if w not in self.word_to_id:
        self.word_to_id[w] = len(self.id_to_word)
        self.id_to_word.append(w)
    self.vocab_size = len(self.id_to_word)

  def encode(self, words: list[str]) -> list[int]:
    return [self.word_to_id[w] for w in words if w in self.word_to_id]

  def decode(self, ids: list[int]) -> str:
    words = [self.id_to_word[i] for i in ids]
    text = ""
    for w in words:
      if w in [".", ",", "!", "?"]:
        text += w
      else:
        text += (" " if text and text[-1] != " " else "") + w
    return text


class TrigramStoryGenerator:

  def __init__(self, vocab_size: int):
    self.vocab_size = vocab_size
    self.bigram_counts = {}
    self.trigram_counts = {}

  def train(self, tokens: list[int]):
    for i in range(len(tokens) - 1):
      w1 = tokens[i]
      w2 = tokens[i + 1]
      self.bigram_counts.setdefault(w1, {})
      self.bigram_counts[w1][w2] = self.bigram_counts[w1].get(w2, 0) + 1

      if i + 2 < len(tokens):
        w3 = tokens[i + 2]
        key = (w1, w2)
        self.trigram_counts.setdefault(key, {})
        self.trigram_counts[key][w3] = self.trigram_counts[key].get(w3, 0) + 1

  def sample_next_bigram(self, w: int, rng: random.Random) -> int:
    counts = self.bigram_counts.get(w, {})
    if not counts:
      return rng.randint(0, self.vocab_size - 1)
    words, weights = list(counts.keys()), list(counts.values())
    return rng.choices(words, weights=weights)[0]

  def sample_next_trigram(self, w1: int, w2: int, rng: random.Random) -> int:
    counts = self.trigram_counts.get((w1, w2), {})
    if counts:
      words, weights = list(counts.keys()), list(counts.values())
      return rng.choices(words, weights=weights)[0]
    return self.sample_next_bigram(w2, rng)


def main():
  print("====================================================================")
  print(" PROGRAM 30: WORD-LEVEL CODEBOOK & N-GRAM STORY GENERATOR (PYTHON)")
  print("====================================================================\n")

  story_corpus = (
      "once upon a time there was a brave knight who lived in a great stone"
      " castle . the castle stood high on a green hill overlooking a peaceful"
      " kingdom . every morning the brave knight woke up and rode his white"
      " horse through the kingdom . in the kingdom there lived a wise king and"
      " a kind queen who loved their people . one sunny afternoon a little"
      " girl lost her golden ring near the dark river . the brave knight rode"
      " to the dark river to search for the golden ring . he searched under the"
      " tall trees and across the green meadow until he found the ring . the"
      " little girl was happy and the wise king gave the brave knight a shiny"
      " gold medal . and so the people in the kingdom celebrated with music and"
      " laughter all night long . once upon a time there was a clever little"
      " fox who lived in the deep forest . the clever little fox loved to run"
      " across the green meadow under the blue sky . at night the shining stars"
      " lit up the sky and the clever fox rested peacefully . "
  )

  raw_words = split_into_words(story_corpus)
  codebook = WordCodebook()
  codebook.build_from_words(raw_words)

  print(
      f">>> Corpus: {len(raw_words)} words | Vocabulary: {codebook.vocab_size}"
      " unique words\n"
  )

  generator = TrigramStoryGenerator(codebook.vocab_size)
  tokens = codebook.encode(raw_words)
  generator.train(tokens)

  rng = random.Random(42)

  # 2-Gram Generation
  curr = codebook.word_to_id["once"]
  out_2 = [curr]
  for _ in range(20):
    curr = generator.sample_next_bigram(curr, rng)
    out_2.append(curr)
  print(f"[A] 2-Gram Story: \"{codebook.decode(out_2)}\"")

  # 3-Gram Generation
  w1, w2 = codebook.word_to_id["once"], codebook.word_to_id["upon"]
  out_3 = [w1, w2]
  for _ in range(20):
    w3 = generator.sample_next_trigram(w1, w2, rng)
    out_3.append(w3)
    w1, w2 = w2, w3
  print(f"[B] 3-Gram Story: \"{codebook.decode(out_3)}\"")


if __name__ == "__main__":
  main()
