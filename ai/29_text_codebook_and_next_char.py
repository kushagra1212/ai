import math
import random


class CharacterCodebook:

  def __init__(self):
    self.id_to_char = []
    self.char_to_id = {}
    self.vocab_size = 0

  def build_from_text(self, text: str):
    unique_chars = sorted(list(set(text)))
    self.id_to_char = unique_chars
    self.char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
    self.vocab_size = len(unique_chars)

  def encode(self, text: str) -> list[int]:
    return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

  def decode(self, ids: list[int]) -> str:
    return ''.join([self.id_to_char[i] for i in ids])

  def format_char(self, idx: int) -> str:
    ch = self.id_to_char[idx]
    if ch == ' ':
      return "' '"
    if ch == '\n':
      return "'\\n'"
    return f"'{ch}'"


class NextLetterPredictor:

  def __init__(self, vocab_size: int):
    self.vocab_size = vocab_size
    self.counts = [[0] * vocab_size for _ in range(vocab_size)]
    self.probabilities = [[0.0] * vocab_size for _ in range(vocab_size)]

  def train(self, token_ids: list[int]):
    for i in range(len(token_ids) - 1):
      c1 = token_ids[i]
      c2 = token_ids[i + 1]
      self.counts[c1][c2] += 1

    # Convert counts to probabilities with +1 Laplace smoothing
    for i in range(self.vocab_size):
      row_total = sum(self.counts[i]) + self.vocab_size
      for j in range(self.vocab_size):
        self.probabilities[i][j] = (self.counts[i][j] + 1) / row_total

  def sample_next(self, current_char: int, rng: random.Random) -> int:
    r = rng.random()
    cumulative = 0.0
    for next_char in range(self.vocab_size):
      cumulative += self.probabilities[current_char][next_char]
      if r <= cumulative:
        return next_char
    return self.vocab_size - 1

  def calculate_surprise(self, token_ids: list[int]) -> float:
    total_loss = 0.0
    pairs = 0
    for i in range(len(token_ids) - 1):
      c1 = token_ids[i]
      c2 = token_ids[i + 1]
      p = self.probabilities[c1][c2]
      total_loss += -math.log(max(1e-12, p))
      pairs += 1
    return total_loss / pairs if pairs > 0 else 0.0


def main():
  print('====================================================================')
  print(' PROGRAM 29: CHARACTER CODEBOOK & NEXT-LETTER PREDICTOR (PYTHON)')
  print('====================================================================\n')

  training_text = (
      'the quick brown fox jumps over the lazy dog.\n'
      'a king and a queen ruled the quiet kingdom with great wisdom.\n'
      'the cat sat on the warm mat and looked at the blue sky.\n'
      'one small step for a man, one giant leap for mankind.\n'
      'all that glitters is not gold, but shining stars light the dark'
      ' night.\n'
      'to be or not to be, that is the question we must ask ourselves.\n'
      'knowledge is power, and curiosity is the spark of all learning.\n'
      'the sun sets in the west and rises in the east every single day.\n'
      'children laugh and play together in the green summer meadow.\n'
      'birds fly high above the mountains searching for fresh food and water.\n'
  )

  codebook = CharacterCodebook()
  codebook.build_from_text(training_text)
  print(f'>>> Built Character Codebook with {codebook.vocab_size} Characters:\n')

  test_word = 'the cat.'
  encoded_ids = codebook.encode(test_word)
  decoded_back = codebook.decode(encoded_ids)
  print(f'Human Text: "{test_word}" -> Numbers: {encoded_ids} -> Decoded:'
        f' "{decoded_back}"\n')

  tokens = codebook.encode(training_text)
  predictor = NextLetterPredictor(codebook.vocab_size)
  predictor.train(tokens)

  # Inspect 't'
  id_t = codebook.char_to_id['t']
  print(
      f'Top letters following {codebook.format_char(id_t)}:'
  )
  top_after_t = sorted(
      [(predictor.probabilities[id_t][j], j) for j in range(codebook.vocab_size)],
      reverse=True,
  )[:5]
  for prob, idx in top_after_t:
    print(f'   Next: {codebook.format_char(idx):<6} | Prob: {prob * 100:.1f}%')

  # Generation
  print('\nAutonomous Text Generation:')
  rng = random.Random(1337)
  curr = codebook.char_to_id['t']
  out = [codebook.id_to_char[curr]]
  for _ in range(80):
    curr = predictor.sample_next(curr, rng)
    out.append(codebook.id_to_char[curr])
  print(f'Prompt "t" -> "{"".join(out)}"')

  # Surprise check
  natural = 'the king looked at the sky.'
  gibberish = 'qxzjk pwvl bmfnq zxy jkwp.'
  print(
      f'\nSurprise Loss (Natural):'
      f' {predictor.calculate_surprise(codebook.encode(natural)):.3f}'
  )
  print(
      f'Surprise Loss (Gibberish):'
      f' {predictor.calculate_surprise(codebook.encode(gibberish)):.3f}'
  )


if __name__ == '__main__':
  main()
