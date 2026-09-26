#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <random>
#include <cmath>
#include <map>
#include <algorithm>

// =============================================================================
// PROGRAM 29: THE CHARACTER CODEBOOK & NEXT-LETTER PREDICTOR (FROM SCRATCH)
//
// 1. Text to Numbers (Encoding): Convert raw letters into integer IDs
// 2. Numbers to Text (Decoding): Convert integer IDs back into human characters
// 3. Transition Grid: Count every pair of consecutive letters in training text
// 4. Probability Table: Turn counts into 100% normalized odds
// 5. Autonomous Text Generator: Roll dice based on probabilities to write text!
// =============================================================================

class CharacterCodebook {
public:
    std::vector<char> id_to_char;
    std::vector<int> char_to_id;
    int vocab_size;

    CharacterCodebook() : char_to_id(256, -1), vocab_size(0) {}

    // Build the codebook from all unique characters in the text
    void build_from_text(const std::string& text) {
        std::vector<bool> seen(256, false);
        for (unsigned char ch : text) {
            seen[ch] = true;
        }

        id_to_char.clear();
        char_to_id.assign(256, -1);

        for (int i = 0; i < 256; ++i) {
            if (seen[i]) {
                char_to_id[i] = id_to_char.size();
                id_to_char.push_back(static_cast<char>(i));
            }
        }
        vocab_size = id_to_char.size();
    }

    // Convert text string -> list of integer IDs
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (unsigned char ch : text) {
            int id = char_to_id[ch];
            if (id != -1) {
                ids.push_back(id);
            }
        }
        return ids;
    }

    // Convert list of integer IDs -> text string
    std::string decode(const std::vector<int>& ids) const {
        std::string text = "";
        text.reserve(ids.size());
        for (int id : ids) {
            if (id >= 0 && id < vocab_size) {
                text += id_to_char[id];
            }
        }
        return text;
    }

    // Display character in printable form (handling space and newline)
    std::string format_char(int id) const {
        char ch = id_to_char[id];
        if (ch == ' ')  return "' '";
        if (ch == '\n') return "'\\n'";
        if (ch == '\t') return "'\\t'";
        return std::string("'") + ch + "'";
    }
};

class NextLetterPredictor {
public:
    int vocab_size;
    // 2D grid: [current_letter_id][next_letter_id]
    std::vector<std::vector<int>> counts;
    std::vector<std::vector<double>> probabilities;

    NextLetterPredictor(int v_size) : vocab_size(v_size) {
        counts.assign(vocab_size, std::vector<int>(vocab_size, 0));
        probabilities.assign(vocab_size, std::vector<double>(vocab_size, 0.0));
    }

    // Count how often each letter is followed by every other letter
    void train(const std::vector<int>& token_ids) {
        for (size_t i = 0; i + 1 < token_ids.size(); ++i) {
            int current_char = token_ids[i];
            int next_char    = token_ids[i + 1];
            counts[current_char][next_char]++;
        }

        // Convert counts to probabilities (summing each row to 100%)
        // We add +1 smoothing (Laplace smoothing) so no letter has 0% chance
        for (int i = 0; i < vocab_size; ++i) {
            double row_total = 0.0;
            for (int j = 0; j < vocab_size; ++j) {
                row_total += (counts[i][j] + 1); // +1 smoothing
            }
            for (int j = 0; j < vocab_size; ++j) {
                probabilities[i][j] = (counts[i][j] + 1) / row_total;
            }
        }
    }

    // Sample the next character by rolling a weighted dice
    int sample_next(int current_char, std::mt19937& rng) const {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);

        double cumulative = 0.0;
        for (int next_char = 0; next_char < vocab_size; ++next_char) {
            cumulative += probabilities[current_char][next_char];
            if (r <= cumulative) {
                return next_char;
            }
        }
        return vocab_size - 1;
    }

    // Calculate how surprised the model is by an unseen text (Loss)
    double calculate_surprise(const std::vector<int>& token_ids) const {
        double total_loss = 0.0;
        int pairs = 0;
        for (size_t i = 0; i + 1 < token_ids.size(); ++i) {
            int c1 = token_ids[i];
            int c2 = token_ids[i + 1];
            double p = probabilities[c1][c2];
            total_loss += -std::log(std::max(1e-12, p));
            pairs++;
        }
        return pairs > 0 ? (total_loss / pairs) : 0.0;
    }
};

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 29: THE CHARACTER CODEBOOK & NEXT-LETTER PREDICTOR (FROM SCRATCH)\n";
    std::cout << " Building the First Building Block of Large Language Models (C++17)\n";
    std::cout << "====================================================================================\n\n";

    // 1. Training Corpus: A collection of English sentences
    std::string training_text = 
        "the quick brown fox jumps over the lazy dog.\n"
        "a king and a queen ruled the quiet kingdom with great wisdom.\n"
        "the cat sat on the warm mat and looked at the blue sky.\n"
        "one small step for a man, one giant leap for mankind.\n"
        "all that glitters is not gold, but shining stars light the dark night.\n"
        "to be or not to be, that is the question we must ask ourselves.\n"
        "knowledge is power, and curiosity is the spark of all learning.\n"
        "the sun sets in the west and rises in the east every single day.\n"
        "children laugh and play together in the green summer meadow.\n"
        "birds fly high above the mountains searching for fresh food and water.\n";

    // 2. Build the Codebook
    CharacterCodebook codebook;
    codebook.build_from_text(training_text);

    std::cout << ">>> Built Character Codebook with " << codebook.vocab_size << " Unique Characters:\n";
    std::cout << "    ";
    for (int i = 0; i < codebook.vocab_size; ++i) {
        std::cout << codebook.format_char(i) << ":" << i << " ";
        if ((i + 1) % 8 == 0) std::cout << "\n    ";
    }
    std::cout << "\n\n";

    // 3. Demonstrate Encoding and Decoding
    std::string test_word = "the cat.";
    std::vector<int> encoded_ids = codebook.encode(test_word);

    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " STEP 1: ENCODING AND DECODING VERIFICATION\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " Human Text: \"" << test_word << "\"\n";
    std::cout << " Computer Numbers (IDs): [ ";
    for (int id : encoded_ids) std::cout << id << " ";
    std::cout << "]\n";

    std::string decoded_back = codebook.decode(encoded_ids);
    std::cout << " Decoded Back from Numbers: \"" << decoded_back << "\"\n";
    std::cout << " Match Check: " << (test_word == decoded_back ? "[PASS] 100% Perfect Match!" : "[FAIL]") << "\n\n";

    // 4. Train the Next-Letter Predictor
    std::vector<int> full_tokens = codebook.encode(training_text);
    NextLetterPredictor predictor(codebook.vocab_size);
    predictor.train(full_tokens);

    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " STEP 2: INSPECTING WHAT THE MODEL LEARNED (TRANSITION ODDS)\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    // Inspect: What follows 't'?
    int id_t = codebook.char_to_id['t'];
    std::cout << " If current letter is " << codebook.format_char(id_t) << " -> Top 5 Most Likely Next Letters:\n";

    std::vector<std::pair<double, int>> top_after_t;
    for (int j = 0; j < codebook.vocab_size; ++j) {
        top_after_t.push_back({predictor.probabilities[id_t][j], j});
    }
    std::sort(top_after_t.rbegin(), top_after_t.rend());

    for (int i = 0; i < 5; ++i) {
        std::cout << "    Next: " << std::setw(6) << std::left << codebook.format_char(top_after_t[i].second)
                  << " | Probability: " << std::fixed << std::setprecision(1) << (top_after_t[i].first * 100.0) << "%\n";
    }
    std::cout << "\n";

    // Inspect: What follows 'q'?
    int id_q = codebook.char_to_id['q'];
    std::cout << " If current letter is " << codebook.format_char(id_q) << " -> Top 3 Most Likely Next Letters:\n";
    std::vector<std::pair<double, int>> top_after_q;
    for (int j = 0; j < codebook.vocab_size; ++j) {
        top_after_q.push_back({predictor.probabilities[id_q][j], j});
    }
    std::sort(top_after_q.rbegin(), top_after_q.rend());

    for (int i = 0; i < 3; ++i) {
        std::cout << "    Next: " << std::setw(6) << std::left << codebook.format_char(top_after_q[i].second)
                  << " | Probability: " << std::fixed << std::setprecision(1) << (top_after_q[i].first * 100.0) << "%\n";
    }
    std::cout << "\n";

    // 5. Autonomous Text Generation
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " STEP 3: AUTONOMOUS TEXT GENERATION (ROLLING THE WEIGHTED DICE)\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    std::mt19937 rng(1337);

    // Prompt 1: Start with 't'
    std::cout << " Prompt 1 (Starting letter 't'):\n \"";
    int current = codebook.char_to_id['t'];
    std::cout << codebook.id_to_char[current];

    for (int step = 0; step < 80; ++step) {
        int next_id = predictor.sample_next(current, rng);
        std::cout << codebook.id_to_char[next_id];
        current = next_id;
    }
    std::cout << "\"\n\n";

    // Prompt 2: Start with 'k'
    std::cout << " Prompt 2 (Starting letter 'k'):\n \"";
    current = codebook.char_to_id['k'];
    std::cout << codebook.id_to_char[current];

    for (int step = 0; step < 80; ++step) {
        int next_id = predictor.sample_next(current, rng);
        std::cout << codebook.id_to_char[next_id];
        current = next_id;
    }
    std::cout << "\"\n\n";

    // 6. Surprise Measurement
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " STEP 4: MEASURING SURPRISE ON UNSEEN TEXT\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    std::string natural_english = "the king looked at the sky.";
    std::string gibberish_text   = "qxzjk pwvl bmfnq zxy jkwp.";

    double surprise_natural   = predictor.calculate_surprise(codebook.encode(natural_english));
    double surprise_gibberish = predictor.calculate_surprise(codebook.encode(gibberish_text));

    std::cout << " Natural English (\"" << natural_english << "\") -> Surprise Loss: " 
              << std::fixed << std::setprecision(3) << surprise_natural << " (Low surprise = Familiar!)\n";
    std::cout << " Gibberish Text  (\"" << gibberish_text   << "\") -> Surprise Loss: " 
              << std::fixed << std::setprecision(3) << surprise_gibberish << " (High surprise = Strange!)\n";

    std::cout << "\n====================================================================================\n";
    std::cout << " PROGRAM 29 COMPLETE: Foundation of text encoding and next-token prediction mastered!\n";
    std::cout << "====================================================================================\n";

    return 0;
}
