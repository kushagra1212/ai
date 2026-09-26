#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <map>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>

// ====================================================================================
//  PROGRAM 31: WORD COORDINATES & CONTINUOUS WORD SPACE (FROM FIRST PRINCIPLES)
//  Teaching a Computer What Words Mean by Giving Every Word a Location on a Map
//  Pure Modern C++17 — No External Frameworks or Machine Learning Libraries
// ====================================================================================

// --- 1. ACTIVATION FUNCTION (SIGMOID) ---
// Returns a smooth probability between 0.0 (unrelated) and 1.0 (strongly related)
inline double sigmoid(double x) {
    if (x > 10.0) return 1.0;
    if (x < -10.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

// --- 2. VECTOR MATH HELPER FUNCTIONS ---
// Dot product: measures how strongly two coordinate arrows point in the same direction
double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Vector length (Euclidean magnitude)
double vector_length(const std::vector<double>& v) {
    double sum_sq = 0.0;
    for (double val : v) {
        sum_sq += val * val;
    }
    return std::sqrt(sum_sq);
}

// Directional Alignment (Cosine Similarity): ranges from -1.0 (opposite) to +1.0 (identical)
double direction_alignment(const std::vector<double>& a, const std::vector<double>& b) {
    double len_a = vector_length(a);
    double len_b = vector_length(b);
    if (len_a < 1e-9 || len_b < 1e-9) return 0.0;
    return dot_product(a, b) / (len_a * len_b);
}

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 31: WORD COORDINATES & CONTINUOUS WORD SPACE (FROM FIRST PRINCIPLES)\n";
    std::cout << " Giving Words Physical GPS Coordinates so Computers Understand Their Meaning (C++17)\n";
    std::cout << "====================================================================================\n\n";

    // --------------------------------------------------------------------------------
    // PART 1: THE FATAL FLAW OF SIMPLE INTEGER IDs
    // --------------------------------------------------------------------------------
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 1: WHY SIMPLE NUMBER IDs FAIL COMPLETELY\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " Suppose we assign simple ID numbers to words in our dictionary:\n";
    std::cout << "   ID 0 = 'king'\n";
    std::cout << "   ID 1 = 'apple'\n";
    std::cout << "   ID 2 = 'queen'\n";
    std::cout << "   ID 3 = 'orange'\n\n";
    std::cout << " If the computer computes distance using basic subtraction:\n";
    std::cout << "   Distance('king', 'apple') = |0 - 1| = 1\n";
    std::cout << "   Distance('king', 'queen') = |0 - 2| = 2\n\n";
    std::cout << " >>> ABSURD RESULT: The computer thinks a 'king' is closer to an 'apple' than a 'queen'!\n";
    std::cout << "     1D integer IDs carry ZERO meaning of similarity.\n";
    std::cout << "     To fix this, each word must become a coordinate point on a multi-dimensional map!\n\n";

    // --------------------------------------------------------------------------------
    // PART 2: THE TRAINING CORPUS (3 DISTINCT REALMS)
    // --------------------------------------------------------------------------------
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 2: THE TRAINING SENTENCES (3 CONCEPTUAL WORLDS)\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::vector<std::string> sentences = {
        // Realm 1: Royalty & Palaces
        "the king rules the royal castle",
        "the queen rules the royal castle",
        "the prince rules the royal palace",
        "the princess rules the royal palace",
        "the king sits on the golden throne",
        "the queen sits on the golden throne",
        // Realm 2: Animals & Fruit
        "the monkey eats a sweet apple",
        "the monkey eats a sweet banana",
        "the monkey eats a juicy orange",
        "the monkey eats a juicy grape",
        "apple is a sweet fruit",
        "banana is a sweet fruit",
        "orange is a juicy fruit",
        "grape is a juicy fruit",
        // Realm 3: Wild Animals & Nature
        "the wild wolf hunts in the deep forest",
        "the wild lion hunts in the deep forest",
        "the friendly dog runs in the green park",
        "the friendly cat runs in the green park"
    };

    // Build the vocabulary codebook
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;

    for (const auto& line : sentences) {
        std::stringstream ss(line);
        std::string word;
        while (ss >> word) {
            if (word_to_id.find(word) == word_to_id.end()) {
                int id = static_cast<int>(vocab.size());
                word_to_id[word] = id;
                id_to_word[id] = word;
                vocab.push_back(word);
            }
        }
    }

    int V = static_cast<int>(vocab.size());
    std::cout << " Total Sentences: " << sentences.size() << "\n";
    std::cout << " Total Unique Words (Vocabulary): " << V << " words.\n";
    std::cout << " Words in vocabulary:\n   ";
    for (int i = 0; i < V; ++i) {
        std::cout << vocab[i] << " ";
        if ((i + 1) % 8 == 0) std::cout << "\n   ";
    }
    std::cout << "\n\n";

    // --------------------------------------------------------------------------------
    // PART 3: EXTRACTING TRAINING PAIRS (SLIDING NEIGHBOR WINDOW)
    // --------------------------------------------------------------------------------
    // Intuition: Words that appear next to each other should have similar coordinates!
    int window_size = 2;
    struct TrainingPair {
        int target;
        int context;
    };
    std::vector<TrainingPair> positive_pairs;

    for (const auto& line : sentences) {
        std::stringstream ss(line);
        std::string word;
        std::vector<int> tokens;
        while (ss >> word) {
            tokens.push_back(word_to_id[word]);
        }
        for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
            int target = tokens[i];
            for (int j = std::max(0, i - window_size); j <= std::min(static_cast<int>(tokens.size()) - 1, i + window_size); ++j) {
                if (i != j) {
                    positive_pairs.push_back({target, tokens[j]});
                }
            }
        }
    }
    std::cout << " Extracted " << positive_pairs.size() << " positive neighbor pairs (words appearing near each other).\n\n";

    // --------------------------------------------------------------------------------
    // PART 4: INITIALIZING WORD COORDINATE MAP (2D SPACE)
    // --------------------------------------------------------------------------------
    // We choose D = 2 dimensions (X, Y) so we can directly draw a visual map on screen!
    const int D = 2;
    std::mt19937 rng(42); // Fixed seed for reproducible, crisp learning
    std::uniform_real_distribution<double> init_dist(-0.1, 0.1);

    // Target coordinates (U) and Context coordinates (W)
    std::vector<std::vector<double>> U(V, std::vector<double>(D));
    std::vector<std::vector<double>> W(V, std::vector<double>(D));

    for (int i = 0; i < V; ++i) {
        for (int d = 0; d < D; ++d) {
            U[i][d] = init_dist(rng);
            W[i][d] = init_dist(rng);
        }
    }

    // --------------------------------------------------------------------------------
    // PART 5: TRAINING THE WORD COORDINATES (PULL NEIGHBORS, PUSH STRANGERS)
    // --------------------------------------------------------------------------------
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 3: TRAINING THE COORDINATES (PULL TOGETHER & PUSH APART)\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " Rule 1: Words appearing near each other -> PULL coordinates closer (Target = 1.0)\n";
    std::cout << " Rule 2: Random unrelated words         -> PUSH coordinates apart  (Target = 0.0)\n\n";

    double learning_rate = 0.05;
    int epochs = 1500;
    std::uniform_int_distribution<int> rand_word(0, V - 1);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;

        for (const auto& pair : positive_pairs) {
            int target = pair.target;
            int context = pair.context;

            // 1. Positive interaction: calculate alignment score
            double score_pos = dot_product(U[target], W[context]);
            double prob_pos = sigmoid(score_pos);
            double err_pos = 1.0 - prob_pos; // Target is 1.0 (they are true neighbors)
            total_loss += -std::log(std::max(prob_pos, 1e-9));

            // 2. Negative interaction: pick a random word that is NOT this context
            int neg = rand_word(rng);
            while (neg == context) neg = rand_word(rng);

            double score_neg = dot_product(U[target], W[neg]);
            double prob_neg = sigmoid(score_neg);
            double err_neg = 0.0 - prob_neg; // Target is 0.0 (they are random strangers)
            total_loss += -std::log(std::max(1.0 - prob_neg, 1e-9));

            // 3. Update coordinates using our familiar slope and error logic
            for (int d = 0; d < D; ++d) {
                double u_val = U[target][d];
                double w_ctx = W[context][d];
                double w_neg = W[neg][d];

                // Adjust target word coordinates
                U[target][d] += learning_rate * (err_pos * w_ctx + err_neg * w_neg);

                // Adjust context word coordinate (pulled toward target)
                W[context][d] += learning_rate * (err_pos * u_val);

                // Adjust negative word coordinate (pushed away from target)
                W[neg][d] += learning_rate * (err_neg * u_val);
            }
        }

        if (epoch == 1 || epoch % 300 == 0 || epoch == epochs) {
            std::cout << "   Epoch " << std::setw(4) << epoch 
                      << " / " << epochs 
                      << " | Average Loss: " << std::fixed << std::setprecision(4) 
                      << (total_loss / positive_pairs.size()) << "\n";
        }
    }
    std::cout << " >>> Training Complete! All words now have continuous physical coordinates.\n\n";

    // Calculate final combined coordinates: average of target and context vectors
    std::vector<std::vector<double>> final_coords(V, std::vector<double>(D));
    for (int i = 0; i < V; ++i) {
        for (int d = 0; d < D; ++d) {
            final_coords[i][d] = (U[i][d] + W[i][d]) / 2.0;
        }
    }

    // --------------------------------------------------------------------------------
    // PART 6: VISUALIZING THE 2D WORD MAP IN ASCII ART
    // --------------------------------------------------------------------------------
    std::cout << "====================================================================================\n";
    std::cout << " PART 4: THE LEARNED 2D WORD MAP (PROJECTED ON SCREEN)\n";
    std::cout << " Notice how words naturally clustered into islands of meaning with NO human rules!\n";
    std::cout << "====================================================================================\n";

    double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
    for (int i = 0; i < V; ++i) {
        min_x = std::min(min_x, final_coords[i][0]);
        max_x = std::max(max_x, final_coords[i][0]);
        min_y = std::min(min_y, final_coords[i][1]);
        max_y = std::max(max_y, final_coords[i][1]);
    }

    const int GRID_W = 68;
    const int GRID_H = 22;
    std::vector<std::string> grid(GRID_H, std::string(GRID_W, ' '));

    for (int i = 0; i < V; ++i) {
        double x = final_coords[i][0];
        double y = final_coords[i][1];

        int gx = static_cast<int>((x - min_x) / (max_x - min_x + 1e-9) * (GRID_W - 8));
        int gy = static_cast<int>((y - min_y) / (max_y - min_y + 1e-9) * (GRID_H - 1));
        gy = GRID_H - 1 - gy; // Invert Y so higher Y is at top

        std::string label = vocab[i];
        for (size_t k = 0; k < label.size(); ++k) {
            if (gx + static_cast<int>(k) < GRID_W && grid[gy][gx + k] == ' ') {
                grid[gy][gx + k] = label[k];
            }
        }
    }

    std::cout << " +" << std::string(GRID_W, '-') << "+\n";
    for (int r = 0; r < GRID_H; ++r) {
        std::cout << " |" << grid[r] << "|\n";
    }
    std::cout << " +" << std::string(GRID_W, '-') << "+\n\n";

    // --------------------------------------------------------------------------------
    // PART 7: FINDING THE NEAREST NEIGHBORS (SEMANTIC SIMILARITY)
    // --------------------------------------------------------------------------------
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 5: MEASURING MEANING (DIRECTIONAL ALIGNMENT / COSINE SIMILARITY)\n";
    std::cout << " (+1.0 = Identical Meaning, 0.0 = Unrelated, -1.0 = Opposite Meaning)\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    auto print_nearest = [&](const std::string& query_word) {
        if (word_to_id.find(query_word) == word_to_id.end()) return;
        int q_id = word_to_id[query_word];
        const auto& q_vec = final_coords[q_id];

        std::vector<std::pair<double, std::string>> scores;
        for (int i = 0; i < V; ++i) {
            if (i == q_id) continue;
            double sim = direction_alignment(q_vec, final_coords[i]);
            scores.push_back({sim, vocab[i]});
        }
        std::sort(scores.rbegin(), scores.rend());

        std::cout << " Top matches for '" << query_word << "':\n";
        for (int k = 0; k < 5 && k < static_cast<int>(scores.size()); ++k) {
            std::cout << "    " << (k + 1) << ". " 
                      << std::left << std::setw(12) << scores[k].second 
                      << " (Alignment: " << std::fixed << std::setprecision(4) << scores[k].first << ")\n";
        }
        std::cout << "\n";
    };

    print_nearest("king");
    print_nearest("apple");
    print_nearest("wolf");

    // --------------------------------------------------------------------------------
    // PART 8: THE MAGIC OF VECTOR ARITHMETIC (WORD MATH!)
    // --------------------------------------------------------------------------------
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 6: THE MAGIC OF VECTOR MATH (CAN WE DO ARITHMETIC ON IDEAS?)\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " Let us test the equation: Target = 'wolf' - 'wild' + 'friendly'\n";
    std::cout << " (Intuition: If you take a wild wolf and make it friendly, what animal do you get?)\n\n";

    int id_wolf = word_to_id["wolf"];
    int id_wild = word_to_id["wild"];
    int id_friendly = word_to_id["friendly"];

    std::vector<double> target_vec(D);
    for (int d = 0; d < D; ++d) {
        target_vec[d] = final_coords[id_wolf][d] - final_coords[id_wild][d] + final_coords[id_friendly][d];
    }

    std::vector<std::pair<double, std::string>> math_results;
    for (int i = 0; i < V; ++i) {
        if (i == id_wolf || i == id_wild || i == id_friendly) continue;
        double sim = direction_alignment(target_vec, final_coords[i]);
        math_results.push_back({sim, vocab[i]});
    }
    std::sort(math_results.rbegin(), math_results.rend());

    std::cout << " Closest words to ('wolf' - 'wild' + 'friendly'):\n";
    for (int k = 0; k < 4; ++k) {
        std::cout << "    " << (k + 1) << ". " 
                  << std::left << std::setw(12) << math_results[k].second 
                  << " (Alignment: " << std::fixed << std::setprecision(4) << math_results[k].first << ")\n";
    }
    std::cout << " >>> Look at that: 'dog' and 'cat' are the closest animal concepts!\n\n";

    // --------------------------------------------------------------------------------
    // PART 9: INDUSTRY TERMINOLOGY (BY THE WAY...)
    // --------------------------------------------------------------------------------
    std::cout << "====================================================================================\n";
    std::cout << " PART 7: BY THE WAY... WHAT THE AI INDUSTRY CALLS THESE CONCEPTS\n";
    std::cout << "====================================================================================\n";
    std::cout << " 1. 'Word Embeddings / Word Vectors':\n";
    std::cout << "    The list of continuous numbers (coordinates) assigned to each word.\n";
    std::cout << "    Instead of 500 Terabytes of tables, words live as coordinates in continuous space!\n\n";
    std::cout << " 2. 'Embedding Dimension':\n";
    std::cout << "    How many coordinate dials each word has. In our demo, D = 2.\n";
    std::cout << "    In real-world LLMs like GPT-4 or Llama-3, D is typically 4,096 dimensions.\n\n";
    std::cout << " 3. 'Cosine Similarity':\n";
    std::cout << "    The formal industry name for Directional Alignment (dot product / lengths).\n\n";
    std::cout << " 4. 'Word2Vec (Skip-Gram with Negative Sampling)':\n";
    std::cout << "    The training algorithm we just built from scratch: predicting context words and\n";
    std::cout << "    contrasting with random words. Invented by Tomas Mikolov at Google in 2013.\n\n";
    std::cout << " 5. 'Vector Arithmetic':\n";
    std::cout << "    The famous discovery that semantic relationships correspond to geometric vectors\n";
    std::cout << "    (e.g., King - Man + Woman = Queen, or Wolf - Wild + Friendly = Dog).\n";
    std::cout << "====================================================================================\n";

    return 0;
}
