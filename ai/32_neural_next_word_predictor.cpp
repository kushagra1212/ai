#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>

// ====================================================================================
//  PROGRAM 32: THE FIRST NEURAL NEXT-WORD PREDICTOR (FROM FIRST PRINCIPLES)
//  Replacing 500-Terabyte Lookup Tables with Word Coordinates + Hidden Neurons
//  Based on Yoshua Bengio's historic 2003 breakthrough that founded modern LLMs
//  Pure Modern C++17 — No External Machine Learning Libraries
// ====================================================================================

// --- 1. ACTIVATION FUNCTIONS ---
// Leaky ReLU: Keeps gradients alive (slope 1.0 for positive, 0.01 for negative)
inline double leaky_relu(double x) {
    return (x > 0.0) ? x : 0.01 * x;
}

inline double leaky_relu_derivative(double x) {
    return (x > 0.0) ? 1.0 : 0.01;
}

// Softmax: Converts raw judge scores into clean probabilities that sum to 1.0 (100%)
std::vector<double> softmax(const std::vector<double>& logits) {
    double max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val); // Subtract max for numerical stability
        sum += probs[i];
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] /= sum;
    }
    return probs;
}

int main() {
    std::cout << "====================================================================================\n";
    std::cout << " PROGRAM 32: THE FIRST NEURAL NEXT-WORD PREDICTOR (BENGIO ARCHITECTURE)\n";
    std::cout << " Feeding Word Coordinates into Hidden Neurons to Predict the Future Word (C++17)\n";
    std::cout << "====================================================================================\n\n";

    // --------------------------------------------------------------------------------
    // PART 1: THE TRAINING CORPUS (STORY SENTENCES)
    // --------------------------------------------------------------------------------
    std::vector<std::string> sentences = {
        "the king sits on the golden throne .",
        "the queen sits on the golden throne .",
        "the king rules the royal castle .",
        "the queen rules the royal castle .",
        "the prince lives in the royal palace .",
        "the princess lives in the royal palace .",
        "the monkey eats a sweet apple .",
        "the monkey eats a sweet banana .",
        "the monkey eats a juicy orange .",
        "the monkey eats a juicy grape .",
        "apple is a sweet delicious fruit .",
        "banana is a sweet delicious fruit .",
        "orange is a juicy delicious fruit .",
        "grape is a juicy delicious fruit .",
        "the wild wolf hunts in the deep forest .",
        "the wild lion hunts in the deep forest .",
        "the friendly dog runs in the green park .",
        "the friendly cat runs in the green park .",
        "the little girl was happy in the castle .",
        "the brave knight rode through the deep forest ."
    };

    // Build vocabulary
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;

    for (const auto& s : sentences) {
        std::stringstream ss(s);
        std::string w;
        while (ss >> w) {
            if (word_to_id.find(w) == word_to_id.end()) {
                int id = static_cast<int>(vocab.size());
                word_to_id[w] = id;
                id_to_word[id] = w;
                vocab.push_back(w);
            }
        }
    }

    int V = static_cast<int>(vocab.size()); // Vocabulary size
    std::cout << " Total Story Sentences: " << sentences.size() << "\n";
    std::cout << " Vocabulary Size: " << V << " unique words.\n\n";

    // --------------------------------------------------------------------------------
    // PART 2: CREATING TRAINING EXAMPLES (2 CONTEXT WORDS -> 1 TARGET WORD)
    // --------------------------------------------------------------------------------
    // Example: Context = ["the", "king"] -> Target = "sits"
    //          Context = ["king", "sits"] -> Target = "on"
    struct TrainingSample {
        int w1; // First context word ID
        int w2; // Second context word ID
        int target; // Ground-truth next word ID
    };

    std::vector<TrainingSample> dataset;
    for (const auto& s : sentences) {
        std::stringstream ss(s);
        std::string w;
        std::vector<int> tokens;
        while (ss >> w) tokens.push_back(word_to_id[w]);

        for (size_t i = 0; i + 2 < tokens.size(); ++i) {
            dataset.push_back({tokens[i], tokens[i + 1], tokens[i + 2]});
        }
    }

    std::cout << " Created " << dataset.size() << " training training trigram pairs.\n";
    std::cout << " Example training row:\n";
    std::cout << "    Input: [\"" << id_to_word[dataset[0].w1] << "\", \"" 
              << id_to_word[dataset[0].w2] << "\"]  ---> Target: \"" 
              << id_to_word[dataset[0].target] << "\"\n\n";

    // --------------------------------------------------------------------------------
    // PART 3: ARCHITECTURE SIZING (FIRST PRINCIPLES)
    // --------------------------------------------------------------------------------
    const int D = 8;   // Coordinate dials per word (Embedding Dimension)
    const int H = 32;  // Intermediate Hidden Neurons (Concept combinations)
    const int IN_DIM = 2 * D; // Two words combined = 2 * 8 = 16 input dials

    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 2: NEURAL ARCHITECTURE SIZING\n";
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " 1. Word Coordinate Space:    " << D << " dials per word\n";
    std::cout << " 2. Combined Context Input:   " << IN_DIM << " dials (2 words x " << D << ")\n";
    std::cout << " 3. Hidden Neurons (Layer 1): " << H << " neurons\n";
    std::cout << " 4. Output Judges (Layer 2):  " << V << " judges (one for every word in vocab)\n";

    int total_dials = (V * D) + (IN_DIM * H + H) + (H * V + V);
    std::cout << " >>> Total Trainable Dials (Parameters): " << total_dials << " dials!\n";
    std::cout << " Compare to Program 30's brute-force table: " << (V * V * V) << " cells!\n";
    std::cout << " The neural network compresses all combinations into shared coordinate dials!\n\n";

    // --------------------------------------------------------------------------------
    // PART 4: WEIGHT INITIALIZATION
    // --------------------------------------------------------------------------------
    std::mt19937 rng(42);
    // Xavier / He initialization scale
    double scale_emb = 0.1;
    double scale_w1  = std::sqrt(2.0 / IN_DIM);
    double scale_w2  = std::sqrt(2.0 / H);

    std::normal_distribution<double> dist_emb(0.0, scale_emb);
    std::normal_distribution<double> dist_w1(0.0, scale_w1);
    std::normal_distribution<double> dist_w2(0.0, scale_w2);

    // 1. Word Coordinates: C[V][D]
    std::vector<std::vector<double>> C(V, std::vector<double>(D));
    for (int i = 0; i < V; ++i) {
        for (int d = 0; d < D; ++d) C[i][d] = dist_emb(rng);
    }

    // 2. Hidden Layer Weights & Biases: W1[IN_DIM][H], B1[H]
    std::vector<std::vector<double>> W1(IN_DIM, std::vector<double>(H));
    std::vector<double> B1(H, 0.0);
    for (int i = 0; i < IN_DIM; ++i) {
        for (int j = 0; j < H; ++j) W1[i][j] = dist_w1(rng);
    }

    // 3. Output Layer Weights & Biases: W2[H][V], B2[V]
    std::vector<std::vector<double>> W2(H, std::vector<double>(V));
    std::vector<double> B2(V, 0.0);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < V; ++j) W2[i][j] = dist_w2(rng);
    }

    // Momentum velocity buffers for smooth, fast learning
    std::vector<std::vector<double>> vC(V, std::vector<double>(D, 0.0));
    std::vector<std::vector<double>> vW1(IN_DIM, std::vector<double>(H, 0.0));
    std::vector<double> vB1(H, 0.0);
    std::vector<std::vector<double>> vW2(H, std::vector<double>(V, 0.0));
    std::vector<double> vB2(V, 0.0);

    // --------------------------------------------------------------------------------
    // PART 5: THE TRAINING LOOP (FORWARD PASS + BACKPROPAGATION)
    // --------------------------------------------------------------------------------
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << " PART 3: TRAINING THE NEURAL PREDICTOR (LEARNING ENGLISH PATTERNS)\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    double learning_rate = 0.03;
    double momentum = 0.85;
    int epochs = 600;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
        int correct_predictions = 0;

        for (const auto& sample : dataset) {
            // ========================================================================
            // A. FORWARD PASS
            // ========================================================================
            // 1. Look up word coordinates and concatenate into 16-D input vector
            std::vector<double> x(IN_DIM);
            for (int d = 0; d < D; ++d) {
                x[d]     = C[sample.w1][d];
                x[D + d] = C[sample.w2][d];
            }

            // 2. Hidden Layer: h_pre = x * W1 + B1, h = LeakyReLU(h_pre)
            std::vector<double> h_pre(H, 0.0);
            std::vector<double> h(H, 0.0);
            for (int j = 0; j < H; ++j) {
                double sum = B1[j];
                for (int i = 0; i < IN_DIM; ++i) {
                    sum += x[i] * W1[i][j];
                }
                h_pre[j] = sum;
                h[j] = leaky_relu(sum);
            }

            // 3. Output Layer: logits = h * W2 + B2
            std::vector<double> logits(V, 0.0);
            for (int k = 0; k < V; ++k) {
                double sum = B2[k];
                for (int j = 0; j < H; ++j) {
                    sum += h[j] * W2[j][k];
                }
                logits[k] = sum;
            }

            // 4. Softmax Probabilities
            std::vector<double> probs = softmax(logits);

            // Compute Cross-Entropy Loss: -log(P(target))
            double target_prob = std::max(probs[sample.target], 1e-12);
            total_loss += -std::log(target_prob);

            // Check if top-1 predicted word is correct
            int best_word = static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
            if (best_word == sample.target) {
                correct_predictions++;
            }

            // ========================================================================
            // B. BACKPROPAGATION (FIRST PRINCIPLES CHAIN RULE)
            // ========================================================================
            // 1. Output error: dLogits[k] = probs[k] - 1.0 (if k == target) else probs[k]
            std::vector<double> dLogits = probs;
            dLogits[sample.target] -= 1.0;

            // 2. Gradients for W2 and B2
            std::vector<double> dh(H, 0.0);
            for (int j = 0; j < H; ++j) {
                for (int k = 0; k < V; ++k) {
                    dh[j] += dLogits[k] * W2[j][k];
                }
            }

            // 3. Gradients through LeakyReLU activation
            std::vector<double> dh_pre(H, 0.0);
            for (int j = 0; j < H; ++j) {
                dh_pre[j] = dh[j] * leaky_relu_derivative(h_pre[j]);
            }

            // 4. Gradients for W1 and input x
            std::vector<double> dx(IN_DIM, 0.0);
            for (int i = 0; i < IN_DIM; ++i) {
                for (int j = 0; j < H; ++j) {
                    dx[i] += dh_pre[j] * W1[i][j];
                }
            }

            // 5. Update Output Layer (W2, B2) with Momentum
            for (int j = 0; j < H; ++j) {
                for (int k = 0; k < V; ++k) {
                    vW2[j][k] = momentum * vW2[j][k] - learning_rate * (dLogits[k] * h[j]);
                    W2[j][k] += vW2[j][k];
                }
            }
            for (int k = 0; k < V; ++k) {
                vB2[k] = momentum * vB2[k] - learning_rate * dLogits[k];
                B2[k] += vB2[k];
            }

            // 6. Update Hidden Layer (W1, B1) with Momentum
            for (int i = 0; i < IN_DIM; ++i) {
                for (int j = 0; j < H; ++j) {
                    vW1[i][j] = momentum * vW1[i][j] - learning_rate * (dh_pre[j] * x[i]);
                    W1[i][j] += vW1[i][j];
                }
            }
            for (int j = 0; j < H; ++j) {
                vB1[j] = momentum * vB1[j] - learning_rate * dh_pre[j];
                B1[j] += vB1[j];
            }

            // 7. Update Word Coordinates (Embeddings) C with Momentum
            for (int d = 0; d < D; ++d) {
                vC[sample.w1][d] = momentum * vC[sample.w1][d] - learning_rate * dx[d];
                C[sample.w1][d] += vC[sample.w1][d];

                vC[sample.w2][d] = momentum * vC[sample.w2][d] - learning_rate * dx[D + d];
                C[sample.w2][d] += vC[sample.w2][d];
            }
        }

        if (epoch == 1 || epoch % 100 == 0 || epoch == epochs) {
            double avg_loss = total_loss / dataset.size();
            double acc = (100.0 * correct_predictions) / dataset.size();
            std::cout << "   Epoch " << std::setw(3) << epoch 
                      << " / " << epochs 
                      << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss 
                      << " | Next-Word Accuracy: " << std::setprecision(1) << acc << "%\n";
        }
    }

    std::cout << " >>> Training Complete! Neural Network learned syntax and word associations.\n\n";

    // --------------------------------------------------------------------------------
    // PART 6: TESTING PREDICTION ON UNSEEN COMBINATIONS (GENERALIZATION TEST)
    // --------------------------------------------------------------------------------
    std::cout << "====================================================================================\n";
    std::cout << " PART 4: TESTING THE NEURAL BRAIN (LIVE NEXT-WORD PREDICTIONS)\n";
    std::cout << "====================================================================================\n";

    auto predict_next = [&](const std::string& word1, const std::string& word2) {
        if (word_to_id.find(word1) == word_to_id.end() || word_to_id.find(word2) == word_to_id.end()) {
            std::cout << " Error: One of the words is unknown!\n";
            return;
        }

        int id1 = word_to_id[word1];
        int id2 = word_to_id[word2];

        // 1. Input vector from word coordinates
        std::vector<double> x(IN_DIM);
        for (int d = 0; d < D; ++d) {
            x[d]     = C[id1][d];
            x[D + d] = C[id2][d];
        }

        // 2. Hidden layer forward
        std::vector<double> h(H, 0.0);
        for (int j = 0; j < H; ++j) {
            double sum = B1[j];
            for (int i = 0; i < IN_DIM; ++i) sum += x[i] * W1[i][j];
            h[j] = leaky_relu(sum);
        }

        // 3. Output logits
        std::vector<double> logits(V, 0.0);
        for (int k = 0; k < V; ++k) {
            double sum = B2[k];
            for (int j = 0; j < H; ++j) sum += h[j] * W2[j][k];
            logits[k] = sum;
        }

        // 4. Probabilities
        std::vector<double> probs = softmax(logits);

        // Sort top 5 candidate words
        std::vector<std::pair<double, int>> ranked;
        for (int k = 0; k < V; ++k) ranked.push_back({probs[k], k});
        std::sort(ranked.rbegin(), ranked.rend());

        std::cout << " Prompt Context: [\"" << word1 << "\", \"" << word2 << "\"] ---> What comes next?\n";
        for (int k = 0; k < 4; ++k) {
            int wid = ranked[k].second;
            double p = ranked[k].first;
            std::cout << "    #" << (k + 1) << " " 
                      << std::left << std::setw(12) << ("\"" + id_to_word[wid] + "\"") 
                      << " Probability: " << std::fixed << std::setprecision(1) << (p * 100.0) << "%\n";
        }
        std::cout << "\n";
    };

    predict_next("the", "king");
    predict_next("sweet", "delicious");
    predict_next("wild", "wolf");
    predict_next("friendly", "dog");

    // --------------------------------------------------------------------------------
    // PART 7: AUTONOMOUS STORY GENERATION (NEURAL AUTOREGRESSIVE LOOP)
    // --------------------------------------------------------------------------------
    std::cout << "====================================================================================\n";
    std::cout << " PART 5: AUTONOMOUS STORY GENERATION (THE NEURAL LLM LOOP)\n";
    std::cout << " Generating sentences word-by-word by feeding predictions back into the input!\n";
    std::cout << "====================================================================================\n";

    auto generate_story = [&](std::string w1, std::string w2, int length) {
        std::cout << " Story Start: \"" << w1 << " " << w2;

        for (int step = 0; step < length; ++step) {
            if (word_to_id.find(w1) == word_to_id.end() || word_to_id.find(w2) == word_to_id.end()) break;
            int id1 = word_to_id[w1];
            int id2 = word_to_id[w2];

            std::vector<double> x(IN_DIM);
            for (int d = 0; d < D; ++d) {
                x[d]     = C[id1][d];
                x[D + d] = C[id2][d];
            }

            std::vector<double> h(H, 0.0);
            for (int j = 0; j < H; ++j) {
                double sum = B1[j];
                for (int i = 0; i < IN_DIM; ++i) sum += x[i] * W1[i][j];
                h[j] = leaky_relu(sum);
            }

            std::vector<double> logits(V, 0.0);
            for (int k = 0; k < V; ++k) {
                double sum = B2[k];
                for (int j = 0; j < H; ++j) sum += h[j] * W2[j][k];
                logits[k] = sum;
            }

            std::vector<double> probs = softmax(logits);

            // Greedy selection (pick most probable word)
            int best_id = static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
            std::string next_word = id_to_word[best_id];

            std::cout << " " << next_word;

            // Slide window: [w1, w2] becomes [w2, next_word]!
            w1 = w2;
            w2 = next_word;
            if (next_word == ".") break;
        }
        std::cout << "\"\n";
    };

    generate_story("the", "queen", 12);
    generate_story("the", "monkey", 12);
    generate_story("the", "wild", 12);
    generate_story("the", "brave", 12);
    std::cout << "\n";

    // --------------------------------------------------------------------------------
    // PART 8: INDUSTRY TERMINOLOGY (BY THE WAY...)
    // --------------------------------------------------------------------------------
    std::cout << "====================================================================================\n";
    std::cout << " PART 6: BY THE WAY... WHAT THE AI INDUSTRY CALLS THESE CONCEPTS\n";
    std::cout << "====================================================================================\n";
    std::cout << " 1. 'Neural Language Model (NLM)':\n";
    std::cout << "    A neural network that predicts the next word given previous words as continuous coordinates.\n";
    std::cout << "    First created by Yoshua Bengio in 2003, winning him the Turing Award in AI.\n\n";
    std::cout << " 2. 'Autoregressive Generation':\n";
    std::cout << "    The loop where the predicted word is appended to the input to predict the next word.\n";
    std::cout << "    This is the exact same fundamental loop used by ChatGPT, Claude, and Llama today!\n\n";
    std::cout << " 3. 'Softmax + Cross-Entropy Loss':\n";
    std::cout << "    The mathematical loss function used across ALL modern LLMs to train next-token prediction.\n\n";
    std::cout << " 4. 'The Foundation for Transformers':\n";
    std::cout << "    In Program 32, our context is fixed to 2 words. In upcoming programs, we will expand\n";
    std::cout << "    this with Attention so the network can remember thousands of words at once!\n";
    std::cout << "====================================================================================\n";

    return 0;
}
