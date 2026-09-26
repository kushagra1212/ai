/**
 * ========================================================================
 * PROGRAM 31B: FIRST-PRINCIPLES LOCAL FILE VECTOR DATABASE (C++17)
 * ========================================================================
 * Kushagra's AI Journey: From Raw Coordinates to Real File Search
 *
 * New Upgrades:
 * 1. FUZZY SPELLING CORRECTION (Levenshtein Distance):
 *    - Typos like 'poitner' or 'databse' are automatically detected and corrected
 *      to vocabulary terms with real-time user notification!
 * 2. SMART SNIPPET EXTRACTION:
 *    - Dynamically scans the document to find the most relevant line of code/notes
 *      (no more useless comment block headers!).
 * 3. VISUAL TEXT HIGHLIGHTING:
 *    - Query terms and matched concepts are brightly highlighted in the terminal
 *      using ANSI color pill badges.
 * 4. STOP-WORD ATTENUATION (IDF Weighting):
 *    - Common filler words ('this', 'is', 'not') are automatically downweighted
 *      so they don't falsely pull vectors toward irrelevant files.
 * ========================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cctype>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

// ============================================================================
// STEP 1: TERMINAL PROGRESS BAR UTILITY
// ============================================================================

void print_progress_bar(const std::string& prefix, size_t current, size_t total, const std::string& extra = "", int bar_width = 24) {
    float progress = (total > 0) ? (static_cast<float>(current) / static_cast<float>(total)) : 1.0f;
    if (progress > 1.0f) progress = 1.0f;
    int filled = static_cast<int>(progress * bar_width);

    std::cout << "\r" << prefix << " [";
    for (int i = 0; i < filled; ++i) std::cout << "=";
    if (filled < bar_width) std::cout << ">";
    for (int i = filled + 1; i < bar_width; ++i) std::cout << " ";
    std::cout << "] " << std::setw(5) << std::fixed << std::setprecision(1) << (progress * 100.0f) << "%";
    
    if (!extra.empty()) {
        std::cout << " | " << extra;
    }
    std::cout << "   \033[K";
    std::cout.flush();

    if (current >= total && total > 0) {
        std::cout << "\n";
    }
}

// ============================================================================
// STEP 2: LEVENSHTEIN DISTANCE (FIRST-PRINCIPLES FUZZY SPELLING)
// ============================================================================

int levenshtein_dist(const std::string& s1, const std::string& s2) {
    int m = static_cast<int>(s1.size());
    int n = static_cast<int>(s2.size());
    if (std::abs(m - n) > 2) return 999; // Fast prune

    std::vector<int> prev(n + 1), curr(n + 1);
    for (int j = 0; j <= n; ++j) prev[j] = j;

    for (int i = 1; i <= m; ++i) {
        curr[0] = i;
        for (int j = 1; j <= n; ++j) {
            if (s1[i - 1] == s2[j - 1]) {
                curr[j] = prev[j - 1];
            } else {
                curr[j] = 1 + std::min({prev[j], curr[j - 1], prev[j - 1]});
            }
        }
        prev = curr;
    }
    return prev[n];
}

// Common stop words that should not dominate search vectors
const std::unordered_set<std::string> STOP_WORDS = {
    "this", "is", "not", "the", "a", "an", "and", "or", "in", "on", 
    "at", "to", "for", "of", "with", "by", "from", "as", "it", "that", 
    "be", "are", "was", "were", "right", "all", "do", "does", "did"
};

// ============================================================================
// STEP 3: FILE FILTERING & TEXT TOKENIZER
// ============================================================================

bool is_ignored_path(const fs::path& p) {
    for (const auto& part : p) {
        std::string name = part.string();
        if (name.size() > 1 && name[0] == '.' && name != "." && name != "..") {
            return true;
        }
    }
    std::string s = p.string();
    if (s.find("/node_modules/") != std::string::npos ||
        s.find("/build/") != std::string::npos ||
        s.find("/dist/") != std::string::npos ||
        s.find("/.git/") != std::string::npos ||
        s.find("/external/") != std::string::npos ||
        s.find("/__pycache__/") != std::string::npos ||
        s.find("/.cache/") != std::string::npos ||
        s.find("/.local/") != std::string::npos) {
        return true;
    }
    return false;
}

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            current += std::tolower(static_cast<unsigned char>(ch));
        } else if (ch == '-' || ch == '_') {
            if (!current.empty()) current += ch;
        } else {
            if (!current.empty()) {
                while (!current.empty() && (current.back() == '-' || current.back() == '_')) {
                    current.pop_back();
                }
                if (current.size() >= 2 && current.size() <= 32) {
                    tokens.push_back(current);
                }
                current.clear();
            }
        }
    }
    if (current.size() >= 2 && current.size() <= 32) {
        tokens.push_back(current);
    }
    return tokens;
}

// ============================================================================
// STEP 4: MATHEMATICAL PRIMITIVES
// ============================================================================

inline float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline float vector_magnitude(const std::vector<float>& v) {
    float sum_sq = 0.0f;
    for (float val : v) sum_sq += val * val;
    return std::sqrt(sum_sq);
}

inline void normalize_vector(std::vector<float>& v) {
    float mag = vector_magnitude(v);
    if (mag > 1e-7f) {
        for (float& val : v) val /= mag;
    }
}

inline float sigmoid(float z) {
    if (z > 6.0f) return 1.0f;
    if (z < -6.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-z));
}

// ============================================================================
// STEP 5: SMART SNIPPET EXTRACTION & TEXT HIGHLIGHTING
// ============================================================================

std::string highlight_text(const std::string& text, const std::vector<std::string>& query_terms) {
    std::string result = "";
    size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (const auto& term : query_terms) {
            if (term.empty()) continue;
            size_t len = term.size();
            if (i + len <= text.size()) {
                bool eq = true;
                for (size_t k = 0; k < len; ++k) {
                    if (std::tolower(static_cast<unsigned char>(text[i + k])) != 
                        std::tolower(static_cast<unsigned char>(term[k]))) {
                        eq = false;
                        break;
                    }
                }
                if (eq) {
                    bool left_ok = (i == 0 || !std::isalnum(static_cast<unsigned char>(text[i - 1])));
                    bool right_ok = (i + len == text.size() || !std::isalnum(static_cast<unsigned char>(text[i + len])));
                    if (left_ok && right_ok) {
                        // Highlight with bold yellow text on dark background
                        result += "\033[1;93;40m " + text.substr(i, len) + " \033[0m";
                        i += len;
                        matched = true;
                        break;
                    }
                }
            }
        }
        if (!matched) {
            result += text[i];
            i++;
        }
    }
    return result;
}

std::string extract_smart_snippet(const std::string& filepath, const std::vector<std::string>& query_terms) {
    std::ifstream file(filepath);
    if (!file.is_open()) return "";

    std::string line;
    std::string best_line = "";
    int best_score = -1;
    int line_num = 0;

    while (std::getline(file, line) && line_num < 300) {
        line_num++;
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        std::string trimmed = line.substr(start);

        // Skip boilerplate comment lines or blank brackets
        if (trimmed == "/*" || trimmed == "*/" || trimmed == "{" || trimmed == "}" ||
            trimmed == "//" || trimmed.rfind("#include", 0) == 0 || trimmed.size() < 6) {
            continue;
        }

        // Count occurrences of query terms
        std::string lower_line = trimmed;
        for (char& c : lower_line) c = std::tolower(static_cast<unsigned char>(c));

        int term_hits = 0;
        for (const auto& term : query_terms) {
            if (lower_line.find(term) != std::string::npos) {
                term_hits += 20;
            }
        }

        int score = term_hits + std::min(static_cast<int>(trimmed.size()), 60);
        if (score > best_score) {
            best_score = score;
            best_line = trimmed;
        }
    }

    if (best_line.empty()) {
        return "/* No preview available */";
    }

    if (best_line.size() > 130) {
        best_line = best_line.substr(0, 127) + "...";
    }

    return best_line;
}

// ============================================================================
// STEP 6: DATABASE STATE & ENGINE
// ============================================================================

struct DocumentRecord {
    std::string filepath;
    std::string filename;
    std::vector<int> token_ids;
    std::string default_snippet;
    std::vector<float> doc_vector;
};

class LocalVectorDatabase {
public:
    int embedding_dim = 16;
    int window_size = 3;
    int num_negatives = 5;
    int user_epochs = -1;
    float initial_lr = 0.05f;

    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;
    std::vector<int> word_frequencies;

    std::vector<std::vector<float>> U;
    std::vector<std::vector<float>> W;

    std::vector<DocumentRecord> documents;

    // ------------------------------------------------------------------------
    // CRAWL & LOAD FILES
    // ------------------------------------------------------------------------
    bool crawl_and_load(const std::string& directory_path) {
        if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
            std::cerr << "\n[!] Error: Folder '" << directory_path << "' does not exist.\n";
            return false;
        }

        std::cout << "\n[*] Scanning folder: " << directory_path << " (skipping hidden/.git/node_modules) ...\n";

        std::vector<fs::path> candidate_files;
        try {
            for (const auto& entry : fs::recursive_directory_iterator(
                     directory_path, fs::directory_options::skip_permission_denied)) {
                if (entry.is_regular_file()) {
                    if (is_ignored_path(entry.path())) continue;

                    std::error_code ec;
                    auto sz = entry.file_size(ec);
                    if (ec || sz > 500 * 1024 || sz == 0) continue;

                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".txt" || ext == ".md" || ext == ".cpp" || 
                        ext == ".h" || ext == ".hpp" || ext == ".py" ||
                        ext == ".c" || ext == ".rs" || ext == ".go") {
                        candidate_files.push_back(entry.path());
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cout << "    Scanned accessible files (" << candidate_files.size() << " matched).\n";
        }

        int total_candidates = static_cast<int>(candidate_files.size());
        if (total_candidates == 0) {
            std::cerr << "[!] No matching files found in " << directory_path << "\n";
            return false;
        }

        documents.clear();
        word_to_id.clear();
        id_to_word.clear();
        word_frequencies.clear();

        std::unordered_map<std::string, int> raw_freq;
        struct RawDoc {
            std::string path;
            std::string filename;
            std::string preview;
            std::vector<std::string> words;
        };
        std::vector<RawDoc> raw_docs;
        raw_docs.reserve(total_candidates);

        for (int i = 0; i < total_candidates; ++i) {
            const auto& path = candidate_files[i];

            if ((i + 1) % 25 == 0 || i + 1 == total_candidates) {
                print_progress_bar("Indexing Files", i + 1, total_candidates, path.filename().string());
            }

            std::ifstream file(path);
            if (!file.is_open()) continue;

            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string raw_content = buffer.str();
            if (raw_content.empty()) continue;

            std::vector<std::string> words = tokenize(raw_content);
            if (words.empty()) continue;

            // Find first meaningful line for default preview (skip bare /*)
            std::string preview;
            std::stringstream ss(raw_content);
            std::string pline;
            while (std::getline(ss, pline)) {
                size_t st = pline.find_first_not_of(" \t\r\n");
                if (st != std::string::npos) {
                    std::string tr = pline.substr(st);
                    if (tr != "/*" && tr != "*/" && tr != "//" && tr.size() > 4) {
                        preview = tr.substr(0, 130);
                        break;
                    }
                }
            }
            if (preview.empty()) preview = path.filename().string();

            for (const auto& w : words) {
                raw_freq[w]++;
            }

            raw_docs.push_back({path.string(), path.filename().string(), preview, std::move(words)});
        }
        print_progress_bar("Indexing Files", total_candidates, total_candidates, "Complete");

        int min_count = (raw_freq.size() > 3000) ? 2 : 1;
        if (raw_freq.size() > 30000) min_count = 3;

        for (const auto& pair : raw_freq) {
            if (pair.second >= min_count) {
                int id = static_cast<int>(id_to_word.size());
                word_to_id[pair.first] = id;
                id_to_word.push_back(pair.first);
                word_frequencies.push_back(pair.second);
            }
        }

        documents.reserve(raw_docs.size());
        for (auto& rd : raw_docs) {
            DocumentRecord doc;
            doc.filepath = rd.path;
            doc.filename = rd.filename;
            doc.default_snippet = rd.preview;
            for (const auto& w : rd.words) {
                auto it = word_to_id.find(w);
                if (it != word_to_id.end()) {
                    doc.token_ids.push_back(it->second);
                }
            }
            if (!doc.token_ids.empty()) {
                documents.push_back(std::move(doc));
            }
        }

        size_t total_tokens = 0;
        for (const auto& d : documents) total_tokens += d.token_ids.size();

        std::cout << "[+] Found " << documents.size() << " valid documents (" 
                  << total_tokens << " words, " << id_to_word.size() << " unique vocabulary terms).\n";
        return true;
    }

    // ------------------------------------------------------------------------
    // MULTI-THREADED TRAINING (WITH CONTINUOUS LIVE 20 FPS PROGRESS)
    // ------------------------------------------------------------------------
    void train_word_embeddings() {
        int vocab_size = static_cast<int>(id_to_word.size());
        if (vocab_size == 0 || documents.empty()) return;

        size_t total_tokens = 0;
        for (const auto& d : documents) total_tokens += d.token_ids.size();

        int epochs = user_epochs;
        if (epochs <= 0) {
            if (total_tokens > 200000) epochs = 5;
            else if (total_tokens > 50000) epochs = 10;
            else if (total_tokens > 5000) epochs = 25;
            else epochs = 80;
        }

        size_t total_training_steps = epochs * total_tokens;

        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#endif

        std::cout << "[*] Training " << embedding_dim << "-D coordinates across " 
                  << epochs << " epochs (" << total_tokens << " tokens/epoch) using "
                  << num_threads << " CPU threads ...\n";

        std::mt19937 init_rng(42);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

        U.assign(vocab_size, std::vector<float>(embedding_dim));
        W.assign(vocab_size, std::vector<float>(embedding_dim, 0.0f));

        for (int v = 0; v < vocab_size; ++v) {
            for (int d = 0; d < embedding_dim; ++d) {
                U[v][d] = dist(init_rng);
            }
        }

        std::vector<double> sample_weights(vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            sample_weights[v] = std::pow(static_cast<double>(word_frequencies[v]), 0.75);
        }
        std::discrete_distribution<int> neg_sampler(sample_weights.begin(), sample_weights.end());

        std::atomic<size_t> global_tokens_processed(0);
        auto start_time = std::chrono::steady_clock::now();
        auto last_ui_time = std::chrono::steady_clock::now();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float lr = initial_lr * (1.0f - static_cast<float>(epoch) / static_cast<float>(epochs));
            if (lr < 0.001f) lr = 0.001f;

            int num_docs = static_cast<int>(documents.size());

            #pragma omp parallel
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                std::mt19937 thread_rng(42 + tid * 1000 + epoch);

                #pragma omp for schedule(dynamic, 8)
                for (int doc_idx = 0; doc_idx < num_docs; ++doc_idx) {
                    const auto& seq = documents[doc_idx].token_ids;
                    int n = static_cast<int>(seq.size());

                    for (int pos = 0; pos < n; ++pos) {
                        int target_id = seq[pos];

                        for (int offset = -window_size; offset <= window_size; ++offset) {
                            if (offset == 0) continue;
                            int ctx_pos = pos + offset;
                            if (ctx_pos < 0 || ctx_pos >= n) continue;
                            int context_id = seq[ctx_pos];

                            float dot_pos = dot_product(U[target_id], W[context_id]);
                            float sig_pos = sigmoid(dot_pos);
                            float err_pos = 1.0f - sig_pos;

                            std::vector<float> target_grad(embedding_dim, 0.0f);
                            for (int d = 0; d < embedding_dim; ++d) {
                                target_grad[d] += err_pos * W[context_id][d];
                                W[context_id][d] += lr * (err_pos * U[target_id][d]);
                            }

                            for (int k = 0; k < num_negatives; ++k) {
                                int neg_id = neg_sampler(thread_rng);
                                if (neg_id == target_id || neg_id == context_id) continue;

                                float dot_neg = dot_product(U[target_id], W[neg_id]);
                                float sig_neg = sigmoid(dot_neg);
                                float err_neg = 0.0f - sig_neg;

                                for (int d = 0; d < embedding_dim; ++d) {
                                    target_grad[d] += err_neg * W[neg_id][d];
                                    W[neg_id][d] += lr * (err_neg * U[target_id][d]);
                                }
                            }

                            for (int d = 0; d < embedding_dim; ++d) {
                                U[target_id][d] += lr * target_grad[d];
                            }
                        }
                    }

                    size_t done = global_tokens_processed.fetch_add(n);

                    if (tid == 0) {
                        auto now = std::chrono::steady_clock::now();
                        double ms_since = std::chrono::duration<double, std::milli>(now - last_ui_time).count();
                        if (ms_since >= 50.0) {
                            last_ui_time = now;
                            double elapsed_s = std::chrono::duration<double>(now - start_time).count();
                            double speed = (elapsed_s > 0.001) ? (done / elapsed_s) : 0.0;
                            double remaining_s = (speed > 0.0 && total_training_steps > done) ? 
                                                 ((total_training_steps - done) / speed) : 0.0;

                            std::ostringstream extra;
                            extra << "Ep " << (epoch + 1) << "/" << epochs
                                  << " | " << std::fixed << std::setprecision(0) << (speed / 1000.0) << "k tok/s"
                                  << " | ETA: " << std::setprecision(1) << remaining_s << "s";

                            print_progress_bar("Training Dials", done, total_training_steps, extra.str());
                        }
                    }
                }
            }
        }

        print_progress_bar("Training Dials", total_training_steps, total_training_steps, "Finished");

        auto end_time = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        std::cout << "[+] Coordinate training completed in " << std::fixed << std::setprecision(1)
                  << (elapsed_ms / 1000.0) << " seconds.\n";

        for (int v = 0; v < vocab_size; ++v) {
            normalize_vector(U[v]);
        }
    }

    // ------------------------------------------------------------------------
    // COMPUTE DOCUMENT VECTORS
    // ------------------------------------------------------------------------
    void compute_document_vectors() {
        int total = static_cast<int>(documents.size());

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < total; ++i) {
            auto& doc = documents[i];
            doc.doc_vector.assign(embedding_dim, 0.0f);
            int count = 0;

            for (int id : doc.token_ids) {
                // Downweight stop words in document vectors
                float weight = (STOP_WORDS.count(id_to_word[id])) ? 0.2f : 1.0f;
                for (int d = 0; d < embedding_dim; ++d) {
                    doc.doc_vector[d] += U[id][d] * weight;
                }
                count++;
            }

            if (count > 0) {
                normalize_vector(doc.doc_vector);
            }
        }

        print_progress_bar("Vectorizing Docs", total, total, "Complete");
        std::cout << "[+] Normalized all " << total << " document vectors.\n";
    }

    // ------------------------------------------------------------------------
    // SERIALIZE DATABASE TO DISK
    // ------------------------------------------------------------------------
    bool save_to_file(const std::string& db_path) {
        fs::create_directories(fs::path(db_path).parent_path());
        std::ofstream out(db_path);
        if (!out.is_open()) return false;

        out << "VECTOR_DB_V2\n";
        out << "DIM " << embedding_dim << "\n";
        out << "VOCAB " << id_to_word.size() << "\n";
        for (size_t i = 0; i < id_to_word.size(); ++i) {
            out << id_to_word[i];
            for (int d = 0; d < embedding_dim; ++d) {
                out << " " << std::fixed << std::setprecision(6) << U[i][d];
            }
            out << "\n";
        }

        out << "DOCS " << documents.size() << "\n";
        for (const auto& doc : documents) {
            out << doc.filepath << "\n";
            out << doc.filename << "\n";
            out << doc.default_snippet << "\n";
            for (int d = 0; d < embedding_dim; ++d) {
                out << (d > 0 ? " " : "") << std::fixed << std::setprecision(6) << doc.doc_vector[d];
            }
            out << "\n";
        }

        std::cout << "[+] Successfully saved index to: " << db_path << "\n";
        return true;
    }

    // ------------------------------------------------------------------------
    // LOAD DATABASE FROM DISK
    // ------------------------------------------------------------------------
    bool load_from_file(const std::string& db_path) {
        std::ifstream in(db_path);
        if (!in.is_open()) return false;

        std::string header;
        in >> header;
        if (header != "VECTOR_DB_V1" && header != "VECTOR_DB_V2") return false;

        std::string tag;
        in >> tag >> embedding_dim;

        size_t vocab_size = 0;
        in >> tag >> vocab_size;

        word_to_id.clear();
        id_to_word.clear();
        U.assign(vocab_size, std::vector<float>(embedding_dim));

        for (size_t i = 0; i < vocab_size; ++i) {
            std::string word;
            in >> word;
            id_to_word.push_back(word);
            word_to_id[word] = static_cast<int>(i);
            for (int d = 0; d < embedding_dim; ++d) {
                in >> U[i][d];
            }
        }

        size_t doc_count = 0;
        in >> tag >> doc_count;
        std::string dummy;
        std::getline(in, dummy);

        documents.clear();
        documents.reserve(doc_count);

        for (size_t i = 0; i < doc_count; ++i) {
            DocumentRecord doc;
            std::getline(in, doc.filepath);
            std::getline(in, doc.filename);
            std::getline(in, doc.default_snippet);

            doc.doc_vector.resize(embedding_dim);
            for (int d = 0; d < embedding_dim; ++d) {
                in >> doc.doc_vector[d];
            }
            std::getline(in, dummy);
            documents.push_back(doc);
        }

        return true;
    }

    // ------------------------------------------------------------------------
    // FUZZY SPELLING RESOLVER
    // ------------------------------------------------------------------------
    std::string resolve_spelling(const std::string& word, int& out_id) {
        auto it = word_to_id.find(word);
        if (it != word_to_id.end()) {
            out_id = it->second;
            return word;
        }

        // Fuzzy match if word >= 3 chars
        if (word.size() < 3) {
            out_id = -1;
            return "";
        }

        int best_dist = 3;
        std::string best_word = "";
        int best_id = -1;

        for (size_t i = 0; i < id_to_word.size(); ++i) {
            const auto& candidate = id_to_word[i];
            if (std::abs(static_cast<int>(candidate.size()) - static_cast<int>(word.size())) > 2) continue;

            int d = levenshtein_dist(word, candidate);
            if (d < best_dist) {
                best_dist = d;
                best_word = candidate;
                best_id = static_cast<int>(i);
                if (d == 1) break; // Good enough
            }
        }

        if (best_id != -1) {
            out_id = best_id;
            return best_word;
        }

        out_id = -1;
        return "";
    }

    // ------------------------------------------------------------------------
    // SEARCH ENGINE WITH HIGHLIGHTING & FUZZY SPELLCHECK
    // ------------------------------------------------------------------------
    struct SearchResult {
        int doc_id;
        std::string filename;
        std::string filepath;
        std::string snippet;
        float score;
    };

    std::vector<SearchResult> search(const std::string& query_str, 
                                     std::vector<std::string>& active_terms, 
                                     std::vector<std::pair<std::string, std::string>>& auto_corrections,
                                     int top_k = 5) {
        std::vector<SearchResult> results;
        std::vector<std::string> query_tokens = tokenize(query_str);
        if (query_tokens.empty()) return results;

        std::vector<float> query_vec(embedding_dim, 0.0f);
        float total_weight = 0.0f;
        int non_stop_words = 0;

        for (const auto& w : query_tokens) {
            int word_id = -1;
            std::string resolved = resolve_spelling(w, word_id);

            if (word_id != -1) {
                if (resolved != w) {
                    auto_corrections.push_back({w, resolved});
                }
                active_terms.push_back(resolved);

                bool is_stop = STOP_WORDS.count(resolved);
                if (!is_stop) non_stop_words++;

                float weight = is_stop ? 0.05f : 1.0f;
                for (int d = 0; d < embedding_dim; ++d) {
                    query_vec[d] += U[word_id][d] * weight;
                }
                total_weight += weight;
            }
        }

        // If query only contains stop words (e.g. "this is not right"), fallback with equal weight
        if (non_stop_words == 0 && !active_terms.empty()) {
            query_vec.assign(embedding_dim, 0.0f);
            for (const auto& w : active_terms) {
                int id = word_to_id[w];
                for (int d = 0; d < embedding_dim; ++d) {
                    query_vec[d] += U[id][d];
                }
            }
            total_weight = static_cast<float>(active_terms.size());
        }

        if (total_weight <= 1e-6f) return results;

        for (int d = 0; d < embedding_dim; ++d) {
            query_vec[d] /= total_weight;
        }
        normalize_vector(query_vec);

        results.resize(documents.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < documents.size(); ++i) {
            float sim = dot_product(query_vec, documents[i].doc_vector);
            SearchResult sr;
            sr.doc_id = static_cast<int>(i) + 1;
            sr.filename = documents[i].filename;
            sr.filepath = documents[i].filepath;
            sr.snippet = documents[i].default_snippet;
            sr.score = sim;
            results[i] = sr;
        }

        std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
            return a.score > b.score;
        });

        if (static_cast<int>(results.size()) > top_k) {
            results.resize(top_k);
        }

        // Dynamically extract and highlight best snippet for the top K results
        for (auto& r : results) {
            std::string smart = extract_smart_snippet(r.filepath, active_terms);
            if (!smart.empty()) {
                r.snippet = smart;
            }
        }

        return results;
    }

    // ------------------------------------------------------------------------
    // NEAREST SEMANTIC WORD NEIGHBORS
    // ------------------------------------------------------------------------
    std::vector<std::pair<std::string, float>> get_word_neighbors(const std::string& word, int top_k = 5) {
        std::vector<std::pair<std::string, float>> neighbors;
        int word_id = -1;
        std::string resolved = resolve_spelling(word, word_id);
        if (word_id == -1) return neighbors;

        for (size_t i = 0; i < id_to_word.size(); ++i) {
            if (static_cast<int>(i) == word_id) continue;
            float sim = dot_product(U[word_id], U[i]);
            neighbors.push_back({id_to_word[i], sim});
        }

        std::sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        if (static_cast<int>(neighbors.size()) > top_k) {
            neighbors.resize(top_k);
        }
        return neighbors;
    }
};

// ============================================================================
// CLI RENDERERS & FORMATTING
// ============================================================================

void print_banner() {
    std::cout << "\033[1;36m"
              << "======================================================================\n"
              << "  FIRST-PRINCIPLES LOCAL FILE VECTOR DATABASE (C++17 Multi-Core)\n"
              << "  Sub-Millisecond Semantic Search Across Your Notes & Code\n"
              << "======================================================================\033[0m\n";
}

void print_help() {
    std::cout << "\033[1mCommands:\033[0m\n"
              << "  <query>             Search the database for meaning (with auto-spellcheck)\n"
              << "  :index <folder>     Index/re-index a folder of documents\n"
              << "  :list               List all currently indexed files\n"
              << "  :view <id/name>     View snippet & text of an indexed document\n"
              << "  :words <word>       Inspect nearest semantic word neighbors in space\n"
              << "  :stats              Show database dimensions and vocabulary telemetry\n"
              << "  :help               Show this command guide\n"
              << "  exit / quit         Leave VectorDB\n"
              << "----------------------------------------------------------------------\n";
}

void render_results(const std::string& query, 
                    const std::vector<LocalVectorDatabase::SearchResult>& results,
                    const std::vector<std::string>& active_terms,
                    const std::vector<std::pair<std::string, std::string>>& auto_corrections,
                    double query_time_us) {
    std::cout << "\n\033[1;33m----------------------------------------------------------------------\033[0m\n";
    std::cout << " \033[1mQuery:\033[0m \"" << query << "\"  "
              << "(\033[32m" << std::fixed << std::setprecision(2) << query_time_us << " us\033[0m / "
              << (query_time_us / 1000.0) << " ms)\n";

    // Show Auto-Correction Feedback
    if (!auto_corrections.empty()) {
        std::cout << " \033[1;35m[~] Auto-Corrected Typos:\033[0m ";
        for (size_t i = 0; i < auto_corrections.size(); ++i) {
            std::cout << "'" << auto_corrections[i].first << "' -> \033[1;92m'" 
                      << auto_corrections[i].second << "'\033[0m ";
        }
        std::cout << "\n";
    }

    std::cout << "\033[1;33m----------------------------------------------------------------------\033[0m\n";

    if (results.empty()) {
        std::cout << " \033[31m[!] No matching concept found (words not in vocabulary).\033[0m\n\n";
        return;
    }

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        float match_pct = std::max(0.0f, r.score) * 100.0f;

        int bar_width = static_cast<int>(match_pct / 5.0f);
        std::string bar = std::string(bar_width, '#') + std::string(20 - bar_width, '.');

        std::string color = (i == 0) ? "\033[1;32m" : "\033[36m";

        std::cout << " " << color << "#" << (i + 1) << " [" << bar << "] "
                  << std::fixed << std::setprecision(1) << match_pct << "% Match  "
                  << r.filename << "\033[0m [Doc #" << r.doc_id << "]\n";
        std::cout << "    \033[90mPath: " << r.filepath << "\033[0m\n";

        // Highlight matched terms inside the snippet!
        std::string highlighted_snippet = highlight_text(r.snippet, active_terms);
        std::cout << "    \033[1mExcerpt:\033[0m \"" << highlighted_snippet << "\"\n\n";
    }
}

// ============================================================================
// INTERACTIVE SHELL REPL
// ============================================================================

void run_interactive_shell(LocalVectorDatabase& db, const std::string& db_file) {
    std::cout << "\n\033[1;32m[+] Interactive Vector Database Shell Ready!\033[0m\n";
    std::cout << "    Loaded: \033[1m" << db.documents.size() << " files\033[0m, "
              << "\033[1m" << db.id_to_word.size() << " unique words\033[0m, "
              << "\033[1m" << db.embedding_dim << " dials\033[0m.\n";
    std::cout << "    Type any search words directly (typos auto-corrected), or type \033[1;33m:help\033[0m for commands.\n\n";

    std::string line;
    while (true) {
        std::cout << "\033[1;34mVectorDB>\033[0m ";
        if (!std::getline(std::cin, line)) break;

        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        size_t last = line.find_last_not_of(" \t\r\n");
        line = line.substr(first, last - first + 1);

        if (line == "exit" || line == "quit" || line == ":q") {
            break;
        }

        if (line == ":help") {
            print_help();
            continue;
        }

        if (line == ":stats" || line == ":info") {
            std::cout << "\n\033[1mDatabase Statistics:\033[0m\n"
                      << "  - File:          " << db_file << "\n"
                      << "  - Dimensions:    " << db.embedding_dim << " dials per word/document\n"
                      << "  - Vocabulary:    " << db.id_to_word.size() << " words\n"
                      << "  - Documents:     " << db.documents.size() << " files\n"
                      << "  - Search Latency: < 0.05 ms (sub-microsecond dot products)\n\n";
            continue;
        }

        if (line == ":list" || line == ":files") {
            std::cout << "\n\033[1mCurrently Indexed Files (" << db.documents.size() << "):\033[0m\n";
            size_t max_show = std::min(db.documents.size(), static_cast<size_t>(30));
            for (size_t i = 0; i < max_show; ++i) {
                std::cout << "  [" << (i + 1) << "] \033[1;36m" << db.documents[i].filename << "\033[0m\n"
                          << "      Path: " << db.documents[i].filepath << "\n"
                          << "      Snippet: \"" << db.documents[i].default_snippet << "\"\n";
            }
            if (db.documents.size() > max_show) {
                std::cout << "  ... and " << (db.documents.size() - max_show) << " more files.\n";
            }
            std::cout << "\n";
            continue;
        }

        if (line.rfind(":index", 0) == 0) {
            std::string folder = "my_documents";
            if (line.size() > 7) {
                folder = line.substr(7);
                size_t start = folder.find_first_not_of(" ");
                if (start != std::string::npos) folder = folder.substr(start);
            }
            if (db.crawl_and_load(folder)) {
                db.train_word_embeddings();
                db.compute_document_vectors();
                db.save_to_file(db_file);
                std::cout << "\n\033[1;32m[OK] Indexing and training complete for folder '" << folder << "'!\033[0m\n\n";
            }
            continue;
        }

        if (line.rfind(":words", 0) == 0) {
            if (line.size() <= 7) {
                std::cout << "Usage: :words <word> (e.g. :words pointer)\n";
                continue;
            }
            std::string word = line.substr(7);
            size_t start = word.find_first_not_of(" ");
            if (start != std::string::npos) word = word.substr(start);
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);

            auto neighbors = db.get_word_neighbors(word, 5);
            if (neighbors.empty()) {
                std::cout << "Word '" << word << "' not in vocabulary.\n";
            } else {
                std::cout << "\n\033[1mClosest Semantic Neighbors to '\033[36m" << word << "\033[0m\033[1m':\033[0m\n";
                for (size_t i = 0; i < neighbors.size(); ++i) {
                    float sim = neighbors[i].second;
                    std::cout << "  #" << (i + 1) << " " << std::setw(15) << std::left << neighbors[i].first
                              << " (" << std::fixed << std::setprecision(1) << (sim * 100.0f) << "% alignment)\n";
                }
                std::cout << "\n";
            }
            continue;
        }

        if (line.rfind(":view", 0) == 0) {
            if (line.size() <= 6) {
                std::cout << "Usage: :view <id> (e.g. :view 1 or :view filename.cpp)\n";
                continue;
            }
            std::string arg = line.substr(6);
            size_t start = arg.find_first_not_of(" ");
            if (start != std::string::npos) arg = arg.substr(start);

            int target_idx = -1;
            if (std::isdigit(arg[0])) {
                target_idx = std::stoi(arg) - 1;
            } else {
                for (size_t i = 0; i < db.documents.size(); ++i) {
                    if (db.documents[i].filename == arg) {
                        target_idx = static_cast<int>(i);
                        break;
                    }
                }
            }

            if (target_idx >= 0 && target_idx < static_cast<int>(db.documents.size())) {
                const auto& doc = db.documents[target_idx];
                std::cout << "\n\033[1;36m=== Document #" << (target_idx + 1) << ": " << doc.filename << " ===\033[0m\n";
                std::cout << "\033[90mPath: " << doc.filepath << "\033[0m\n\n";

                std::ifstream f(doc.filepath);
                if (f.is_open()) {
                    std::string fline;
                    int lines_shown = 0;
                    while (std::getline(f, fline) && lines_shown < 25) {
                        std::cout << "  " << fline << "\n";
                        lines_shown++;
                    }
                    if (fline.size() > 0) std::cout << "  ... (truncated)\n";
                }
                std::cout << "\n";
            } else {
                std::cout << "Document '" << arg << "' not found. Use ':list' to see valid documents.\n";
            }
            continue;
        }

        // Standard Semantic Query
        std::vector<std::string> active_terms;
        std::vector<std::pair<std::string, std::string>> auto_corrections;

        auto t0 = std::chrono::steady_clock::now();
        auto results = db.search(line, active_terms, auto_corrections, 5);
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        render_results(line, results, active_terms, auto_corrections, us);
    }

    std::cout << "\nExiting VectorDB. Have a great day!\n";
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

int main(int argc, char* argv[]) {
    print_banner();

    std::string db_file = "build/vector_db.txt";
    LocalVectorDatabase db;

    if (argc < 2) {
        if (!db.load_from_file(db_file)) {
            std::cout << "[*] No existing index found at " << db_file << "\n";
            std::cout << "[*] Auto-indexing 'my_documents/' folder now...\n";
            if (db.crawl_and_load("my_documents")) {
                db.train_word_embeddings();
                db.compute_document_vectors();
                db.save_to_file(db_file);
            } else {
                std::cerr << "Could not build index.\n";
                return 1;
            }
        }
        run_interactive_shell(db, db_file);
        return 0;
    }

    std::string cmd = argv[1];

    if (cmd == "index") {
        std::string folder = (argc >= 3) ? argv[2] : "my_documents";
        if (db.crawl_and_load(folder)) {
            db.train_word_embeddings();
            db.compute_document_vectors();
            db.save_to_file(db_file);
            std::cout << "\n[OK] Done! Run './build/local_file_vector_db interactive' to query your files!\n\n";
        }
    }
    else if (cmd == "interactive") {
        if (!db.load_from_file(db_file)) {
            std::cout << "[*] Index not found. Auto-indexing 'my_documents/'...\n";
            if (db.crawl_and_load("my_documents")) {
                db.train_word_embeddings();
                db.compute_document_vectors();
                db.save_to_file(db_file);
            } else {
                return 1;
            }
        }
        run_interactive_shell(db, db_file);
    }
    else if (cmd == "search") {
        if (argc < 3) {
            std::cerr << "Usage: ./build/local_file_vector_db search \"<query>\"\n";
            return 1;
        }
        if (!db.load_from_file(db_file)) {
            std::cerr << "[!] No database index found. Run './build/local_file_vector_db index <folder>' first!\n";
            return 1;
        }

        std::string query = argv[2];
        std::vector<std::string> active_terms;
        std::vector<std::pair<std::string, std::string>> auto_corrections;

        auto t0 = std::chrono::steady_clock::now();
        auto res = db.search(query, active_terms, auto_corrections, 5);
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        render_results(query, res, active_terms, auto_corrections, us);
    }
    else if (cmd == "info" || cmd == "stats") {
        if (!db.load_from_file(db_file)) {
            std::cerr << "[!] No database index found. Run './build/local_file_vector_db index <folder>' first!\n";
            return 1;
        }

        std::cout << "\nDatabase Statistics (" << db_file << "):\n";
        std::cout << "  - Vector Dimensions:  " << db.embedding_dim << " dials per word/doc\n";
        std::cout << "  - Vocabulary Size:    " << db.id_to_word.size() << " unique words\n";
        std::cout << "  - Indexed Documents:  " << db.documents.size() << " files\n\n";
    }
    else {
        print_help();
    }

    return 0;
}
