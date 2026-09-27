CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
BUILD_DIR = build

.PHONY: all clean help train-26 test-26 train-28

# Default target: show help
help:
	@echo "========================================================================"
	@echo " FIRST-PRINCIPLES AI CODEBASE MAKEFILE"
	@echo "========================================================================"
	@echo " Quick Targets:"
	@echo "   make build-26    - Build Grand Unified Real MNIST Brain (Program 26)"
	@echo "   make train-26    - Run training for Program 26"
	@echo "   make test-26     - Run test runner on 1000 unseen real MNIST digits"
	@echo "   make build-28    - Build Real MNIST CNN (Program 28)"
	@echo "   make train-28    - Run training for Program 28"
	@echo ""
	@echo " Generic Build & Run for ANY file:"
	@echo "   make build/<name_without_cpp>"
	@echo "   Example: make build/01_island_finder"
	@echo "   Example: make run FILE=01_island_finder"
	@echo ""
	@echo " Maintenance:"
	@echo "   make clean       - Remove all compiled binaries in build/"
	@echo "========================================================================"

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Generic pattern rule: Compile ANY .cpp file into build/<binary_name>
$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# Program 26 Targets
build-26: $(BUILD_DIR)/grand_unified_train $(BUILD_DIR)/grand_unified_test

$(BUILD_DIR)/grand_unified_train: 26_grand_unified_cnn_smooth_multiclass_evolution_momentum.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/grand_unified_test: 26_test_grand_unified_brain.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

train-26: $(BUILD_DIR)/grand_unified_train
	./$(BUILD_DIR)/grand_unified_train $(GENS)

test-26: $(BUILD_DIR)/grand_unified_test
	./$(BUILD_DIR)/grand_unified_test

# Program 28 Targets
build-28: $(BUILD_DIR)/mnist_cnn_train

$(BUILD_DIR)/mnist_cnn_train: 28_real_mnist_cnn_brain.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

train-28: $(BUILD_DIR)/mnist_cnn_train
	./$(BUILD_DIR)/mnist_cnn_train

# Program 29 Targets (Text Codebook & Next-Letter Predictor)
build-29: $(BUILD_DIR)/text_codebook_and_next_char

$(BUILD_DIR)/text_codebook_and_next_char: 29_text_codebook_and_next_char.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

run-29: $(BUILD_DIR)/text_codebook_and_next_char
	./$(BUILD_DIR)/text_codebook_and_next_char

# Program 30 Targets (Word Codebook & Trigram Story Generator)
build-30: $(BUILD_DIR)/word_codebook_and_ngram

$(BUILD_DIR)/word_codebook_and_ngram: 30_word_codebook_and_ngram.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

run-30: $(BUILD_DIR)/word_codebook_and_ngram
	./$(BUILD_DIR)/word_codebook_and_ngram

# Program 31 Targets (Word Coordinates / Embeddings)
build-31: $(BUILD_DIR)/word_embeddings

$(BUILD_DIR)/word_embeddings: 31_word_embeddings.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

run-31: $(BUILD_DIR)/word_embeddings
	./$(BUILD_DIR)/word_embeddings

# Program 31 Raylib 3D Visualizer Targets
RAYLIB_INC = -I./external/raylib/src -I./external/sys_deps/usr/include -I./external/sys_deps/usr/include/x86_64-linux-gnu
RAYLIB_LIB = ./external/raylib/src/libraylib.a -L./external/sys_deps/usr/lib/x86_64-linux-gnu -lGL -lm -lpthread -ldl -lrt -lX11

build-31-raylib: $(BUILD_DIR)/word_embeddings_3d_raylib

$(BUILD_DIR)/word_embeddings_3d_raylib: 31_word_embeddings_3d_raylib.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Wno-missing-field-initializers $(RAYLIB_INC) $< $(RAYLIB_LIB) -o $@

run-31-raylib: $(BUILD_DIR)/word_embeddings_3d_raylib
	./$(BUILD_DIR)/word_embeddings_3d_raylib

# Program 31 Semantic Search Engine GUI Targets
build-search: $(BUILD_DIR)/semantic_search_engine_gui

$(BUILD_DIR)/semantic_search_engine_gui: 31_semantic_search_engine_gui.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Wno-missing-field-initializers $(RAYLIB_INC) $< $(RAYLIB_LIB) -o $@

run-search: $(BUILD_DIR)/semantic_search_engine_gui
	./$(BUILD_DIR)/semantic_search_engine_gui

# Program 31B Local File Vector Database CLI Targets
build-31b: $(BUILD_DIR)/local_file_vector_db

$(BUILD_DIR)/local_file_vector_db: 31b_local_file_vector_db.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

run-31b: $(BUILD_DIR)/local_file_vector_db
	./$(BUILD_DIR)/local_file_vector_db demo

# Program 32 Targets (The First Neural Next-Word Predictor)
build-32: $(BUILD_DIR)/neural_next_word_predictor

$(BUILD_DIR)/neural_next_word_predictor: 32_neural_next_word_predictor.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

run-32: $(BUILD_DIR)/neural_next_word_predictor
	./$(BUILD_DIR)/neural_next_word_predictor

# Program 33 Targets (Dynamic-Context RNN & Adam Optimizer)
build-33: $(BUILD_DIR)/rnn_dynamic_context_adam

$(BUILD_DIR)/rnn_dynamic_context_adam: 33_rnn_dynamic_context_adam.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

run-33: $(BUILD_DIR)/rnn_dynamic_context_adam
	./$(BUILD_DIR)/rnn_dynamic_context_adam

# Program 34 Targets (Gated Memory LSTM & Adam Optimizer)
build-34: $(BUILD_DIR)/lstm_gated_memory_adam

$(BUILD_DIR)/lstm_gated_memory_adam: 34_lstm_gated_memory_adam.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

run-34: $(BUILD_DIR)/lstm_gated_memory_adam
	./$(BUILD_DIR)/lstm_gated_memory_adam

# Program 35 Targets (Causal Self-Attention Engine & Adam Optimizer)
build-35: $(BUILD_DIR)/self_attention_engine

$(BUILD_DIR)/self_attention_engine: 35_self_attention_engine.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

run-35: $(BUILD_DIR)/self_attention_engine
	./$(BUILD_DIR)/self_attention_engine

# Program 35 Raylib Visualizer
build-35-raylib: $(BUILD_DIR)/self_attention_visualizer_raylib

$(BUILD_DIR)/self_attention_visualizer_raylib: 35_self_attention_visualizer_raylib.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Wno-missing-field-initializers $(RAYLIB_INC) $< $(RAYLIB_LIB) -o $@

run-35-raylib: $(BUILD_DIR)/self_attention_visualizer_raylib
	./$(BUILD_DIR)/self_attention_visualizer_raylib

# Run ANY binary by specifying FILE=<name_without_cpp>
run: $(BUILD_DIR)/$(FILE)
	./$(BUILD_DIR)/$(FILE)

# Clean all binaries
clean:
	rm -rf $(BUILD_DIR)/*




