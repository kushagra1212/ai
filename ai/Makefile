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

# Run ANY binary by specifying FILE=<name_without_cpp>
run: $(BUILD_DIR)/$(FILE)
	./$(BUILD_DIR)/$(FILE)

# Clean all binaries
clean:
	rm -rf $(BUILD_DIR)/*
