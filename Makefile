CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -pthread
DEBUG_FLAGS = -g -O0 -DDEBUG

SRC_DIR = src
BUILD_DIR = build
SRCS = $(SRC_DIR)/main.cpp \
       $(SRC_DIR)/Graph.cpp \
       $(SRC_DIR)/ThreadPool.cpp \
       $(SRC_DIR)/CommunityDetector.cpp \
       $(SRC_DIR)/TwoPhaseQuasiCliqueSolver.cpp
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/two_phase_solver

.PHONY: all clean debug

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: all

clean:
	rm -rf $(BUILD_DIR)