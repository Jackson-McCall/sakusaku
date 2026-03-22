#pragma once
#include <string>
#include <vector>
#include "llama.h"
#include "RawData.h"

class LLMManager {
public:
	// model_path: full path to our .gguf file
	// n_gpu_layers: how many layers to offload to GPU
	//               set to 99 to offload everything possible to GPU
	LLMManager(const std::string& model_path, int n_gpu_layers = 99);
	~LLMManager();

	// Analyze a single article and return the LLM's response
	std::string AnalyzeArticle(const RawData& article);

	// Process all articles sequentially
	void AnalyzeAll(const std::vector<RawData>& articles);

private:
	std::string RunInference(const std::string& prompt);
	std::string BuildPrompt(const RawData& article);

	// llama.cpp model and context objects
	// llama_model holds the weights loaded from the .gguf file
	// llama_context holds the state for a single inference session
	// llama_sampler controls how the next token is chosen
	llama_model* model_ = nullptr;
	llama_context* ctx_ = nullptr;
	llama_sampler* sampler_ = nullptr;

	int n_gpu_layers_;
	std::string model_path_;
};