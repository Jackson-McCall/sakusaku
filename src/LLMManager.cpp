#include "LLMManager.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

LLMManager::LLMManager(const std::string& model_path, int n_gpu_layers)
	: model_path_(model_path), n_gpu_layers_(n_gpu_layers) {

	// Must be called once before any llama.cpp functions.
	// Initializes CUDA, memory allocators, and internal state.
	llama_backend_init();

	// Model params control how the weights are loaded.
	llama_model_params model_params = llama_model_default_params();

	// How many transformer layers to put on the GPU.
	// 99 means "as many as will fit" — llama.cpp caps it at the actual layer count.
	model_params.n_gpu_layers = n_gpu_layers_;

	// Load the .gguf file into memory/VRAM.
	// This is the slow step — can take several seconds for large models.
	std::cout << "Beginning to load model." << std::endl;
	model_ = llama_load_model_from_file(model_path_.c_str(), model_params);
	if (!model_) {
		std::cerr << "Failed to load model from: " << model_path_ << std::endl;
		return;
	}
	std::cout << "Model loaded successfully." << std::endl;

	// Context params control inference behavior.
	llama_context_params ctx_params = llama_context_default_params();

	// n_ctx: how many tokens the model can see at once (context window).
	// The KV cache scales linearly with n_ctx — at 16384 it'll use ~512MB instead of 128MB, still well within our headroom.
	// If that's still not enough for some articles we can try to push to 32768
	ctx_params.n_ctx = 16384;

	// n_batch: how many tokens to process in parallel during prompt ingestion.
	// Higher = faster prompt processing but more VRAM.
	ctx_params.n_batch = 16384;

	// Create the inference context from the loaded model.
	ctx_ = llama_new_context_with_model(model_, ctx_params);
	if (!ctx_) {
		std::cerr << "Failed to create inference context." << std::endl;
		return;
	}

	// Set up the sampler chain — controls how the next token is picked
	// from the probability distribution the model outputs.
	llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
	sampler_ = llama_sampler_chain_init(sampler_params);

	llama_sampler_chain_add(sampler_, llama_sampler_init_penalties(
		128,   // look back further
		1.3f,  // more aggressive repeat penalty
		0.1f,  // small frequency penalty also helps
		0.0f
	));

	// Temperature controls randomness.
	// 0.7 = good balance between coherent and varied output.
	// Lower (0.1) = more deterministic. Higher (1.0+) = more creative.
	llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.1f));

	// Greedy sampling picks the highest probability token after temperature.
	llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());

	std::cout << "LLMManager ready. GPU layers: " << n_gpu_layers_ << std::endl;
}

LLMManager::~LLMManager() {
	// Free in reverse order of creation
	if (sampler_) llama_sampler_free(sampler_);
	if (ctx_)     llama_free(ctx_);
	if (model_)   llama_model_free(model_);
	llama_backend_free();
}

std::string LLMManager::BuildPrompt(const RawData& article) {
	return "You are a financial editor. Strip fluff, preserve facts.\n\n"
		"Respond ONLY with the following three lines. No preamble, no explanation, nothing else:\n\n"
		"[Article Title] : " + article.title + "\n"
		"[Stocks Involved] : <comma delimited ticker symbols only, or NONE if no tickers mentioned>\n"
		"[No-Fluff Article] : <the article rewritten with zero fluff, every financial fact preserved, "
		"maximum 3 paragraphs, no commentary>\n\n"
		"Article:\n" + article.body;
}

std::string LLMManager::RunInference(const std::string& prompt) {
	// Tokenize — convert the text string into integer token IDs the model understands.
	// We allocate a buffer large enough for the full context window.
	const int max_tokens = 16384;
	std::vector<llama_token> tokens(max_tokens);

    // llama_tokenize converts text to token IDs.
    // Arguments:
    //   model    — needed for the vocabulary
    //   text     — input string
    //   length   — input length
    //   buffer   — output token array
    //   max      — max tokens to produce
    //   add_bos  — add beginning-of-sequence token
    //   special  — parse special tokens like [INST]
    int n_tokens = llama_tokenize(
		llama_model_get_vocab(llama_get_model(ctx_)),
		prompt.c_str(),
		prompt.size(),
		tokens.data(),
		max_tokens,
		true,
		true
	);

	if (n_tokens < 0) {
		std::cerr << "Tokenization failed." << std::endl;
		return "";
	}

	tokens.resize(n_tokens);

	// llama_batch_get_one wraps our token array into a single batch
	// for processing the entire prompt at once.
	llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

	// Process the prompt — runs it through the model to build up the
	// KV cache (key-value cache that stores attention state).
	if (llama_decode(ctx_, batch) != 0) {
		std::cerr << "Failed to process prompt." << std::endl;
		return "";
	}

	// Generation loop — generate one token at a time until end-of-sequence
	// or we hit the max output length.
	std::string output;
	const int max_new_tokens = 4096;

	for (int i = 0; i < max_new_tokens; i++) {
		// Sample the next token using our sampler chain
		llama_token new_token = llama_sampler_sample(sampler_, ctx_, -1);

		// End-of-generation token — model is done
		if (llama_token_is_eog(llama_model_get_vocab(llama_get_model(ctx_)), new_token)) {
			break;
		}

		// Convert token ID back to text and append to output
		char buf[256];
		int n = llama_token_to_piece(
			llama_model_get_vocab(llama_get_model(ctx_)),
			new_token,
			buf,
			sizeof(buf),
			0,
			true
		);
		if (n > 0) {
			output.append(buf, n);
		}

		// Feed the new token back so the model can attend to what it just generated
		llama_batch next_batch = llama_batch_get_one(&new_token, 1);
		if (llama_decode(ctx_, next_batch) != 0) {
			std::cerr << "Decode failed during generation." << std::endl;
			break;
		}
	}

    // Clear the KV cache so the next call starts fresh with no bleed
    // from this conversation.
	llama_memory_clear(llama_get_memory(ctx_), true);

	return output;
}

std::string LLMManager::AnalyzeArticle(const RawData& article) {
	std::cout << "\nAnalyzing: " << article.title << std::endl;

	std::string prompt = BuildPrompt(article);
	std::string response = RunInference(prompt);

	std::cout << "Response:\n" << response << std::endl;
	return response;
}

void LLMManager::AnalyzeAll(const std::vector<RawData>& articles) {
	std::cout << "Analyzing " << articles.size() << " articles." << std::endl;

	// =========================================================================
	// Output File Setup
	// =========================================================================
	// Open output.txt in the current working directory.
	// std::ios::trunc clears the file if it already exists from a previous run.
	std::ofstream output_file("output.txt", std::ios::trunc);
	if (!output_file.is_open()) {
		std::cerr << "Warning: could not open output.txt for writing." << std::endl;
	}
	// =========================================================================

	for (int i = 0; i < (int)articles.size(); i++) {
		std::cout << "\n[" << i + 1 << "/" << articles.size() << "]" << std::endl;
		std::string response = AnalyzeArticle(articles[i]);

		// =====================================================================
		// Write to Output File
		// =====================================================================
		if (output_file.is_open()) {
			output_file << "=== [" << i + 1 << "/" << articles.size() << "] ===\n";
			output_file << "Title: " << articles[i].title << "\n";
			output_file << "URL: " << articles[i].url << "\n\n";
			output_file << response << "\n\n";
			output_file << "=============================================================\n\n";
			output_file.flush(); // write immediately so if it crashes midway you don't lose data
		}
		// =====================================================================
	}

	// =========================================================================
	// Close Output File
	// =========================================================================
	if (output_file.is_open()) {
		output_file.close();
		std::cout << "\nResults saved to output.txt" << std::endl;
	}
	// =========================================================================

	std::cout << "\nAll articles analyzed." << std::endl;
}