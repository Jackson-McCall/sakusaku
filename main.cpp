// ============================================================================
// SakuSaku
// This program spins up an LLM to consolidate the data from the raw database
// and then places it into the consolidated database
// ============================================================================
#include <string>
#include "src/DatabaseManager.h"
#include "src/LLMManager.h"

int main(int argc, char* argv[]) {

	// Pull data from database
	std::string raw_db_path_ = "C:\\Users\\jacks\\source\\repos\\wakuwaku\\articles.db";
	std::string condensed_db_path_ = "C:\\Users\\jacks\\source\\repos\\sakusaku\\condensed_articles.db";

	DatabaseManager rawDbManager(raw_db_path_, condensed_db_path_);

	std::vector<RawData> rawArticles = rawDbManager.RetrieveRawData();
	// Spawn LLM
	LLMManager llm("C:/Users/jacks/OneDrive/Documents/CompSci/ohyeahwooyeah/LocalLLMs/GGUF_Model_Files/llama-3.2-1b-instruct-q8_0.gguf", 99);
	// Pass data one by one to LLM
	llm.AnalyzeAll(rawArticles);
	// Save output data to new database

	return 0;
}