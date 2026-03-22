# pragma once
#include "RawData.h"
#include <vector>
#include <string>


class DatabaseManager {
public:
	DatabaseManager(std::string rawDbPath, std::string condensedDbPath);
	std::vector<RawData> RetrieveRawData();
	void StoreCondensedData();

private:
	std::string db_raw_path_;
	std::string db_condensed_path_;
	std::vector<RawData> rawDataVector;
};