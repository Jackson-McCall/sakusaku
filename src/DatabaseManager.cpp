#include "DatabaseManager.h"
#include <vector>
#include "../sqlite3.h"
#include <iostream>

DatabaseManager::DatabaseManager(std::string rawDbPath, std::string condensedDbPath) {
	db_raw_path_ = rawDbPath;
	db_condensed_path_ = condensedDbPath;
}


std::vector<RawData> DatabaseManager::RetrieveRawData() {
	std::cout << "retrieve raw data reached" << std::endl;
	sqlite3* dbRaw;  // sqlite3 database objects

	sqlite3_open(db_raw_path_.c_str(), &dbRaw); // open connection to raw db
	const char* sql = "SELECT url, title, body, source FROM articles;";

	sqlite3_stmt* stmt;
	sqlite3_prepare_v2(dbRaw, sql, -1, &stmt, nullptr);

	std::cout << "Line 22" << std::endl;
	// sqlite3_step returns SQLITE_ROW each time there is another row to read
	// When there are no more rows it returns SQLITE_DONE and the loop ends
	int i = 0;
	while (sqlite3_step(stmt) == SQLITE_ROW) {
		// read each column by index
		std::cout << i << std::endl;
		i++;
		rawDataVector.push_back({ (const char*)sqlite3_column_text(stmt, 0),
								  (const char*)sqlite3_column_text(stmt, 1),
								  (const char*)sqlite3_column_text(stmt, 2),
								  (const char*)sqlite3_column_text(stmt, 3), });
	}

	std::cout << "33" << std::endl;
	sqlite3_finalize(stmt);
	sqlite3_close(dbRaw);
	
	// debugging
	/*
	for (int i = 0; i < rawDataVector.size(); i++) {
		std::cout << rawDataVector[i].url << std::endl;
		std::cout << rawDataVector[i].title << std::endl;
		std::cout << rawDataVector[i].body << std::endl;
		std::cout << rawDataVector[i].source << std::endl;
	}
	*/
	return rawDataVector;
	
}

void DatabaseManager::StoreCondensedData() {

}