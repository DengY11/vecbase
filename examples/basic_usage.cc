#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "vecbase.hpp"

namespace {

void CheckStatus(const vecbase::Status &status, const std::string &step) {
  if (!status.ok()) {
    std::cerr << step << " failed: " << status.ToString() << "\n";
    std::exit(1);
  }
}

} // namespace

int main() {
  const std::filesystem::path db_path = "examples/demo_db";
  std::error_code ec;
  std::filesystem::remove_all(db_path, ec);

  vecbase::Options db_options;
  vecbase::DB *raw_db = nullptr;
  CheckStatus(vecbase::DB::Open(db_options, db_path.string(), &raw_db), "open db");
  std::unique_ptr<vecbase::DB> db(raw_db);

  vecbase::IndexOptions index_options;
  index_options.dimension = 3;
  index_options.metric = vecbase::MetricType::kCosine;
  index_options.max_degree = 8;
  index_options.ef_construction = 32;

  CheckStatus(db->CreateIndex({}, "docs", index_options), "create index");

  CheckStatus(db->Put({}, "docs",
                      vecbase::Record{1, {0.95f, 0.05f, 0.00f},
                                      "vector database internals"}),
              "insert record 1");
  CheckStatus(db->Put({}, "docs",
                      vecbase::Record{2, {0.90f, 0.10f, 0.00f},
                                      "approximate nearest neighbor graph"}),
              "insert record 2");
  CheckStatus(db->Put({}, "docs",
                      vecbase::Record{3, {0.05f, 0.90f, 0.05f},
                                      "write-ahead logging and recovery"}),
              "insert record 3");

  vecbase::SearchOptions search_options;
  search_options.index_name = "docs";
  search_options.top_k = 2;
  search_options.ef_search = 16;
  search_options.include_payload = true;

  std::vector<vecbase::SearchResult> results;
  CheckStatus(db->Search({}, search_options, {1.0f, 0.0f, 0.0f}, &results),
              "search");

  std::cout << "Top results for query [1, 0, 0]\n";
  for (const vecbase::SearchResult &result : results) {
    std::cout << "  id=" << result.id << " score=" << result.score
              << " payload=\"" << result.payload << "\"\n";
  }

  vecbase::IndexStats stats;
  CheckStatus(db->GetIndexStats("docs", &stats), "get stats");
  std::cout << "\nIndex stats\n";
  std::cout << "  dimension=" << stats.dimension << "\n";
  std::cout << "  size=" << stats.size << "\n";
  std::cout << "  graph_edges=" << stats.graph_edges << "\n";

  return 0;
}
