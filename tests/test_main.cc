#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <string>
#include <thread>
#include <vector>

#include "index/hnsw_index.h"
#include "table/buffer_pool.h"
#include "util/crc32c.h"
#include "vecbase/db.h"

namespace {

struct WalHeaderView {
  std::uint32_t magic = 0;
  std::uint32_t page_id = 0;
  std::uint32_t payload_size = 0;
  std::uint32_t checksum = 0;
};

bool Expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

float L2Distance(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const float diff = lhs[i] - rhs[i];
    sum += diff * diff;
  }
  return sum;
}

std::vector<vecbase::VectorId>
BruteForceTopK(const std::unordered_map<vecbase::VectorId, vecbase::Record> &records,
               const std::vector<float> &query, std::size_t top_k) {
  struct Item {
    vecbase::VectorId id = 0;
    float distance = 0.0f;
  };

  std::vector<Item> items;
  items.reserve(records.size());
  for (const auto &[id, record] : records) {
    items.push_back(Item{id, L2Distance(record.embedding, query)});
  }
  std::sort(items.begin(), items.end(), [](const Item &lhs, const Item &rhs) {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    return lhs.id < rhs.id;
  });

  std::vector<vecbase::VectorId> ids;
  for (std::size_t i = 0; i < std::min(top_k, items.size()); ++i) {
    ids.push_back(items[i].id);
  }
  return ids;
}

bool TestCRC32CKnownValue() {
  const char *text = "123456789";
  const auto *bytes = reinterpret_cast<const std::byte *>(text);
  return Expect(vecbase::CRC32C(bytes, 9) == 0xE3069283u,
                "CRC32C should match the known Castagnoli vector");
}

bool TestBufferPoolWalMagicAndChecksum() {
  const std::filesystem::path dir = "/tmp/vecbase_test_wal_magic";
  std::filesystem::remove_all(dir);

  vecbase::BufferPool pool((dir / "index.bin").string(), 256, 8, 2);
  if (!Expect(pool.Open(true).ok(), "buffer pool open should succeed")) {
    return false;
  }

  vecbase::BufferPool::PageHandle handle;
  if (!Expect(pool.NewPage(&handle).ok(), "new page should succeed")) {
    return false;
  }
  std::memset(handle.data, 0x5A, handle.size);
  if (!Expect(pool.UnpinPage(handle, true).ok(),
              "unpin dirty page should succeed")) {
    return false;
  }

  std::ifstream input((dir / "index.bin.wal").string(), std::ios::binary);
  if (!Expect(input.is_open(), "WAL file should exist")) {
    return false;
  }

  WalHeaderView header;
  input.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!Expect(static_cast<bool>(input), "WAL header should be readable")) {
    return false;
  }
  std::vector<std::byte> payload(header.payload_size);
  input.read(reinterpret_cast<char *>(payload.data()),
             static_cast<std::streamsize>(payload.size()));
  if (!Expect(static_cast<bool>(input), "WAL payload should be readable")) {
    return false;
  }

  if (!Expect(header.magic == 0x5642574Cu, "WAL magic should match")) {
    return false;
  }
  return Expect(header.checksum == vecbase::CRC32C(payload),
                "WAL checksum should match page payload");
}

bool TestCheckpointStatePersists() {
  const std::filesystem::path dir = "/tmp/vecbase_test_checkpoint";
  std::filesystem::remove_all(dir);

  {
    vecbase::BufferPool pool((dir / "index.bin").string(), 256, 8, 2);
    if (!Expect(pool.Open(true).ok(), "buffer pool open should succeed")) {
      return false;
    }
    if (!Expect(pool.SetCheckpointState(7, 42).ok(),
                "setting checkpoint state should succeed")) {
      return false;
    }
    if (!Expect(pool.Flush().ok(), "flush should succeed")) {
      return false;
    }
  }

  {
    vecbase::BufferPool pool((dir / "index.bin").string(), 256, 8, 2);
    if (!Expect(pool.Open(true).ok(), "reopen buffer pool should succeed")) {
      return false;
    }
    const auto state = pool.checkpoint_state();
    if (!Expect(state.logical_version == 7,
                "checkpoint logical version should persist")) {
      return false;
    }
    if (!Expect(state.root_page_id == 42,
                "checkpoint root page id should persist")) {
      return false;
    }
  }

  return true;
}

bool TestDbEmptyIndexSearch() {
  const std::filesystem::path dir = "/tmp/vecbase_test_empty_search";
  std::filesystem::remove_all(dir);

  vecbase::DB *raw = nullptr;
  auto status = vecbase::DB::Open(vecbase::Options{}, dir.string(), &raw);
  if (!Expect(status.ok(), "DB open for empty search should succeed")) {
    return false;
  }
  std::unique_ptr<vecbase::DB> db(raw);

  vecbase::IndexOptions opts;
  opts.dimension = 2;
  status = db->CreateIndex({}, "docs", opts);
  if (!Expect(status.ok(), "create index for empty search should succeed")) {
    return false;
  }

  vecbase::SearchOptions search;
  search.index_name = "docs";
  search.top_k = 3;
  std::vector<vecbase::SearchResult> results;
  status = db->Search({}, search, {0.0f, 0.0f}, &results);
  if (!Expect(status.ok(), "empty index search should succeed")) {
    return false;
  }
  return Expect(results.empty(), "empty index search should return no results");
}

bool TestDbInvalidArguments() {
  const std::filesystem::path dir = "/tmp/vecbase_test_invalid_args";
  std::filesystem::remove_all(dir);

  vecbase::DB *raw = nullptr;
  auto status = vecbase::DB::Open(vecbase::Options{}, dir.string(), &raw);
  if (!Expect(status.ok(), "DB open for invalid-args test should succeed")) {
    return false;
  }
  std::unique_ptr<vecbase::DB> db(raw);

  vecbase::IndexOptions opts;
  opts.dimension = 2;
  status = db->CreateIndex({}, "docs", opts);
  if (!Expect(status.ok(), "create index should succeed")) {
    return false;
  }
  status = db->CreateIndex({}, "docs", opts);
  if (!Expect(status.code() == vecbase::Status::Code::kAlreadyExists,
              "duplicate create should report already exists")) {
    return false;
  }

  status = db->Put({}, "docs", vecbase::Record{1, {1.0f}, "bad-dim"});
  if (!Expect(status.code() == vecbase::Status::Code::kInvalidArgument,
              "bad-dimension put should be rejected")) {
    return false;
  }

  vecbase::SearchOptions search;
  search.index_name = "docs";
  std::vector<vecbase::SearchResult> results;
  status = db->Search({}, search, {1.0f}, &results);
  if (!Expect(status.code() == vecbase::Status::Code::kInvalidArgument,
              "bad-dimension query should be rejected")) {
    return false;
  }

  status = db->Delete({}, "docs", 999);
  if (!Expect(status.code() == vecbase::Status::Code::kNotFound,
              "delete missing id should return not found")) {
    return false;
  }

  status = db->DropIndex({}, "missing");
  return Expect(status.code() == vecbase::Status::Code::kNotFound,
                "drop missing index should return not found");
}

bool TestBufferPoolMultiPageAndCheckpointAdvance() {
  const std::filesystem::path dir = "/tmp/vecbase_test_multipage";
  std::filesystem::remove_all(dir);

  vecbase::BufferPool pool((dir / "index.bin").string(), 128, 2, 2);
  if (!Expect(pool.Open(true).ok(), "buffer pool open should succeed")) {
    return false;
  }

  vecbase::BufferPool::PageHandle p0;
  vecbase::BufferPool::PageHandle p1;
  vecbase::BufferPool::PageHandle p2;
  if (!Expect(pool.NewPage(&p0).ok(), "new page 0 should succeed")) {
    return false;
  }
  p0.data[0] = std::byte{1};
  if (!Expect(pool.UnpinPage(p0, true).ok(), "unpin p0 should succeed")) {
    return false;
  }

  if (!Expect(pool.NewPage(&p1).ok(), "new page 1 should succeed")) {
    return false;
  }
  p1.data[0] = std::byte{2};
  if (!Expect(pool.UnpinPage(p1, true).ok(), "unpin p1 should succeed")) {
    return false;
  }

  if (!Expect(pool.NewPage(&p2).ok(), "new page 2 should succeed")) {
    return false;
  }
  p2.data[0] = std::byte{3};
  if (!Expect(pool.UnpinPage(p2, true).ok(), "unpin p2 should succeed")) {
    return false;
  }

  if (!Expect(pool.SetCheckpointState(11, 7).ok(),
              "set checkpoint state should succeed")) {
    return false;
  }
  if (!Expect(pool.Flush().ok(), "flush should succeed")) {
    return false;
  }
  const auto state = pool.checkpoint_state();
  if (!Expect(state.logical_version == 11, "logical version should match")) {
    return false;
  }
  if (!Expect(state.root_page_id == 7, "root page id should match")) {
    return false;
  }
  return Expect(state.page_count == 3, "page count should track allocated pages");
}

bool TestDbRecoverSearch() {
  const std::filesystem::path dir = "/tmp/vecbase_test_db_recover";
  std::filesystem::remove_all(dir);

  {
    std::cout << "  phase1-open\n";
    vecbase::DB *raw = nullptr;
    auto status = vecbase::DB::Open(vecbase::Options{}, dir.string(), &raw);
    if (!Expect(status.ok(), "DB open should succeed")) {
      return false;
    }
    std::unique_ptr<vecbase::DB> db(raw);

    vecbase::IndexOptions opts;
    opts.dimension = 2;
    std::cout << "  phase1-create\n";
    status = db->CreateIndex({}, "docs", opts);
    if (!Expect(status.ok(), "create index should succeed")) {
      return false;
    }

    std::cout << "  phase1-put1\n";
    status =
        db->Put({}, "docs", vecbase::Record{1, {0.0f, 0.0f}, "payload-a"});
    if (!Expect(status.ok(), "put #1 should succeed")) {
      return false;
    }
    std::cout << "  phase1-put2\n";
    status =
        db->Put({}, "docs", vecbase::Record{2, {0.2f, 0.2f}, "payload-b"});
    if (!Expect(status.ok(), "put #2 should succeed")) {
      return false;
    }
  }

  {
    std::cout << "  phase2-open\n";
    vecbase::DB *raw = nullptr;
    auto status = vecbase::DB::Open(vecbase::Options{}, dir.string(), &raw);
    if (!Expect(status.ok(), "DB reopen should succeed")) {
      return false;
    }
    std::unique_ptr<vecbase::DB> db(raw);

    vecbase::SearchOptions search;
    search.index_name = "docs";
    search.top_k = 2;
    search.ef_search = 8;
    search.include_payload = true;

    std::vector<vecbase::SearchResult> results;
    std::cout << "  phase2-search\n";
    status = db->Search({}, search, {0.05f, 0.05f}, &results);
    if (!Expect(status.ok(), "search after reopen should succeed")) {
      return false;
    }
    if (!Expect(!results.empty(), "search after reopen should return results")) {
      return false;
    }
    if (!Expect(results.front().id == 1, "nearest id should match")) {
      return false;
    }
    if (!Expect(results.front().payload == "payload-a",
                "payload should be fetched from dkv")) {
      return false;
    }
  }

  return true;
}

bool TestHnswDirectVectorUpdate() {
  const std::filesystem::path dir = "/tmp/vecbase_test_hnsw_update";
  std::filesystem::remove_all(dir);

  vecbase::IndexOptions opts;
  opts.dimension = 2;
  opts.max_degree = 4;
  opts.ef_construction = 8;

  vecbase::HnswIndex index(opts);
  std::cout << "  hnsw-open\n";
  if (!Expect(index.OpenStorage((dir / "index.hnsw").string()).ok(),
              "open HNSW storage should succeed")) {
    return false;
  }
  std::cout << "  hnsw-put1\n";
  if (!Expect(index.Upsert(vecbase::VectorRecord{1, {0.0f, 0.0f}}).ok(),
              "initial HNSW upsert should succeed")) {
    return false;
  }
  std::cout << "  hnsw-put2\n";
  if (!Expect(index.Upsert(vecbase::VectorRecord{2, {1.0f, 1.0f}}).ok(),
              "second HNSW upsert should succeed")) {
    return false;
  }
  std::cout << "  hnsw-update\n";
  if (!Expect(index.Upsert(vecbase::VectorRecord{1, {2.0f, 2.0f}}).ok(),
              "same-id embedding update should succeed")) {
    return false;
  }

  vecbase::SearchOptions search;
  search.top_k = 1;
  search.ef_search = 8;
  std::vector<vecbase::SearchResult> results;
  std::cout << "  hnsw-search\n";
  if (!Expect(index.Search(search, {2.1f, 2.1f}, &results).ok(),
              "search after direct HNSW update should succeed")) {
    return false;
  }
  return Expect(results.size() == 1 && results[0].id == 1,
                "updated HNSW vector should become nearest result");
}

bool TestDbVectorUpdateAndReopen() {
  const std::filesystem::path dir = "/tmp/vecbase_test_db_update";
  std::filesystem::remove_all(dir);

  {
    vecbase::DB *raw = nullptr;
    auto status = vecbase::DB::Open(vecbase::Options{}, dir.string(), &raw);
    if (!Expect(status.ok(), "DB open for update test should succeed")) {
      return false;
    }
    std::unique_ptr<vecbase::DB> db(raw);

    vecbase::IndexOptions opts;
    opts.dimension = 2;
    status = db->CreateIndex({}, "docs", opts);
    if (!Expect(status.ok(), "create index for update test should succeed")) {
      return false;
    }

    status = db->Put({}, "docs", vecbase::Record{1, {0.0f, 0.0f}, "v1"});
    if (!Expect(status.ok(), "initial DB put should succeed")) {
      return false;
    }
    status = db->Put({}, "docs", vecbase::Record{2, {1.0f, 1.0f}, "v2"});
    if (!Expect(status.ok(), "second DB put should succeed")) {
      return false;
    }
    status = db->Put({}, "docs", vecbase::Record{1, {2.0f, 2.0f}, "v1-new"});
    if (!Expect(status.ok(), "DB same-id vector update should succeed")) {
      return false;
    }
  }

  {
    vecbase::DB *raw = nullptr;
    auto status = vecbase::DB::Open(vecbase::Options{}, dir.string(), &raw);
    if (!Expect(status.ok(), "DB reopen for update test should succeed")) {
      return false;
    }
    std::unique_ptr<vecbase::DB> db(raw);

    vecbase::SearchOptions search;
    search.index_name = "docs";
    search.top_k = 1;
    search.ef_search = 8;
    search.include_payload = true;

    std::vector<vecbase::SearchResult> results;
    status = db->Search({}, search, {2.1f, 2.1f}, &results);
    if (!Expect(status.ok(), "search after DB reopen should succeed")) {
      return false;
    }
    if (!Expect(results.size() == 1 && results[0].id == 1,
                "reopened DB should reflect updated nearest vector")) {
      return false;
    }
    return Expect(results[0].payload == "v1-new",
                  "reopened DB should return updated payload");
  }
}

bool TestBufferPoolPageRoundTrip() {
  const std::filesystem::path dir = "/tmp/vecbase_test_page_roundtrip";
  std::filesystem::remove_all(dir);

  {
    vecbase::BufferPool pool((dir / "index.bin").string(), 256, 8, 2);
    if (!Expect(pool.Open(true).ok(), "buffer pool open should succeed")) {
      return false;
    }
    vecbase::BufferPool::PageHandle handle;
    if (!Expect(pool.NewPage(&handle).ok(), "new page should succeed")) {
      return false;
    }
    for (std::size_t i = 0; i < handle.size; ++i) {
      handle.data[i] = static_cast<std::byte>(i % 251);
    }
    if (!Expect(pool.UnpinPage(handle, true).ok(),
                "dirty unpin should succeed")) {
      return false;
    }
    if (!Expect(pool.Flush().ok(), "flush should succeed")) {
      return false;
    }
  }

  {
    vecbase::BufferPool pool((dir / "index.bin").string(), 256, 8, 2);
    if (!Expect(pool.Open(true).ok(), "reopen buffer pool should succeed")) {
      return false;
    }
    vecbase::BufferPool::PageHandle handle;
    if (!Expect(pool.FetchPage(0, &handle).ok(), "fetch page 0 should succeed")) {
      return false;
    }
    for (std::size_t i = 0; i < handle.size; ++i) {
      if (!Expect(handle.data[i] == static_cast<std::byte>(i % 251),
                  "page content should round-trip through disk")) {
        return false;
      }
    }
    if (!Expect(pool.UnpinPage(handle, false).ok(),
                "clean unpin should succeed")) {
      return false;
    }
  }

  return true;
}

bool TestFuzzDbRandomOps() {
  const std::filesystem::path dir = "/tmp/vecbase_test_fuzz_db";
  std::filesystem::remove_all(dir);

  std::mt19937 rng(123456u);
  std::uniform_real_distribution<float> coord(-5.0f, 5.0f);
  std::uniform_int_distribution<int> id_dist(1, 12);
  std::uniform_int_distribution<int> op_dist(0, 9);

  std::unordered_map<vecbase::VectorId, vecbase::Record> oracle;

  vecbase::DB *raw = nullptr;
  auto status = vecbase::DB::Open(vecbase::Options{}, dir.string(), &raw);
  if (!Expect(status.ok(), "DB open for fuzz test should succeed")) {
    return false;
  }
  std::unique_ptr<vecbase::DB> db(raw);

  vecbase::IndexOptions opts;
  opts.dimension = 2;
  status = db->CreateIndex({}, "docs", opts);
  if (!Expect(status.ok(), "create index for fuzz test should succeed")) {
    return false;
  }

  auto reopen = [&]() -> bool {
    db.reset();
    vecbase::DB *tmp = nullptr;
    auto st = vecbase::DB::Open(vecbase::Options{}, dir.string(), &tmp);
    if (!Expect(st.ok(), "DB reopen in fuzz test should succeed")) {
      return false;
    }
    db.reset(tmp);
    return true;
  };

  for (int step = 0; step < 150; ++step) {
    const int op = op_dist(rng);
    const vecbase::VectorId id = static_cast<vecbase::VectorId>(id_dist(rng));

    if (op <= 4) {
      vecbase::Record record;
      record.id = id;
      record.embedding = {coord(rng), coord(rng)};
      record.payload = "p" + std::to_string(id) + "_" + std::to_string(step);
      status = db->Put({}, "docs", record);
      if (!Expect(status.ok(), "fuzz put should succeed")) {
        return false;
      }
      oracle[id] = record;
    } else if (op <= 6) {
      status = db->Delete({}, "docs", id);
      const bool existed = oracle.erase(id) > 0;
      if (existed) {
        if (!Expect(status.ok(), "fuzz delete existing id should succeed")) {
          return false;
        }
      } else if (!Expect(status.code() == vecbase::Status::Code::kNotFound,
                         "fuzz delete missing id should return not found")) {
        return false;
      }
    } else if (op == 7) {
      if (!reopen()) {
        return false;
      }
    } else {
      const std::vector<float> query = {coord(rng), coord(rng)};
      vecbase::SearchOptions search;
      search.index_name = "docs";
      search.top_k = 3;
      search.ef_search = 16;
      std::vector<vecbase::SearchResult> results;
      status = db->Search({}, search, query, &results);
      if (!Expect(status.ok(), "fuzz search should succeed")) {
        return false;
      }
      const auto expected = BruteForceTopK(oracle, query, 3);
      if (!Expect(results.size() == expected.size(),
                  "fuzz search result size should match oracle")) {
        return false;
      }
      for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!Expect(results[i].id == expected[i],
                    "fuzz nearest ids should match brute-force oracle")) {
          return false;
        }
      }
    }
  }

  return true;
}

} // namespace

int main() {
  const std::array<std::pair<const char *, bool (*)()>, 11> tests = {{
      {"TestCRC32CKnownValue", &TestCRC32CKnownValue},
      {"TestBufferPoolWalMagicAndChecksum", &TestBufferPoolWalMagicAndChecksum},
      {"TestCheckpointStatePersists", &TestCheckpointStatePersists},
      {"TestDbEmptyIndexSearch", &TestDbEmptyIndexSearch},
      {"TestDbInvalidArguments", &TestDbInvalidArguments},
      {"TestBufferPoolMultiPageAndCheckpointAdvance",
       &TestBufferPoolMultiPageAndCheckpointAdvance},
      {"TestDbRecoverSearch", &TestDbRecoverSearch},
      {"TestHnswDirectVectorUpdate", &TestHnswDirectVectorUpdate},
      {"TestDbVectorUpdateAndReopen", &TestDbVectorUpdateAndReopen},
      {"TestBufferPoolPageRoundTrip", &TestBufferPoolPageRoundTrip},
      {"TestFuzzDbRandomOps", &TestFuzzDbRandomOps},
  }};

  for (const auto &[name, test] : tests) {
    std::cout << "Running " << name << std::endl;
    if (!test()) {
      return 1;
    }
  };

  std::cout << "All tests passed\n";
  return 0;
}
