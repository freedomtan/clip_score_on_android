#include <fstream>
#include <iostream>

#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature_util.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/types.h"

// CaptionRecord is equivlent to records in the ground truth tfrecord file.
struct CaptionRecord {
  explicit CaptionRecord(const tensorflow::tstring& record) {
    using string = std::string;
    tensorflow::Example example;
    example.ParseFromString(record);

    id = tensorflow::GetFeatureValues<int64_t>("id", example)[0];
    // std::cout << "id: " << id << "\n";

    auto caption_list = tensorflow::GetFeatureValues<string>("caption", example);
    caption = std::vector<std::string>(caption_list.begin(), caption_list.end());

    auto tokenized_id_list = tensorflow::GetFeatureValues<int64_t>("tokenized_ids", example);
    tokenized_ids = std::vector<int64_t>(tokenized_id_list.begin(), tokenized_id_list.end());

    auto filename_list = tensorflow::GetFeatureValues<string>("file_name", example);
    filename = std::vector<std::string>(filename_list.begin(), filename_list.end());

    auto clip_score_list = tensorflow::GetFeatureValues<float>("clip_score", example);
    clip_score = std::vector<float>(clip_score_list.begin(), clip_score_list.end())[0];
  }

  void dump() {
    std::cout << "  id: " << id << "\n";
    std::cout << "  caption: " << caption[0] << "\n";
    std::cout << "  token_id: ";
    for (auto t: tokenized_ids) {
	    std::cout << t << ", ";
    }
    std::cout << "\n";
    std::cout << "  file_name: " << filename[0] << "\n";
    std::cout << "  clip_score: " << clip_score << "\n";
  }

  int64_t id;
  std::vector<std::string> caption;
  std::vector<int64_t> tokenized_ids;
  std::vector<std::string> filename;
  float clip_score;
};


int main(int argc, char* argv[]) {
  std::unique_ptr<tensorflow::RandomAccessFile> random_access_file;
  std::unique_ptr<tensorflow::io::RecordReader> reader;
  std::unordered_map<uint32_t, tensorflow::uint64> index_to_offset;

  const std::string& filename = "val_toke_ids_and_clip_scores.tfrecord";

  auto ret = tensorflow::Env::Default()->NewRandomAccessFile(
      filename, &random_access_file);
  std::cout << "ret: " << ret << "\n";

  auto options =
      tensorflow::io::RecordReaderOptions::CreateRecordReaderOptions("ZLIB");
  reader.reset(
      new tensorflow::io::RecordReader(random_access_file.get(), options));

  tensorflow::uint64 id = 0, offset = 0, noffset = 0;
  tensorflow::tstring record;

  while (reader->ReadRecord(&noffset, &record).ok()) {
    index_to_offset[id++] = offset;
    offset = noffset;
    CaptionRecord(record).dump();
  }
}
