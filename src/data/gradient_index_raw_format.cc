/*!
 * Copyright (c) 2021 by Contributors
 */
#include <xgboost/data.h>
#include <dmlc/registry.h>

#include "xgboost/logging.h"
#include "sparse_page_writer.h"
#include "gradient_index.h"

namespace xgboost {
namespace data {
class GHistIndexPageFormat : public SparsePageFormat<GHistIndexMatrix> {
 public:
  bool Read(GHistIndexMatrix* page, dmlc::SeekStream* fi) override {
    // read cuts
    fi->Read(&page->cut.cut_values_.HostVector());
    fi->Read(&page->cut.cut_ptrs_.HostVector());
    fi->Read(&page->cut.min_vals_.HostVector());

    // read index
    std::vector<uint8_t> data;
    fi->Read(&data);
    page->index.Resize(data.size());
    std::copy(data.cbegin(), data.cend(), page->index.begin());

    std::vector<uint32_t> offset;
    fi->Read(&offset);

    common::BinTypeSize type_size;
    fi->Read(&type_size);
    page->index.SetBinTypeSize(type_size);

    // others
    fi->Read(&page->row_ptr);
    fi->Read(&page->hit_count);
    if (!fi->Read(&page->max_num_bins)) {
      return false;
    }
    return true;
  }

  size_t Write(GHistIndexMatrix const &page, dmlc::Stream *fo) override {
    size_t bytes = 0;
    auto const &cuts = page.cut;
    // write cuts
    fo->Write(cuts.Values());
    bytes += cuts.cut_values_.ConstHostSpan().size_bytes() + sizeof(uint64_t);
    fo->Write(cuts.Ptrs());
    bytes += cuts.cut_ptrs_.ConstHostSpan().size_bytes() + sizeof(uint64_t);
    fo->Write(cuts.MinValues());
    bytes += cuts.min_vals_.ConstHostSpan().size_bytes() + sizeof(uint64_t);

    // write index
    fo->Write(page.index.RawData());
    bytes += page.index.RawData().size() *
                 sizeof(std::remove_reference_t<
                        decltype(page.index.RawData())>::value_type) +
             sizeof(uint64_t);
    fo->Write(page.index.RawOffset());
    bytes += page.index.RawData().size() *
                 sizeof(std::remove_reference_t<
                        decltype(page.index.RawOffset())>::value_type) +
             sizeof(uint64_t);
    fo->Write(page.index.GetBinTypeSize());
    bytes += sizeof(page.index.OffsetSize());

    // others
    fo->Write(page.row_ptr);
    bytes += page.row_ptr.size() * sizeof(decltype(page.row_ptr)::value_type) +
             sizeof(uint64_t);
    fo->Write(page.hit_count);
    bytes +=
        page.row_ptr.size() * sizeof(decltype(page.hit_count)::value_type) +
        sizeof(uint64_t);
    fo->Write(page.max_num_bins);
    bytes += sizeof(page.max_num_bins);

    return bytes;
  }
};

XGBOOST_REGISTER_GHIST_INDEX_PAGE_FORMAT(raw)
    .describe("Raw GHistIndex binary data format.")
    .set_body([]() {
      return new GHistIndexPageFormat();
    });

}  // namespace data
}  // namespace xgboost
