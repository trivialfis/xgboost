/*!
 * Copyright (c) by Contributors 2019
 */
#ifndef XGBOOST_JSON_IO_H_
#define XGBOOST_JSON_IO_H_
#include <xgboost/json.h>

#include <string>
#include <cinttypes>
#include <utility>
#include <map>

namespace xgboost {
class JsonReader {
 protected:
  size_t constexpr static kMaxNumLength = 17;

  struct SourceLocation {
    int32_t cl_;      // current line
    int32_t cc_;      // current column
    size_t pos_;  // current position in raw_str_

   public:
    SourceLocation() : cl_(0), cc_(0), pos_(0) {}

    int32_t Line() const { return cl_;  }
    int32_t Col()  const { return cc_;  }
    size_t  Pos()  const { return pos_; }

    SourceLocation& Forward(char c = 0) {
      if (c == '\n') {
        cc_ = 0;
        cl_++;
      } else {
        cc_++;
      }
      pos_++;
      return *this;
    }
  } cursor_;

  StringView raw_str_;
  bool ignore_specialization_;

 protected:
  void SkipSpaces();

  char GetNextChar() {
    if (cursor_.Pos() == raw_str_.size()) {
      return -1;
    }
    char ch = raw_str_[cursor_.Pos()];
    cursor_.Forward();
    return ch;
  }

  char PeekNextChar() {
    if (cursor_.Pos() == raw_str_.size()) {
      return -1;
    }
    char ch = raw_str_[cursor_.Pos()];
    return ch;
  }

  char GetNextNonSpaceChar() {
    SkipSpaces();
    return GetNextChar();
  }

  char GetChar(char c) {
    char result = GetNextNonSpaceChar();
    if (result != c) { Expect(c, result); }
    return result;
  }

  void Error(std::string msg) const;

  // Report expected character
  void Expect(char c, char got) {
    std::string msg = "Expecting: \"";
    msg += c;
    msg += "\", got: \"";
    msg += std::string {got};
    msg += " || ";
    msg += std::string {raw_str_[cursor_.Pos()]} + "\"\n";
    Error(msg);
  }

  virtual Json ParseString();
  virtual Json ParseObject();
  virtual Json ParseArray();
  virtual Json ParseNumber();
  virtual Json ParseBoolean();

  Json Parse();

  void SetStr(StringView const& str) {
    raw_str_ = str;
  }
  void SetCursor(size_t pos) {
    cursor_.pos_ = pos;
  }

 private:
  std::locale original_locale_;
  using Fn = std::function<Json (StringView, size_t*)>;

 public:
  explicit JsonReader(StringView str, bool ignore = false) :
      raw_str_{str},
      ignore_specialization_{ignore} {}

  virtual ~JsonReader() = default;

  Json Load();

  static std::map<std::string, Fn>& getRegistry() {
    static std::map<std::string, Fn> set;
    return set;
  }

  static std::map<std::string, Fn> const& registry(
      std::string const& key, Fn fn) {
    getRegistry()[key] = fn;
    return getRegistry();
  }
};

class JsonWriter {
  static constexpr size_t kIndentSize = 2;

  size_t n_spaces_;
  std::ostream* stream_;
  bool pretty_;

  std::locale original_locale;

 public:
  JsonWriter(std::ostream* stream, bool pretty) : n_spaces_{0}, stream_{stream},
                                                  pretty_{pretty} {
    original_locale = std::locale("");
    stream_->imbue(std::locale("en_US.UTF-8"));
  }
  virtual ~JsonWriter() {
    stream_->imbue(original_locale);
  }

  void NewLine() {
    if (pretty_) {
      *stream_ << u8"\n" << std::string(n_spaces_, ' ');
    }
  }

  void BeginIndent() {
    n_spaces_ += kIndentSize;
  }
  void EndIndent() {
    n_spaces_ -= kIndentSize;
  }

  void Write(std::string str) {
    *stream_ << str;
  }

  void Save(Json json);

  virtual void Visit(JsonArray  const* arr);
  virtual void Visit(JsonObject const* obj);
  virtual void Visit(JsonNumber const* num);
  virtual void Visit(JsonRaw    const* raw);
  virtual void Visit(JsonNull   const* null);
  virtual void Visit(JsonString const* str);
  virtual void Visit(JsonBoolean const* boolean);
};
}      // namespace xgboost

#endif  // XGBOOST_JSON_IO_H_
