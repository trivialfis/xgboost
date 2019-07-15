/*!
 * Copyright (c) by Contributors 2019
 */
#if defined(__unix__)
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // defined(__unix__)
#include <cstdio>

#include <locale>
#include <sstream>

#include "xgboost/logging.h"
#include "xgboost/json.h"
#include "xgboost/json_io.h"
#include "../common/timer.h"

namespace xgboost {

void JsonWriter::Save(Json json) {
  json.ptr_->Save(this);
}

void JsonWriter::Visit(JsonArray const* arr) {
  this->Write("[");
  auto const& vec = arr->getArray();
  size_t size = vec.size();
  for (size_t i = 0; i < size; ++i) {
    auto& value = vec[i];
    this->Save(value);
    if (i != size-1) { Write(", "); }
  }
  this->Write("]");
}

void JsonWriter::Visit(JsonObject const* obj) {
  this->Write("{");
  this->BeginIndent();
  this->NewLine();

  size_t i = 0;
  size_t size = obj->getObject().size();

  for (auto& value : obj->getObject()) {
    this->Write("\"" + value.first + "\": ");
    this->Save(value.second);

    if (i != size-1) {
      this->Write(",");
      this->NewLine();
    }
    i++;
  }
  this->EndIndent();
  this->NewLine();
  this->Write("}");
}

void JsonWriter::Visit(JsonNumber const* num) {
  this->Write(std::to_string(num->getNumber()));
}

void JsonWriter::Visit(JsonRaw const* raw) {
  auto const& str = raw->getRaw();
  this->Write(str);
}

void JsonWriter::Visit(JsonNull const* null) {}

void JsonWriter::Visit(JsonString const* str) {
  std::string buffer;
  buffer += '"';
  auto const& string = str->getString();
  for (size_t i = 0; i < string.length(); i++) {
    const char ch = string[i];
    if (ch == '\\') {
      if (i < string.size() && string[i+1] == 'u') {
        buffer += "\\";
      } else {
        buffer += "\\\\";
      }
    } else if (ch == '"') {
      buffer += "\\\"";
    } else if (ch == '\b') {
      buffer += "\\b";
    } else if (ch == '\f') {
      buffer += "\\f";
    } else if (ch == '\n') {
      buffer += "\\n";
    } else if (ch == '\r') {
      buffer += "\\r";
    } else if (ch == '\t') {
      buffer += "\\t";
    } else if (static_cast<uint8_t>(ch) <= 0x1f) {
      // Unit separator
      char buf[8];
      snprintf(buf, sizeof buf, "\\u%04x", ch);
      buffer += buf;
    } else {
      buffer += ch;
    }
  }
  buffer += '"';
  this->Write(buffer);
}

void JsonWriter::Visit(JsonBoolean const* boolean) {
  bool val = boolean->getBoolean();
  if (val) {
    this->Write(u8"true");
  } else {
    this->Write(u8"false");
  }
}

// Value
std::string Value::TypeStr() const {
  switch (kind_) {
    case ValueKind::String: return "String";  break;
    case ValueKind::Number: return "Number";  break;
    case ValueKind::Object: return "Object";  break;
    case ValueKind::Array:  return "Array";   break;
    case ValueKind::Boolean:return "Boolean"; break;
    case ValueKind::Null:   return "Null";    break;
    case ValueKind::Raw:    return "Raw";     break;
    case ValueKind::Integer:     return "Integer"; break;
  }
  return "";
}

// Only used for keeping old compilers happy about non-reaching return
// statement.
Json& DummyJsonObject() {
  static Json obj;
  return obj;
}

// Json Object
JsonObject::JsonObject(std::map<std::string, Json>&& object)
    : Value(ValueKind::Object), object_{std::move(object)} {}

Json& JsonObject::operator[](std::string const & key) {
  return object_[key];
}

Json& JsonObject::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonObject::operator==(Value const& rhs) const {
  if (!IsA<JsonObject>(&rhs)) { return false; }
  return object_ == Cast<JsonObject const>(&rhs)->getObject();
}

Value& JsonObject::operator=(Value const &rhs) {
  JsonObject const* casted = Cast<JsonObject const>(&rhs);
  object_ = casted->getObject();
  return *this;
}

void JsonObject::Save(JsonWriter* writer) {
  writer->Visit(this);
}

// Json String
Json& JsonString::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonString::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer."
             << "  Please try obtaining std::string first.";
  return DummyJsonObject();
}

bool JsonString::operator==(Value const& rhs) const {
  if (!IsA<JsonString>(&rhs)) { return false; }
  return Cast<JsonString const>(&rhs)->getString() == str_;
}

Value & JsonString::operator=(Value const &rhs) {
  JsonString const* casted = Cast<JsonString const>(&rhs);
  str_ = casted->getString();
  return *this;
}

// FIXME: UTF-8 parsing support.
void JsonString::Save(JsonWriter* writer) {
  writer->Visit(this);
}

// Json Array
Json& JsonArray::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonArray::operator[](int ind) {
  return vec_.at(ind);
}

bool JsonArray::operator==(Value const& rhs) const {
  if (!IsA<JsonArray>(&rhs)) { return false; }
  auto& arr = Cast<JsonArray const>(&rhs)->getArray();
  return std::equal(arr.cbegin(), arr.cend(), vec_.cbegin());
}

Value & JsonArray::operator=(Value const &rhs) {
  JsonArray const* casted = Cast<JsonArray const>(&rhs);
  vec_ = casted->getArray();
  return *this;
}

void JsonArray::Save(JsonWriter* writer) {
  writer->Visit(this);
}

// Json raw
Json& JsonRaw::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonRaw::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonRaw::operator==(Value const& rhs) const {
  if (!IsA<JsonRaw>(&rhs)) { return false; }
  auto& arr = Cast<JsonRaw const>(&rhs)->getRaw();
  return std::equal(arr.cbegin(), arr.cend(), str_.cbegin());
}

Value & JsonRaw::operator=(Value const &rhs) {
  auto const* casted = Cast<JsonRaw const>(&rhs);
  str_ = casted->getRaw();
  return *this;
}

void JsonRaw::Save(JsonWriter* writer) {
  writer->Visit(this);
}

// Json Number
Json& JsonNumber::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonNumber::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonNumber::operator==(Value const& rhs) const {
  if (!IsA<JsonNumber>(&rhs)) { return false; }
  return number_ == Cast<JsonNumber const>(&rhs)->getNumber();
}

Value & JsonNumber::operator=(Value const &rhs) {
  JsonNumber const* casted = Cast<JsonNumber const>(&rhs);
  number_ = casted->getNumber();
  return *this;
}

void JsonNumber::Save(JsonWriter* writer) {
  writer->Visit(this);
}

// Json Null
Json& JsonNull::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonNull::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonNull::operator==(Value const& rhs) const {
  if (!IsA<JsonNull>(&rhs)) { return false; }
  return true;
}

Value & JsonNull::operator=(Value const &rhs) {
  Cast<JsonNull const>(&rhs);  // Checking only.
  return *this;
}

void JsonNull::Save(JsonWriter* writer) {
  writer->Write("null");
}

// Json Boolean
Json& JsonBoolean::operator[](std::string const & key) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by string.";
  return DummyJsonObject();
}

Json& JsonBoolean::operator[](int ind) {
  LOG(FATAL) << "Object of type "
             << Value::TypeStr() << " can not be indexed by Integer.";
  return DummyJsonObject();
}

bool JsonBoolean::operator==(Value const& rhs) const {
  if (!IsA<JsonBoolean>(&rhs)) { return false; }
  return boolean_ == Cast<JsonBoolean const>(&rhs)->getBoolean();
}

Value & JsonBoolean::operator=(Value const &rhs) {
  JsonBoolean const* casted = Cast<JsonBoolean const>(&rhs);
  boolean_ = casted->getBoolean();
  return *this;
}

void JsonBoolean::Save(JsonWriter *writer) {
  writer->Visit(this);
}

size_t constexpr JsonReader::kMaxNumLength;

Json JsonReader::Parse() {
  while (true) {
    SkipSpaces();
    char c = PeekNextChar();
    if (c == -1) { break; }

    if (c == '{') {
      return ParseObject();
    } else if ( c == '[' ) {
      return ParseArray();
    } else if ( c == '-' || std::isdigit(c) ) {
      return ParseNumber();
    } else if ( c == '\"' ) {
      return ParseString();
    } else if ( c == 't' || c == 'f' ) {
      return ParseBoolean();
    } else {
      Error("Unknown construct");
    }
  }
  return Json();
}

Json JsonReader::Load() {
  Json result = Parse();
  return result;
}

void JsonReader::Error(std::string msg) const {
  // just copy it.
  std::istringstream str_s(raw_str_.substr(0, raw_str_.size()));

  msg += ", at ("
         + std::to_string(cursor_.Line()) + ", "
         + std::to_string(cursor_.Col()) + ")\n";
  std::string line;
  int line_count = 0;
  while (std::getline(str_s, line) && line_count < cursor_.Line()) {
    line_count++;
  }
  msg+= line += '\n';
  std::string spaces(cursor_.Col(), ' ');
  msg+= spaces + "^\n";

  LOG(FATAL) << msg;
}

// Json class
void JsonReader::SkipSpaces() {
  while (cursor_.Pos() < raw_str_.size()) {
    char c = raw_str_[cursor_.Pos()];
    if (std::isspace(c)) {
      cursor_.Forward(c);
    } else {
      break;
    }
  }
}

void ParseStr(std::string const& str) {
  size_t end = 0;
  for (size_t i = 0; i < str.size(); ++i) {
    if (str[i] == '"' && i > 0 && str[i-1] != '\\') {
      end = i;
      break;
    }
  }
  std::string result;
  result.resize(end);
}

Json JsonReader::ParseString() {
  char ch = GetChar('\"');
  std::ostringstream output;
  std::string str;
  while (true) {
    ch = GetNextChar();
    if (ch == '\\') {
      char next = static_cast<char>(GetNextChar());
      switch (next) {
        case 'r':  str += u8"\r"; break;
        case 'n':  str += u8"\n"; break;
        case '\\': str += u8"\\"; break;
        case 't':  str += u8"\t"; break;
        case '\"': str += u8"\""; break;
        case 'u':
          str += ch;
          str += 'u';
          break;
        default: Error("Unknown escape");
      }
    } else {
      if (ch == '\"') break;
      str += ch;
    }
    if (ch == EOF || ch == '\r' || ch == '\n') {
      Expect('\"', ch);
    }
  }
  return Json(std::move(str));
}

Json JsonReader::ParseArray() {
  std::vector<Json> data;

  char ch = GetChar('[');
  while (true) {
    if (PeekNextChar() == ']') {
      GetChar(']');
      return Json(std::move(data));
    }
    auto obj = Parse();
    data.push_back(obj);
    ch = GetNextNonSpaceChar();
    if (ch == ']') break;
    if (ch != ',') {
      Expect(',', ch);
    }
  }

  return Json(std::move(data));
}

Json JsonReader::ParseObject() {
  char ch = GetChar('{');

  std::map<std::string, Json> data;
  if (ch == '}') return Json(std::move(data));

  while (true) {
    SkipSpaces();
    ch = PeekNextChar();
    if (ch != '"') {
      Expect('"', ch);
    }
    Json key = ParseString();

    ch = GetNextNonSpaceChar();

    if (ch != ':') {
      Expect(':', ch);
    }

    Json value;
    if (!ignore_specialization_ && (getRegistry().find(get<String>(key)) != getRegistry().cend())) {
      value = getRegistry().at(get<String>(key))(raw_str_, &(cursor_.pos_));
    } else {
      value = Parse();
    }

    // Json value {parse()};
    data[get<JsonString>(key)] = std::move(value);

    ch = GetNextNonSpaceChar();

    if (ch == '}') break;
    if (ch != ',') {
      Expect(',', ch);
    }
  }

  return Json(std::move(data));
}

Json JsonReader::ParseNumber() {
  std::string substr = raw_str_.substr(cursor_.Pos(), 17);
  size_t pos = 0;
  double number = std::stod(substr, &pos);
  for (size_t i = 0; i < pos; ++i) {
    GetNextChar();
  }
  return Json(number);
}

Json JsonReader::ParseBoolean() {
  bool result = false;
  char ch = GetNextNonSpaceChar();
  std::string const t_value = u8"true";
  std::string const f_value = u8"false";
  std::string buffer;

  if (ch == 't') {
    for (size_t i = 0; i < 3; ++i) {
      buffer.push_back(GetNextNonSpaceChar());
    }
    if (buffer != u8"rue") {
      Error("Expecting boolean value \"true\".");
    }
    result = true;
  } else {
    for (size_t i = 0; i < 4; ++i) {
      buffer.push_back(GetNextNonSpaceChar());
    }
    if (buffer != u8"alse") {
      Error("Expecting boolean value \"false\".");
    }
    result = false;
  }
  return Json{JsonBoolean{result}};
}

Json Json::Load(StringView str, bool ignore_specialization) {
  JsonReader reader(str, ignore_specialization);
  common::Timer t;
  t.Start();
  Json json{reader.Load()};
  t.Stop();
  t.PrintElapsed("Json::load");
  return json;
}

Json Json::Load(JsonReader* reader) {
  common::Timer t;
  t.Start();
  Json json{reader->Load()};
  t.Stop();
  t.PrintElapsed("Json::load");
  return json;
}

void Json::Dump(Json json, std::ostream *stream, bool pretty) {
  JsonWriter writer(stream, true);
  common::Timer t;
  t.Start();
  writer.Save(json);
  t.Stop();
  t.PrintElapsed("Json::dump");
}

Json& Json::operator=(Json const &other) {
  auto type = other.GetValue().Type();
  switch (type) {
  case Value::ValueKind::Array:
    ptr_.reset(new JsonArray(*Cast<JsonArray const>(&other.GetValue())));
    break;
  case Value::ValueKind::Boolean:
    ptr_.reset(new JsonBoolean(*Cast<JsonBoolean const>(&other.GetValue())));
    break;
  case Value::ValueKind::Null:
    ptr_.reset(new JsonNull(*Cast<JsonNull const>(&other.GetValue())));
    break;
  case Value::ValueKind::Number:
    ptr_.reset(new JsonNumber(*Cast<JsonNumber const>(&other.GetValue())));
    break;
  case Value::ValueKind::Object:
    ptr_.reset(new JsonObject(*Cast<JsonObject const>(&other.GetValue())));
    break;
  case Value::ValueKind::String:
    ptr_.reset(new JsonString(*Cast<JsonString const>(&other.GetValue())));
    break;
  default:
    LOG(FATAL) << "Unknown value kind.";
  }
  return *this;
}

std::string LoadFile(std::string fname) {
  auto OpenErr = [&fname]() {
                   std::string msg;
                   msg = "Opening " + fname + " failed: ";
                   msg += strerror(errno);
                   LOG(FATAL) << msg;
                 };
  auto ReadErr = [&fname]() {
                   std::string msg {"Error in reading file: "};
                   msg += fname;
                   msg += ": ";
                   msg += strerror(errno);
                   LOG(FATAL) << msg;
                 };

  std::string buffer;
#if defined(__unix__)
  struct stat fs;
  if (stat(fname.c_str(), &fs) != 0) {
    OpenErr();
  }

  size_t f_size_bytes = fs.st_size;
  buffer.resize(f_size_bytes+1);
  int32_t fd = open(fname.c_str(), O_RDONLY);
  posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
  ssize_t bytes_read = read(fd, &buffer[0], f_size_bytes);
  if (bytes_read < 0) {
    close(fd);
    ReadErr();
  }
  close(fd);
#else
  FILE *f = fopen(fname.c_str(), "r");
  if (f == NULL) {
    std::string msg;
    OpenErr();
  }
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  buffer.resize(fsize + 1);
  fread(&buffer[0], 1, fsize, f);
  fclose(f);
#endif  // defined(__unix__)
  return buffer;
}


}  // namespace xgboost
