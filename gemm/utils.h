#pragma once

#include <fstream>
#include <ostream>
#include <sstream>
#include <string>

#include "gemm/matrix.h"

inline std::string read_file(const std::string& path)
{
  std::ifstream file(path);

  if (!file) {
    throw std::runtime_error("Cannot read file");
  }

  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  file.ignore(std::numeric_limits<std::streamsize>::max());

  try {
    auto size = file.gcount();

    if (size > 0x10000)  // 64kib sanity check for shaders:
      return std::string();

    file.clear();
    file.seekg(0, std::ios_base::beg);

    std::stringstream sstr;
    sstr << file.rdbuf();
    file.close();

    return sstr.str();
  } catch (const std::ifstream::failure& e) {
    throw std::runtime_error("cannot read file: " + path + " (" + e.what()
                             + ")");
  }
}

// https://stackoverflow.com/questions/25201131/writing-csv-files-from-c
// https://gist.github.com/rudolfovich/f250900f1a833e715260a66c87369d15
class CSVWriter
{
  std::ofstream fs_;
  bool is_first_;
  const std::string separator_;
  const std::string escape_seq_;
  const std::string special_chars_;

public:
  CSVWriter(const std::string filename, const std::string separator = ",")
      : fs_()
      , is_first_(true)
      , separator_(separator)
      , escape_seq_("\"")
      , special_chars_("\"")
  {
    fs_.exceptions(std::ios::failbit | std::ios::badbit);
    fs_.open(std::string(OUTPUTS_PATH) + filename,
             std::fstream::in | std::fstream::out | std::fstream::trunc);
  }

  ~CSVWriter()
  {
    flush();
    fs_.close();
  }

  void flush() { fs_.flush(); }

  void endrow()
  {
    fs_ << std::endl;
    is_first_ = true;
  }

  CSVWriter& operator<<(CSVWriter& (*val)(CSVWriter&)) { return val(*this); }

  CSVWriter& operator<<(const char* val) { return write(escape(val)); }

  CSVWriter& operator<<(const std::string& val) { return write(escape(val)); }

  template <typename T>
  CSVWriter& operator<<(const T& val)
  {
    return write(val);
  }

private:
  template <typename T>
  CSVWriter& write(const T& val)
  {
    if (!is_first_) {
      fs_ << separator_;
    } else {
      is_first_ = false;
    }
    fs_ << val;
    return *this;
  }

  std::string escape(const std::string& val)
  {
    std::ostringstream result;
    result << '"';
    std::string::size_type to, from = 0u, len = val.length();
    while (from < len
           && std::string::npos
               != (to = val.find_first_of(special_chars_, from)))
    {
      result << val.substr(from, to - from) << escape_seq_ << val[to];
      from = to + 1;
    }
    result << val.substr(from) << '"';
    return result.str();
  }
};

inline static CSVWriter& endrow(CSVWriter& file)
{
  file.endrow();
  return file;
}

inline static CSVWriter& flush(CSVWriter& file)
{
  file.flush();
  return file;
}

inline float matmul_time_to_gflops(float rows,
                                   float cols,
                                   float inner_dim,
                                   float microsecs)
{
  float FLOPS = 2 * rows * cols * inner_dim;
  return FLOPS / (microsecs * 1e6);
}

inline void matmul_cpu(const Matrix& A, const Matrix& B, Matrix& C)
{
  assert(A.cols == A.rows);  // assume square matrix
  assert(B.cols == B.rows);  // assume square matrix
  assert(A.cols == B.rows);

  assert(C.cols == B.cols);
  assert(C.rows = A.rows);

  uint M = C.rows;
  uint N = C.cols;
  uint K = A.cols;

  uint LDA = K;
  uint LDB = N;
  uint LDC = N;

  for (uint i = 0; i < M; ++i) {
    for (uint j = 0; j < N; ++j) {
      for (uint p = 0; p < K; ++p) {
        C[i * LDC + j] += (A[i * LDA + p] * B[p * LDB + j]);
      }
    }
  }
}