#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gf {

// Memory map arrays of type T to a flat file
// create(path, n): make a new n-element file, mapped read-write so we can fill it in. 
//      Call finalize_read_only() once done writing.
// open_existing(path, n): reopen a previously-written file of exactly n elements, mapped read-only. 
//      Used to skip recomputing stuff on a resumed run.
template<class T>
class MappedArray {
    static_assert(std::is_trivially_copyable<T>::value, "MappedArray<T> requires a trivially copyable T");
public:
    MappedArray() = default;
    MappedArray(const MappedArray&) = delete;
    MappedArray& operator=(const MappedArray&) = delete;
    MappedArray(MappedArray&& other) noexcept { *this = std::move(other); }
    MappedArray& operator=(MappedArray&& other) noexcept {
        if (this != &other) {
            reset();
            fd_ = other.fd_; ptr_ = other.ptr_; n_ = other.n_; bytes_ = other.bytes_;
            other.fd_ = -1; other.ptr_ = nullptr; other.n_ = 0; other.bytes_ = 0;
        }
        return *this;
    }
    ~MappedArray() { reset(); }

    // Create a brand new backing file of n elements, mapped read/write
    static MappedArray create(const std::string& path, size_t n) {
        MappedArray arr;
        arr.n_ = n;
        arr.bytes_ = n * sizeof(T);
        arr.fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (arr.fd_ < 0) throw std::runtime_error("MappedArray: could not create " + path);
        if (arr.bytes_ > 0) {
            if (::ftruncate(arr.fd_, static_cast<off_t>(arr.bytes_)) != 0) {
                ::close(arr.fd_);
                throw std::runtime_error("MappedArray: ftruncate failed for " + path);
            }
            void* p = ::mmap(nullptr, arr.bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, arr.fd_, 0);
            if (p == MAP_FAILED) {
                ::close(arr.fd_);
                throw std::runtime_error("MappedArray: mmap (create) failed for " + path);
            }
            arr.ptr_ = static_cast<T*>(p);
        }
        return arr;
    }

    // Reopen an existing backing file of exactly n elements, read-only
    // Returns false (rather than throwing) if the file is missing or doesn't match the expected size, so we fall back to recomputing
    static bool open_existing(const std::string& path, size_t n, MappedArray& out) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (::fstat(fd, &st) != 0) { ::close(fd); return false; }
        size_t expected_bytes = n * sizeof(T);
        if (static_cast<size_t>(st.st_size) != expected_bytes) {
            ::close(fd);
            return false; // stale/mismatched cache: recompute
        }
        out.reset();
        out.n_ = n;
        out.bytes_ = expected_bytes;
        out.fd_ = fd;
        if (expected_bytes > 0) {
            void* p = ::mmap(nullptr, expected_bytes, PROT_READ, MAP_SHARED, fd, 0);
            if (p == MAP_FAILED) { ::close(fd); out.fd_ = -1; return false; }
            out.ptr_ = static_cast<T*>(p);
        }
        return true;
    }

    // Flush + drop write access once fully populated (ideally should rm them too. something to add another time)
    void finalize_read_only() {
        if (ptr_ && bytes_ > 0) {
            ::msync(ptr_, bytes_, MS_SYNC);
            ::mprotect(ptr_, bytes_, PROT_READ);
        }
    }

    T&       operator[](size_t i)       { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }
    T*       data()       { return ptr_; }
    const T* data() const { return ptr_; }
    size_t   size() const { return n_; }
    bool     empty() const { return n_ == 0; }

private:
    void reset() {
        if (ptr_ && bytes_ > 0) ::munmap(ptr_, bytes_);
        if (fd_ >= 0) ::close(fd_);
        ptr_ = nullptr; fd_ = -1; n_ = 0; bytes_ = 0;
    }
    int fd_ = -1;
    T* ptr_ = nullptr;
    size_t n_ = 0;
    size_t bytes_ = 0;
};

// Needed so that kdtree_search can accept either a std::vector<Vec3> or a MappedArray<Vec3>
struct Vec3View {
    const Vec3* ptr = nullptr;
    size_t n = 0;
    Vec3View() = default;
    Vec3View(const Vec3* p, size_t count) : ptr(p), n(count) {}
    Vec3View(const std::vector<Vec3>& v) : ptr(v.data()), n(v.size()) {}
    Vec3View(const MappedArray<Vec3>& m) : ptr(m.data()), n(m.size()) {}
    const Vec3& operator[](size_t i) const { return ptr[i]; }
    size_t size() const { return n; }
};

} // namespace gf
