# Bazel: add include path

description for adding include path while using bazel

- Make all include paths relative to the workspace directory
- Use quoted includes `#include "foo.h"` for non-system headers
- Avoiding using Unix directory shortcuts, such as `.` for current dir or `..` for parent dir
- use the `inlcude_prefix` and `strip_include_prefix` arguments on the `cc_library` rule target
- All header files thar are used in the build must be declared in the `hdrs` or `srcs` of `cc_*` rules. This is enforced

