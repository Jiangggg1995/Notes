# Preface

Bazel is an open source build tool for multiple computer languages. This tutorial is written for a beginner who want to use bazel to build a C/C++ project.

 Most content of this tutorial is copied from the [bazel's website](https://www.bazel.build/). I write this tutorial because I don't like the website's complex structure. If you don't like it too, hope you can enjoy this one.

Author: Yunheng Jiang

Email: yunheng_jiang@outlook.com

# Contents

[TOC]



# Bazel Overview

Bazel can be used to build

If you want to use bazel, firstly [install](https://docs.bazel.build/versions/4.0.0/install.html) bazel follow the instruction.

When running a build, Bazel does the following:

1. **Loads** the `BUILD` file relevant to the target.
2. **Analyzes** the inputs and their dependencies.
3. **Executes ** the build actions.

# Concepts

## Workspace

Usually there is a text file `WORKSPACE`  which may be empty in the root directory to identify a *workspace*. Bazel will work in a workspace.

## Packages

A package is a collection of related files and a specification of the dependencies among them. A package is defined as a directory containing a file named `BUILD`.

## Targets

Elements in packages called *targets*.

## Labels

The name of a target is called its *labels*.

eg. `@my_workspace_name//my/src/main:my_binary`

This is a label point to target *my_binary* which in my workspace's *my/src/main/BUILD* .

## Rules

Rules describe the relation between inputs' file and outputs' file and how to build.

eg. 

```
cc_binary(
	name = "my_app",
	srcs = ["my_app.cc"],
	deps = ["//absl/strings"],
)
```

*cc_binary* is a macro to build a c/c++ object. this example will use compiler to build *my_app.cc* which depends on *absl/strings*.

## BUILD

*BUILD* identify a unit of some source file called *Package*. *Rules* are defined in *BUILD*. Bazel extension are files ending in *.bzl*. Use *load* statement to import a symbol from an extension.

`load("@//foo/bar/:file.bzl", "some_library")`

This code will load the file `foo/bar/file.bzl` and add `some_library` symbol to the building environment.  

# Example

Here is an example for bazel building c++.