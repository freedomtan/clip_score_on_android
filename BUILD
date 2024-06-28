load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//:build_def.bzl", "android_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "clip_score",
    srcs = ["clip_score.cc"],
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)
