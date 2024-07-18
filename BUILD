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

cc_binary(
    name = "read_tfrecord",
    srcs = ["read_tfrecord.cc"],
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ] + select({
        "@org_tensorflow//tensorflow:android": [
            "@org_tensorflow//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core:lib",
            "@org_tensorflow//tensorflow/core:protos_all_cc",
        ],
    }),
)
