COPTS = ["-DUSE_SSL=1"] + select({
    "@platforms//os:windows": [
        "-D_CRT_DECLARE_NONSTDC_NAMES=0",  # don't define off_t, to avoid conflicts
        "-D_WIN32",
        "-DOPENSSL_IS_BORINGSSL",
        "-DWIN32_LEAN_AND_MEAN"
    ],
    "//conditions:default": [
    ],
}) + select({
    "@//:msvc-cl": [
    ],
    "//conditions:default": [
        # Old versions of GCC (e.g. 4.9.2) can fail to compile Redis's C without this.
        "-std=c99",
    ],
})

LOPTS = select({
    "@platforms//os:windows": [
        "-DefaultLib:" + "Crypt32.lib",
    ],
    "//conditions:default": [
    ],
})

# This library is for internal hiredis use, because hiredis assumes a
# different include prefix for itself than external libraries do.
cc_library(
    name = "_hiredis",
    hdrs = [
        "dict.c",
    ],
    copts = COPTS,
)

cc_library(
    name = "hiredis",
    srcs = glob(
        [
            "*.c",
            "*.h",
        ],
        exclude =
        [
            "test.c",
        ],
    ),
    hdrs = glob([
        "*.h",
        "adapters/*.h",
    ]),
    includes = [
        ".",
    ],
    copts = COPTS,
    linkopts = LOPTS,
    include_prefix = "hiredis",
    deps = [
        ":_hiredis",
        "@boringssl//:ssl",
        "@boringssl//:crypto"
    ],
    visibility = ["//visibility:public"],
)
